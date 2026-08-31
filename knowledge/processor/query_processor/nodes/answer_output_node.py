import time

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT_TEMPLATE,
)
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.evidence_packing_util import coverage_ordered_documents
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import set_task_result


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1 获取用户的问题（重写后）
        user_query = state.get("rewritten_query")

        # 获取task_id
        task_id = state.get("task_id") or ""

        # 2 获取answer
        answer = state.get("answer") or ""

        # 3 获取是否进行流式输出
        is_stream = bool(state.get("is_stream"))

        # 4 判断是否有answer
        if answer:
            # 表示在答案输出节点之前就有答案了，将已有答案输出即可
            self._push_exist_answer(answer, is_stream, task_id)
        else:
            # 说明在答案输出节点之前，没有生成答案，那么要在答案输出节点中调用大模型来输出答案
            # 组装提示词
            prompt = self._build_answer_prompt(state, self.config.max_context_chars)

            # 调用LLM生成答案
            llm_answer = self._generate_answer(prompt, state)
            state["answer"] = llm_answer

            # 流式输出完，要告诉前端SSE通道关闭了
            if is_stream:
                push_sse_event(task_id,SSEEvent.FINAL,{})

        # 保存历史会话到MongoDb
        self._save_to_mongo_db(state)
        
        return state

    # 输出已有答案
    def _push_exist_answer(self, answer: str, is_stream: bool, task_id: str):
        # 要是流式输出
        if is_stream:
            # 使用SSE队列
            push_sse_event(task_id, SSEEvent.FINAL, {"answer": answer})
        else:
            # 非流式输出
            set_task_result(task_id, "answer", answer)

    # 构建答案生成的上下文
    def _build_answer_prompt(self, state, max_context_chars):
        # max_context_chars 统一约束“检索文档 + 历史对话”的动态上下文长度。
        max_context_chars = max(int(max_context_chars or 0), 0)

        # 优先使用已经改写为可独立理解的问题。
        user_question = (
            state.get("rewritten_query")
            or state.get("original_query")
            or ""
        )
        theme_names = state.get("theme_names") or []
        history_messages = state.get("history") or []
        # 优先使用 Rerank 后按章节边界扩展的上下文；
        # 旧数据无法扩展时会自动回退到原重排结果。
        reranked_docs = (
            state.get("expanded_docs")
            or state.get("reranked_docs")
            or []
        )

        # 历史对话先只预留较小的预算，主要空间留给检索文档。
        # 文档不足时，未使用的空间会在下面重新分配给最近的历史记录。
        history_reserved_chars = min(2000, max_context_chars // 10)
        history_preview = self._format_recent_history(
            history_messages,
            history_reserved_chars,
        )
        document_budget = max_context_chars - len(history_preview)

        final_context_chunk_ids = []
        content = self._format_document_context(
            reranked_docs,
            document_budget,
            selected_chunk_ids=final_context_chunk_ids,
        )
        state["final_context_chunk_ids"] = final_context_chunk_ids

        # 文档使用后剩余的全部空间用于历史，并优先保留最近的消息。
        history_budget = max_context_chars - len(content)
        format_history_content = self._format_recent_history(
            history_messages,
            history_budget,
        )

        return ANSWER_USER_PROMPT_TEMPLATE.format(
            context=content,
            history=format_history_content,
            theme_names="、".join(theme_names),
            question=user_question,
        )

    def _format_document_context(
            self,
            reranked_docs,
            max_chars,
            selected_chunk_ids=None,
    ):
        """覆盖证据优先打包，并限制总预算及单篇证据长度。"""
        if max_chars <= 0:
            return ""

        content_list = []
        current_chars = 0

        ordered_documents = coverage_ordered_documents(reranked_docs)
        per_doc_max_chars = self.config.answer_max_chars_per_evidence

        for doc in ordered_documents:
            if not isinstance(doc, dict):
                continue

            # 获取并校验正文
            doc_content = (doc.get("content") or "").strip()
            if not doc_content:
                continue

            source = doc.get("source")
            title = (doc.get("title") or "未命名资料").strip()

            # 使用有效文档数量编号，避免出现资料1、资料3
            document_number = len(content_list) + 1

            if source == "local":
                chunk_id = doc.get("chunk_id")
                file_title = (
                    doc.get("canonical_title")
                    or doc.get("file_title")
                    or doc.get("theme_name")
                    or "未命名文档"
                )
                section_path = (
                    doc.get("section_path")
                    or doc.get("parent_title")
                    or title
                )
                expanded_chunk_ids = doc.get("expanded_chunk_ids") or [chunk_id]

                document_header = (
                    f"【资料{document_number}】\n"
                    f"来源：本地知识库\n"
                    f"文档：{file_title}\n"
                    f"章节：{section_path}\n"
                    f"命中标题：{title}\n"
                    f"切片ID：{expanded_chunk_ids}\n"
                    f"正文：\n"
                )

            else:
                continue

            # 计算还能放多少字符
            separator = "\n\n" if content_list else ""
            remaining_chars = (
                    max_chars
                    - current_chars
                    - len(separator)
                    - len(document_header)
            )

            if remaining_chars <= 0:
                break

            # 按剩余字符数截取正文
            truncated_content = doc_content[:min(remaining_chars, per_doc_max_chars)]
            document_block = document_header + truncated_content

            content_list.append(document_block)
            current_chars += len(separator) + len(document_block)
            if selected_chunk_ids is not None:
                selected_chunk_ids.append(doc.get("chunk_id"))

        return "\n\n".join(content_list)

    def _format_recent_history(self, history_messages, max_chars):
        """截取最近的历史消息，并恢复成从旧到新的阅读顺序。

        历史消息使用MongoDB的字典结构；如果包含ts字段，会先按时间
        从旧到新排序。
        """
        if max_chars <= 0 or not history_messages:
            return ""

        ordered_messages = list(history_messages)
        if all(
                isinstance(message, dict)
                and isinstance(message.get("ts"), (int, float))
                for message in ordered_messages
        ):
            ordered_messages.sort(key=lambda message: message["ts"])

        role_mapping = {
            "user": "用户",
            "assistant": "助手",
        }
        selected_lines = []
        current_chars = 0

        # 从最新消息向前选，确保被截掉的是更早的历史。
        for message in reversed(ordered_messages):
            if not isinstance(message, dict):
                continue

            role_name = role_mapping.get(message.get("role"))
            text = (message.get("text") or "").strip()
            if not role_name or not text:
                continue
            line = f"{role_name}：{text}"

            if not line:
                continue

            separator_length = 1 if selected_lines else 0
            remaining_chars = max_chars - current_chars - separator_length
            if remaining_chars <= 0:
                break

            if len(line) > remaining_chars:
                selected_lines.append(line[:remaining_chars])
                break

            selected_lines.append(line)
            current_chars += separator_length + len(line)

        # 上面按“新到旧”选择，这里恢复为“旧到新”交给大模型。
        selected_lines.reverse()
        return "\n".join(selected_lines)




    #调用大模型生成答案
    def _generate_answer(self, prompt:str, state:QueryGraphState):
        task_id = state.get("task_id")
        #1 创建大模型对象
        try:
            llm_client = AIClients.get_llm_client(
                response_format=False,
                role="answer",
                thinking="light",
            )
        except ConnectionError as e:
            self.logger.error(f"Failed to connect to LLM server,{e}")
            return ("抱歉，智能问答服务当前暂时不可用，无法生成回答。"
                    "请稍后重试；如果问题持续出现，请联系系统管理员。")
        #2 判断是流式输出还是非流式输出，
        if state.get("is_stream"):
            answer = ""
            # 2.1 要是流式输出，需要边生成边推送SSE事件
            started_at = time.perf_counter()
            try:
                for chunk in llm_client.stream([
                        ("system", ANSWER_SYSTEM_PROMPT),
                        ("user", prompt)]):
                    content = chunk.content
                    #chunk不为空，将增量推送至SSE事件中
                    if content:
                        push_sse_event(task_id,SSEEvent.DELTA,{"delta":content})

                    answer += content
            finally:
                self.logger.info(
                    "LLM latency: node_name=answer_output model_name=%s "
                    "thinking_level=light elapsed_ms=%.1f",
                    getattr(llm_client, "model_name", "unknown"),
                    (time.perf_counter() - started_at) * 1000,
                )
        else:
            # 2.2 要是非流式输出，直接调用大模型生成结果
            try:
                started_at = time.perf_counter()
                try:
                    llm_response = llm_client.invoke([
                        ("system", ANSWER_SYSTEM_PROMPT),
                        ("user", prompt)])
                finally:
                    self.logger.info(
                        "LLM latency: node_name=answer_output model_name=%s "
                        "thinking_level=light elapsed_ms=%.1f",
                        getattr(llm_client, "model_name", "unknown"),
                        (time.perf_counter() - started_at) * 1000,
                    )
            except Exception as e:
                self.logger.error(f"Failed to invoke the LLM server,{e}")
                return "抱歉，智能问答服务暂无答案生成"

            #将生成的答案放到任务队列中
            answer = llm_response.content
            if not answer:
                return "抱歉，智能问答服务暂无答案生成"
            set_task_result(task_id,"answer",answer)

        return answer

    def _save_to_mongo_db(self, state:QueryGraphState):


        #保存用户的对话内容
        save_chat_message(session_id=state.get("session_id"),
                          role="user",
                          text=(
                              state.get("display_query")
                              or state.get("original_query")
                          ),
                          rewritten_query=state.get("rewritten_query"),
                          theme_names=state.get("theme_names"),
                          selected_documents=state.get("selected_documents"))
        #保存AI的对话内容
        save_chat_message(session_id=state.get("session_id"),
                          role="assistant",
                          text=state.get("answer"),
                          theme_names=state.get("theme_names"),
                          selected_documents=state.get("selected_documents"))
        pass
