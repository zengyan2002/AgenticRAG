import os
import re
import hashlib
from typing import Tuple, List, Dict, Any

import json

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.core.settings import get_settings
from knowledge.utils.formula_util import FormulaProcessor
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1 参数校验
        [md_content, file_title, min_content_length, max_content_length] = self._valid_state(state)

        # 2 按标题切分
        block_list_by_title = self._split_by_title(md_content, file_title)

        # 3 二次切分或者合并
        block_list_final = self._split_and_merge(block_list_by_title, min_content_length, max_content_length)

        # 4 将切分完成的内容组装成后续节点可直接使用的chunks
        chunks = self._assemble_chunks(block_list_final)

        # 调试时才落盘切片，生产环境默认只在图状态中传递。
        if get_settings().import_debug_artifacts:
            self._backup_chunks(chunks, state)

        # 6 将切片结果写回图状态，供下游节点使用
        state["chunks"] = chunks

        return state

    # 参数校验
    def _valid_state(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        # 1 获取md_content
        md_content = state["md_content"]

        # 2 统一换行符,先将"\r\n"替换成"\n\n"，再将"\r"替换成"\n"
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3 校验config中的切片的最小长度min_content_length和切片的最大长度max_content_length
        min_content_length = self.config.min_content_length
        max_content_length = self.config.max_content_length
        if min_content_length > max_content_length or min_content_length <= 0 or max_content_length <= 0:
            self.logger.error("")
            raise ValueError("切片长度参数配置错误")

        # 4 获取文档标题file_title（给后续步骤作为兜底使用）
        file_title = state["file_title"]

        # 5 返回上述内容   md_content、file_title、config.min_content_length、config.max_content_length
        return md_content, file_title, min_content_length, max_content_length

    '''
        按照标题对文档进行分割
        最终都得到一个切成的块列表，每一块的结构为一个字典，结构为
        {
            "body":[当前块正文的内容],
            "title":[当前块的标题],
            "parent_title":[当前块的父标题],
            "file_title":[文件标题]
        }
    '''

    def _split_by_title(self, md_content: str, file_title: str) -> List[Dict[str, Any]]:

        # 文本块列表
        block_list = []

        # 1 将md_content按行切分
        md_content_lines = md_content.split("\n")

        # 定义匹配标题的正则表达式title_patern = re.compile(r"^\s*(#{1,6})\s+(.+)")
        title_pattern = re.compile(r"^\s*(#{1,6})\s+(.+)")
        # 声明`hierarchy = [""] * 7`用于存储各级标题，目的是为了识别父标题
        hierarchy = [""] * 7  # 意思是创建一个长度为 7 的列表，里面每个元素都是空字符串

        # 默认不在代码块中
        in_code_block = False

        # 声明匹配代码块的正则表达式
        code_block_pattern = re.compile(r"^\s*```")

        # 存储当前块的内容
        current_body = []

        # 当前块的标题
        current_title = ""

        # 当前标题的层级
        current_level = 0

        # 声明一个内部方法用于收集某个标题对应的块的内容
        def _collection() -> None:
            #设置块文本内容
            body = "\n".join(current_body).strip()
            #如果bod
            if not body:
                return
            #设置标题
            if current_title:
                title = current_title
            else:
                title = file_title
            #设置父标题（找到距离当前标题最近的标题）
            parent_title = title

            if current_level>1:
                for level1 in range(current_level-1,0,-1):
                    if hierarchy[level1]:
                        parent_title = hierarchy[level1]
                        break

            section_titles = [
                item
                for item in hierarchy[1:current_level + 1]
                if item
            ]
            if not section_titles:
                section_titles = [title]
            section_path = " > ".join(section_titles)
            section_id = self._stable_section_id(file_title, section_path)
            parent_section_path = " > ".join(section_titles[:-1])
            parent_section_id = (
                self._stable_section_id(file_title, parent_section_path)
                if parent_section_path
                else ""
            )

            #组装
            block_list.append({
                "body": body,
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title,
                "section_title": title,
                "section_path": section_path,
                "section_id": section_id,
                "parent_section_id": parent_section_id,
                # 这里的“父块”指按Markdown标题切分得到、尚未递归拆分的原始章节块。
                "parent_summary": self._extract_parent_summary(body),
            })

        # 2 遍历md_content的每一行，判断是否为标题，排除代码块的干扰
        for index, md_line in enumerate(md_content_lines):
            # 判断当前行是否在代码块中
            if code_block_pattern.match(md_line):
                in_code_block = not in_code_block

            title_match = title_pattern.match(md_line)
            if title_match and not in_code_block:
                # 2.1 如果是标题行并且不在代码块中，则将上一个标题对应的内容收集起来
                _collection()
                # 设置当前标题的值
                current_title = title_match.group(2).strip()
                # 清空当前块的内容部分
                current_body = []
                # 将当前标题添加到hierarchy数组中，并且要记录当前标题的层级
                current_level = len(title_match.group(1))
                hierarchy[current_level] = current_title
                # 清空数组中当前标题下的所有子标题
                for level in range(current_level + 1, 7):
                    hierarchy[level] = ""

            else:
                # 2.2 如果不是标题行，则将当前行添加到当前标题的body中
                current_body.append(md_line)
        #收集最后一个标题块后无标题的部分
        _collection()

        return block_list

    @staticmethod
    def _stable_section_id(file_title: str, section_path: str) -> str:
        raw_value = f"{file_title}\x1f{section_path}".encode("utf-8")
        return hashlib.sha256(raw_value).hexdigest()[:24]

    @staticmethod
    def _extract_parent_summary(body: str, max_chars: int = 400) -> str:
        """为重排构建可追溯的抽取式父块摘要，不调用LLM。"""
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", str(body or ""))
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]

    def _split_and_merge(self, block_list_by_title:List[Dict[str, Any]],min_content_length:int, max_content_length:int):
        #声明一个存储block的最后结果
        block_list_current = []

        #遍历block列表对每一个block进行二次切分
        for block in block_list_by_title:
            #将长文本进一步分割
            split_blocks = self._split_long_block(block, max_content_length)
            #分割后的结果加入最终文本块列表
            block_list_current.extend(split_blocks)

        #合并短的文本块
        block_list_final = self._merge_short_block(block_list_current, min_content_length, max_content_length)

        return block_list_final


    '''
    {
            "body":[当前块正文的内容],
            "title":[当前块的标题],
            "parent_title":[当前块的父标题]
            "file_title":[文件标题]
        }
    '''
    #进一步分解长块
    def _split_long_block(self, block: Dict[str, Any],max_content_length:int) -> List[Dict[str, Any]]:
        #1 判断是否为长块，即是否超过了max_content_length
        #1.1 计算block的长度
        #1.1.1 获取block的title
        title = block.get("title")
        #如果标题太长，截取前80个字符
        if title and len(title)>80:
            title = title[:80]

        #在标题下拼接一个空行
        title_prefix = f"{title}\n\n"


        #1.1.2 获取block中的body
        body = block.get("body")

        # 表格线性化可能把公式中的竖线、对齐符号误认为表格分隔符。
        # 先隐藏公式，完成表格转换后再原样恢复。
        protected_body, formula_mapping = FormulaProcessor.protect(body)
        if "<table" in protected_body.lower() or "|" in protected_body:
            protected_body = MarkdownTableLinearizer.process(protected_body)
        body = FormulaProcessor.restore(protected_body, formula_mapping)
        block["body"] = body

        #1.1.3 获取总长度
        total_length = len(body)+len(title_prefix)

        #1.2 判断total_length是否大于max_content_length
        if total_length <= max_content_length:
            #不需要切分
            return [block]

        #当前block的长度超过了最大长度
        #计算切割文本body的最大长度
        max_body_length = max_content_length-len(title_prefix)
        #如果计算出的要切割文本的最大长度小于等于0，则直接返回[block]
        if max_body_length <= 0:
            return [block]

        # 将公式视为不可拆分单元。普通文本仍由
        # RecursiveCharacterTextSplitter 按原有分隔符处理。
        split_body_parts = FormulaProcessor.split_text_preserving_formulas(
            body,
            chunk_size=max_body_length,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

        split_blocks = []
        for index,split_body in enumerate(split_body_parts):
            split_blocks.append({
                "body": split_body,
                "title": f"{title}_{index+1}",
                "parent_title": block["parent_title"],
                "file_title": block["file_title"],
                "section_title": block.get("section_title", title),
                "section_path": block.get("section_path", title),
                "section_id": block.get("section_id", ""),
                "parent_section_id": block.get("parent_section_id", ""),
                "parent_summary": block.get("parent_summary", ""),
            })

        return split_blocks



    def _merge_short_block(self, block_list_current,min_content_length:int, max_content_length:int):

        if not block_list_current:
            return []

        current_block = block_list_current[0]

        final_block_list = []

        #遍历剩下的blocks
        for next_block in block_list_current[1:]:
            #当前块的标题
            current_title = current_block["title"]
            # 当前文本块的父标题
            current_parent_title = current_block["parent_title"]
            # 将现有标题拼接“\n\n”得到前缀标题
            current_title_prefix = f"{current_title}\n\n"

            # 只合并同一原始章节块内的短切片，避免破坏父子关系。
            is_same_section = (
                next_block.get("section_id")
                and next_block.get("section_id") == current_block.get("section_id")
            )
            #当前文本块的主体部分加上前缀标题的大小小于最小文本块要求
            is_less_than_min_length = (len(current_title_prefix)+len(current_block["body"])) < min_content_length
            #当前文本块的主体部分加上下一个文本块的长度加上公共前缀标题的大小小于最大文本块要求
            merge_length = len(f"{current_parent_title}\n\n")+len(f"{current_block['body'].rstrip()}\n\n")+len(f"{next_block['body']}")
            is_merge_less_than_max_length = merge_length <= max_content_length
            if is_same_section and is_less_than_min_length and is_merge_less_than_max_length:
                current_block["body"] = f"{current_block['body'].rstrip()}\n\n{next_block['body']}"
                current_block["title"] = current_block.get(
                    "section_title",
                    current_block["title"],
                )
            else:
                final_block_list.append(current_block)
                current_block = next_block

        #把最后一块加入
        final_block_list.append(current_block)

        return final_block_list

    def _assemble_chunks(self, block_list_final:List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        #把已经切好的 block 组装成 chunks
        '''
        {
                "content": content,
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title
            }
        :param block_list_final:
        :return:
        '''
        chunks = []
        section_chunk_indexes: Dict[str, int] = {}
        for chunk_index, block in enumerate(block_list_final):
            content = f"{block.get('title')}\n\n{block.get('body')}"
            title = block.get("title")
            parent_title = block.get("parent_title")
            file_title = block.get("file_title")
            formulas = FormulaProcessor.extract(content)
            formula_search_text = FormulaProcessor.build_search_text(formulas)
            section_id = str(block.get("section_id") or "")
            section_chunk_index = section_chunk_indexes.get(section_id, 0)
            section_chunk_indexes[section_id] = section_chunk_index + 1
            chunks.append({
                "content": content,
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title,
                "section_title": block.get("section_title") or title,
                "section_path": block.get("section_path") or title,
                "section_id": section_id,
                "parent_section_id": block.get("parent_section_id") or "",
                "parent_summary": block.get("parent_summary") or "",
                "chunk_index": chunk_index,
                "section_chunk_index": section_chunk_index,
                "has_formula": bool(formulas),
                "formulas": formulas,
                "formula_search_text": formula_search_text,
            })
        return chunks

    def _backup_chunks(self, chunks:List[Dict[str,Any]], state:ImportGraphState):
        #保存的文件目录
        file_dir = state.get("file_dir")
        if not file_dir:
            return

        #创建目录
        os.makedirs(file_dir, exist_ok=True)
        try:
            #文件路径
            backup_file_path = os.path.join(file_dir, "chunks.json")

            with open(backup_file_path, "w", encoding="utf-8") as f:
                json.dump(chunks,f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"Failed to back up split results, but this does not affect the overall process: {e}")
