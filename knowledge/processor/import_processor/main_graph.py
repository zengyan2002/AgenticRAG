"""文档入库流程主图。

使用 LangGraph 构建科研文档入库工作流。

流程结构::

    START
      │
      ▼
    entry_node
    文件检查与类型识别
      │
      ├── Markdown ───────────────────────────────────────┐
      │                                                   │
      ├── PDF ──▶ pdf_to_md_node ────────────────────────┤
      │              PDF 转 Markdown                      │
      │                                                   ▼
      └── Word ─▶ word_to_pdf_node ─▶ pdf_to_md_node ─▶ md_image_node
                    Word 转 PDF                         图片语义解析
                                                           │
                                                           ▼
                                                  document_split_node
                                                     层级感知切片
                                                           │
                                                           ▼
                                                 document_identity_node
                                                    文档身份建档
                                                           │
                                                           ▼
                                                  formula_semantic_node
                                                    公式语义增强
                                                           │
                                                           ▼
                                             bge_embedding_chunks_node
                                               BGE-M3 稠密/稀疏向量化
                                                           │
                                                           ▼
                                                  milvus_import_node
                                                   文档切片向量入库
                                                           │
                                                           ▼
                                                document_registry_node
                                                   文档注册表入库
                                                           │
                                                           ▼
                                                          END

说明：
    - Markdown 文件跳过格式转换，直接进入图片解析节点。
    - Word 文件先转为 PDF，再与 PDF 文件共用 Markdown 解析链路。
    - document_identity_node 已替代旧的独立主题识别节点，在生成
      doc_id、规范标题和别名的同时保留 primary_subject 主题信息。
"""

from langgraph.constants import START,END
from langgraph.graph import StateGraph

from knowledge.processor.import_processor.nodes.bge_embedding_chunks_node import BGEEmbeddingChunksNode
from knowledge.processor.import_processor.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.formula_semantic_node import FormulaSemanticNode
from knowledge.processor.import_processor.nodes.md_image_node import MdImageNode
from knowledge.processor.import_processor.nodes.milvus_import_node import MilvusImportNode
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_processor.nodes.document_identity_node import DocumentIdentityNode
from knowledge.processor.import_processor.nodes.document_registry_node import DocumentRegistryNode
from knowledge.processor.import_processor.nodes.word_to_pdf_node import WordToPdfNode
from knowledge.processor.import_processor.state import ImportGraphState


#定义路由函数
def route_fun(state: ImportGraphState):
    #如果是md文件，则直接去
    if state.get("is_md_read_enabled"):
        return "md_image_node"

    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"

    if state.get("is_word_read_enabled"):
        return "word_to_pdf_node"

    raise ValueError("The imported file type has not been identified.")

def create_import_graph():
    #1 创建StateGraph，传入状态结构类型
    work_flow = StateGraph(ImportGraphState)

    #2 注册节点
    work_flow.add_node("entry_node", EntryNode())
    work_flow.add_node("pdf_to_md_node", PdfToMdNode())
    work_flow.add_node("word_to_pdf_node", WordToPdfNode())
    work_flow.add_node("md_image_node", MdImageNode())
    work_flow.add_node("document_split_node", DocumentSplitNode())
    work_flow.add_node("document_identity_node", DocumentIdentityNode())
    work_flow.add_node("formula_semantic_node", FormulaSemanticNode())
    work_flow.add_node("bge_embedding_chunks_node", BGEEmbeddingChunksNode())
    work_flow.add_node("milvus_import_node", MilvusImportNode())
    work_flow.add_node("document_registry_node", DocumentRegistryNode())


    #3 添加边
    work_flow.add_edge(START,"entry_node")
    work_flow.add_conditional_edges(
        "entry_node",
        route_fun,
        {
            "md_image_node": "md_image_node",
            "pdf_to_md_node": "pdf_to_md_node",
            "word_to_pdf_node": "word_to_pdf_node",
        },
    )
    work_flow.add_edge("word_to_pdf_node", "pdf_to_md_node")
    work_flow.add_edge("pdf_to_md_node","md_image_node")
    work_flow.add_edge("md_image_node","document_split_node")
    work_flow.add_edge("document_split_node","document_identity_node")
    work_flow.add_edge("document_identity_node","formula_semantic_node")
    work_flow.add_edge("formula_semantic_node","bge_embedding_chunks_node")
    work_flow.add_edge("bge_embedding_chunks_node","milvus_import_node")
    work_flow.add_edge("milvus_import_node","document_registry_node")
    work_flow.add_edge("document_registry_node",END)

    #4 编译
    compiled_work_flow = work_flow.compile()

    return compiled_work_flow

