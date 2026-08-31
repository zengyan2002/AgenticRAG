import logging
from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_processor.state import ImportGraphState


class EntryNode(BaseNode):
    name = "entry_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        #1. 获取state中的import_file_path、file_dir，并且判断是否为空
        self.log_step(step_name="STEP1",message="获取state中的import_file_path、file_dir，并且判断是否为空")
        import_file_path = state["import_file_path"]
        file_dir = state["file_dir"]
        if not import_file_path:
            self.logger.error("import_file_path is None")
            raise StateFieldError(node_name=self.name,field_name="import_file_path",expected_type=str)

        if not file_dir:
            self.logger.error("file_dir is None")
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=str)

        #2. 判断import_file_path、file_dir是否真实存在
        self.log_step(step_name="STEP2", message="判断import_file_path、file_dir是否真实存在")
        import_file_path_obj = Path(import_file_path)
        file_dir_obj = Path(file_dir)
        if not import_file_path_obj.exists():
            self.logger.error("import_file_path not exist")
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=Path)

        if not file_dir_obj.exists():
            self.logger.error("file_dir not exist")
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=Path)

        #3 判断文件的类型
        self.log_step(step_name="STEP3", message="判断文件类型")
        #3.1 获取文件的后缀
        suffix = import_file_path_obj.suffix.lower()
        #3.2 判断文件的后缀
        if suffix == ".pdf":
            #如果是pdf文件，则设置is_pdf_read_enable为True，并设置pdf_path = import_file_path
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif suffix == ".md":
            # 如果是md文件，则设置is_md_enable为True，并设置md_path = import_file_path
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        elif suffix in {".doc", ".docx"}:
            # Word 先转成 PDF，再复用现有 MinerU 解析流程。
            state["is_word_read_enabled"] = True
            state["word_path"] = import_file_path
        else:
            # 如果是其它文件，直接抛出异常
            self.logger.error(f"unsupported suffix {suffix}")
            raise ValidationError(
                "仅支持 PDF、Markdown、DOC 和 DOCX 文件",
                node_name=self.name,
            )

        #4 获取不含后缀的文件标题，并且将得到的标题放入state中
        self.log_step(step_name="STEP4", message="获取不含后缀的文件标题，并且将得到的标题放入state中")
        file_title = import_file_path_obj.stem
        state["file_title"] = file_title


        return state
