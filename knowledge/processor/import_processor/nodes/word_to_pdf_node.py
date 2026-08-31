import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import (
    FileProcessingError,
    StateFieldError,
)
from knowledge.processor.import_processor.state import ImportGraphState


class WordToPdfNode(BaseNode):
    """Convert DOC/DOCX to PDF before handing the file to MinerU."""

    name = "word_to_pdf_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        word_path_value = state.get("word_path")
        if not word_path_value:
            raise StateFieldError(
                node_name=self.name,
                field_name="word_path",
                expected_type=str,
            )

        word_path = Path(word_path_value).resolve()
        if not word_path.is_file():
            raise FileProcessingError(
                f"Word 文件不存在: {word_path}",
                node_name=self.name,
            )

        output_dir = Path(state.get("file_dir") or word_path.parent).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / f"{word_path.stem}.pdf"

        libreoffice = self._find_libreoffice()
        if libreoffice:
            self._convert_with_libreoffice(word_path, output_dir, libreoffice)
        elif os.name == "nt":
            self._convert_with_microsoft_word(word_path, pdf_path)
        else:
            raise FileProcessingError(
                "未找到 LibreOffice。请安装 LibreOffice，或通过 "
                "LIBREOFFICE_PATH 指定 soffice 可执行文件。",
                node_name=self.name,
            )

        if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
            raise FileProcessingError(
                f"Word 转 PDF 失败，未生成有效文件: {pdf_path}",
                node_name=self.name,
            )

        state["pdf_path"] = str(pdf_path)
        state["is_pdf_read_enabled"] = True
        return state

    def _find_libreoffice(self) -> Optional[str]:
        configured_path = (self.config.libreoffice_path or "").strip()
        if configured_path:
            configured = Path(configured_path)
            if configured.is_file():
                return str(configured)
            raise FileProcessingError(
                f"LIBREOFFICE_PATH 指向的文件不存在: {configured}",
                node_name=self.name,
            )

        command = shutil.which("soffice") or shutil.which("libreoffice")
        if command:
            return command

        if os.name == "nt":
            candidates = (
                Path(os.environ.get("ProgramFiles", "C:/Program Files"))
                / "LibreOffice/program/soffice.exe",
                Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
                / "LibreOffice/program/soffice.exe",
            )
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)

        return None

    def _convert_with_libreoffice(
            self,
            word_path: Path,
            output_dir: Path,
            executable: str,
    ) -> None:
        command = [
            executable,
            "--headless",
            "--convert-to",
            "pdf:writer_pdf_Export",
            "--outdir",
            str(output_dir),
            str(word_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.word_convert_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise FileProcessingError(
                f"LibreOffice 转换 Word 失败: {exc}",
                node_name=self.name,
            ) from exc

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "未知错误").strip()
            raise FileProcessingError(
                f"LibreOffice 转换 Word 失败: {details}",
                node_name=self.name,
            )

    def _convert_with_microsoft_word(
            self,
            word_path: Path,
            pdf_path: Path,
    ) -> None:
        try:
            import pythoncom
            import win32com.client
        except ImportError as exc:
            raise FileProcessingError(
                "未找到 LibreOffice，且当前 Python 环境未安装 pywin32。",
                node_name=self.name,
            ) from exc

        pythoncom.CoInitialize()
        word_app = None
        document = None
        try:
            word_app = win32com.client.DispatchEx("Word.Application")
            word_app.Visible = False
            word_app.DisplayAlerts = 0
            document = word_app.Documents.Open(
                str(word_path),
                ReadOnly=True,
                AddToRecentFiles=False,
            )
            document.ExportAsFixedFormat(
                OutputFileName=str(pdf_path),
                ExportFormat=17,
                OpenAfterExport=False,
                OptimizeFor=0,
                CreateBookmarks=1,
            )
        except Exception as exc:
            raise FileProcessingError(
                f"Microsoft Word 转 PDF 失败: {exc}",
                node_name=self.name,
            ) from exc
        finally:
            if document is not None:
                document.Close(SaveChanges=False)
            if word_app is not None:
                word_app.Quit()
            pythoncom.CoUninitialize()
