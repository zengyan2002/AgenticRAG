import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional,Dict

import base64
from openai import OpenAI

from knowledge.processor.import_processor.config import ImportConfig, get_config
from knowledge.core.settings import get_settings


from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import StateFieldError, FileProcessingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients


#定义一个类来表示图片的信息属性
@dataclass
class ImageContext:
    #图片的名称
    above_title:str
    above_context_content:str
    following_context_content: str

@dataclass
class ImageInfo:
    #图片的名称
    image_name:str
    image_path:str
    image_context: ImageContext



#处理md文件的类
class _MdFileHandler:

    def __init__(self,logger:logging.Logger,node_name:str):
        self.logger = logger
        self.node_name = node_name

    def read_md(self, md_path:str)-> Tuple[str,Path,Path]:
        #1 判断md_path是否为空
        if not md_path:
            self.logger.error(f"md_path is empty")
            raise StateFieldError(node_name=self.node_name,field_name="md_path",expected_type=str)

        #2 判断md_path是否真实存在
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            self.logger.error(f"md_path is not exist")
            raise FileProcessingError(f"md_path {md_path} does not exist",node_name=self.node_name)

        #3 读取md文件到内存中
        with open(md_path_obj,"r",encoding="UTF-8") as f:
            md_content = f.read()

        #4 获取md文件的图片的目录路径
        image_dir_obj = md_path_obj.parent/"images"

        #5 返回md_content,md_path_obj,image_dir_obj
        return md_content,md_path_obj,image_dir_obj

     # 备份md文件
    def _backup(self, md_path_obj: Path, new_md_content: str) -> str:
        # 1. 新文件的路径
        backup_md_path = md_path_obj.with_name(md_path_obj.stem + "_backup" + md_path_obj.suffix)
        # 2. 将新的md内容写入新的文件中
        try:
            with open(backup_md_path, 'w', encoding='utf-8') as f:
                f.write(new_md_content)
            self.logger.info(f"md文件已备份到{backup_md_path}")
        except Exception as e:
            self.logger.warning(f"备份md文件失败: {str(e)}")

        return str(backup_md_path)

#用来提取图片信息，包含提取上下文
class _ImageScanner:
    def __init__(self, logger: logging.Logger, node_name: str, config: Optional[ImportConfig] = None):
        self.logger = logger
        self.node_name = node_name
        self.config = config or get_config()

    def scan_image_dir(self,md_content:str,image_dir_obj:Path) ->List[ImageInfo]:
        image_info_list = []
        for image_file in image_dir_obj.iterdir():
            #1 过滤子目录
            if not image_file.is_file():
                continue

            #2 过滤非图片文件
            if not image_file.suffix in (".png",".jpg",".jpeg",".jpeg",".gif",".bmp",".svg",".png",".webp"):
                continue

            #3 在md_content中匹配到图片所在的行
            #3.1 将md_content按行进行切分
            md_content_lines = md_content.split("\n")

            # 声明匹配md图片的正则表达式
            image_pattern = re.compile(rf"!\[.*?\]\(.*?{re.escape(image_file.name)}.*?\)")

            #默认没在代码块中
            in_code_block = False

            #声明匹配代码块的正则表达式
            code_block_pattern = re.compile(r"^\s*```")

            #声明匹配标题的正则表达式（一级标题、二级标题、三级标题）
            title_pattern = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")

            #3.2 遍历每一行，对图片进行匹配
            for index,md_content_line in enumerate(md_content_lines):
                # 3.2.1 判断图片是否在代码块中
                if code_block_pattern.search(md_content_line):
                    in_code_block = not in_code_block

                #3.2.2 判断当前行是否是这张图片（利用正则匹配）
                #在代码块中，或者没匹配上图片直接跳过本次循环
                if in_code_block or not image_pattern.search(md_content_line):
                    continue

                # 3.3 到这已经匹配到当前的图片，开始查找该图片的上下文
                #3.3.1 查找图片的上文（需要标题，因为上文标题和当前的图片有关）
                (above_title,above_context) = self.find_above_context(index,title_pattern,code_block_pattern,md_content_lines)

                # print(type(above_context))
                # print(above_context)
                # print("************"*40)

                #3.3.2 查找图片的下文（不需要标题，因为下文标题和当前的图片无关）
                following_context = self.find_following_context(index,title_pattern,code_block_pattern,md_content_lines)

                # 3.4 根据配置中设置的最大长度对上下文进行截取
                # 3.4.1 获取配置中的最大上下文长度
                img_content_length = self.config.img_content_length
                # 3.4.2 截取上文内容
                above_context_content = self.extract_contextual_content(above_context,img_content_length,True)
                # 3.4.3 截取下文内容
                following_context_content = self.extract_contextual_content(following_context, img_content_length, False)

                #将图片的上下文封装到图片的上下文对象中
                image_context = ImageContext(above_title = above_title,above_context_content=above_context_content,following_context_content = following_context_content)

                #4 组装Image_Info对象
                image_info = ImageInfo(image_name=image_file.name,image_path = str(image_file),image_context = image_context )
                image_info_list.append(image_info)

        return image_info_list

    #查找图片上文的方法
    def find_above_context(self,index:int,title_pattern:re.Pattern,code_block_pattern:re.Pattern,md_content_lines:List[str] ) -> Tuple[str,List[str]]:
        #上文的起始位置
        above_context_start_index = -1
        #上文的标题
        above_title = ""
        #排除代码块的干扰
        in_code_block = False

        # 从图片所在的上一行开始往上找最近标题，能找到标题，则拿标题作为上文的起始位置
        for index0 in range(index - 1, -1, -1):
            #判断是否在代码块中
            if code_block_pattern.search(md_content_lines[index0]):
                in_code_block = not in_code_block

            match = title_pattern.search(md_content_lines[index0])
            # 匹配上了
            if not in_code_block and match:
                above_title = match.group(2).strip()
                above_context_start_index = index0
                break

        above_context = md_content_lines[above_context_start_index:index]

        #返回上文的标题和上文内容
        return above_title,above_context

    def find_following_context(self,index:int,title_pattern:re.Pattern,code_block_pattern:re.Pattern,md_content_lines:List[str] ) -> List[str]:
        # 默认下文一直取到文档结尾
        following_context_end_index = len(md_content_lines)

        # 排除代码块里的标题干扰
        in_code_block = False

        # 从当前图片的下一行开始往下找
        for index1 in range(index + 1, len(md_content_lines)):
            # 判断是否进入/退出代码块
            if code_block_pattern.search(md_content_lines[index1]):
                in_code_block = not in_code_block
                continue

            # 遇到代码块外的标题，就把这个标题所在行作为下文终止位置
            if not in_code_block and title_pattern.search(md_content_lines[index1]):
                following_context_end_index = index1
                break

        # 从图片下一行，取到标题上一行
        following_context = md_content_lines[index + 1:following_context_end_index]

        return following_context

    #按照要求截取上下文内容
    def extract_contextual_content(self,context:List[str],img_content_length:int,is_above_context:bool) -> str:
        #段落列表
        paragraph_list = []

        #当前正在收集的段落列表
        current_paragraph = []

        #匹配图片的正则表达式
        md_image_pattern = re.compile(r"!\[.*?\]\(.*?\)")

        #遍历上下文列表，按空行或者其它图片进行切分段落
        for context_line in context:

            context_line_strip = context_line.strip()

            #如果当前行是空行或者是有其它图片，则说明到头了
            if not context_line_strip or md_image_pattern.search(context_line):
                #将current_paragraph中的内容移动到paragraph_list
                if current_paragraph:
                    paragraph_list.append("\n".join(current_paragraph))

                #清空current_paragraph
                current_paragraph = []
                continue

            #如果不是上述情况，则继续往当前段落列表追加
            current_paragraph.append(context_line)

        #遍历结束后加入最后一段
        if current_paragraph:
            paragraph_list.append("\n".join(current_paragraph))

        #如果是上文，则需要倒着找，因为离图片越近关系越紧密
        if is_above_context:
            paragraph_list.reverse()

        #按照config文件中的最大上下文进行截取
        current_length = 0
        limited_paragraph_context = []
        for paragraph in paragraph_list:
            limited_paragraph_context.append(paragraph)
            current_length += len(paragraph)
            if current_length >= img_content_length:
                break

        # 如果是上文，还需要反转回去
        if is_above_context:
            limited_paragraph_context.reverse()

        return "\n\n".join(limited_paragraph_context)

#利用VLM模型去生成图片的描述信息
class _VlmSummarizer:
    def __init__(self, logger: logging.Logger, node_name: str, config: Optional[ImportConfig] = None):
        self.logger = logger
        self.node_name = node_name
        self.config = config or get_config()

    def summarize_all(self,image_info_list:List[ImageInfo],file_title)->dict[str, str]:
        image_summaries = {}
        #1 创建vlm对象
        try:
            vlm_client = AIClients.get_vlm_client()
        except ConnectionError as e:
            self.logger.warning(f"Failed to connect to the VLM model. Reason: {e}")
            #需要给每张图片设置信息为"暂无摘要信息"
            for image_info in image_info_list:
                image_summaries[image_info.image_name] = "No summary available"
            return image_summaries

        #2 遍历每一张图片，给每一张图片生成摘要信息，并放在字典中
        for image_info in image_info_list:
            image_summaries[image_info.image_name] = self._summarize_single(file_title,image_info,vlm_client)

        return image_summaries


    # 给单个图片生成描述信息
    def _summarize_single(self,file_title:str,image_info:ImageInfo,vlm_client:OpenAI) -> str:
        #1 组装最后给大模型的关于图片的上下文文本信息
        context_content = '\n'.join([image_info.image_context.above_title,image_info.image_context.above_context_content,image_info.image_context.following_context_content])

        #2 读取图片，先将本地图片读取成二进制，利用base64将其编码成字符串
        try:
            with open(image_info.image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except IOError as e:
            self.logger.warning(f"Failed to read image file, unable to generate image summary: {e}")
            return "No summary available"

        #3 组装要发送给大模型的信息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                            f"背景信息：\n"
                            f"  1. 所属文档标题：\"{file_title}\"\n"
                            f"  2. 图片上下文：{context_content}\n"
                            f"请结合图片内容和上述上下文信息，"
                            f"用中文简要总结这张图片的内容，"
                            f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        },
                    },
                ]
            }
        ]

        #4 调用大模型，获取大模型的输出
        try:
            vlm_result = vlm_client.chat.completions.create(
                model=self.config.vl_model,
                messages=messages
            )
            summary = vlm_result.choices[0].message.content
            if not summary:
                return "No summary available"
            return summary.strip()
        except Exception as e:
            self.logger.warning(
                f"Failed to generate image summary for {image_info.image_name}: {e}"
            )
            return "No summary available"

#上传图片到MinIO并替换md文件中的内容
class _ImageUploader:
    def __init__(self, logger: logging.Logger, node_name: str, config: Optional[ImportConfig] = None):
        self.logger = logger
        self.node_name = node_name
        self.config = config or get_config()

    #上传图片到minio，并且替换掉md文件中的图片链接
    def upload_and_replace_content(self,image_info_list:List[ImageInfo],md_path_obj:Path,image_summaries:Dict[str,str])->str:
        #1 将图片上传到minio  并返回图片存储的url
        image_urls = self.upload_image_to_minio(image_info_list,md_path_obj)

        #2 替换md文件中图片的链接以及摘要
        new_md_content = self.replace_content_in_md(md_path_obj,image_urls,image_summaries)

        return new_md_content
        

    # 上传图片到minio
    def upload_image_to_minio(self, image_info_list:List[ImageInfo],md_path_obj:Path):
        #字典，key是图片的名称，value是图片的url
        image_urls = {}
        #1 创建minio客户端
        try:
            minio_client = StorageClients.get_minio()
        except ConnectionError as e:
            logging.warning(f"Failed to connect to the minio model. Reason: {e}")
            #连接minio失败，将本地路径作为图片的url
            for image_info in image_info_list:
                image_urls[image_info.image_name] = image_info.image_path
            return image_urls

        #2 遍历图片列表，上传图片到minio
        #从配置文件中拿到backet_name
        bucket_name = self.config.minio_bucket
        minio_base_url = self.config.get_minio_base_url()
        for image_info in image_info_list:
            destination_file = f"{md_path_obj.stem}/{image_info.image_name}"
            minio_client.fput_object(
                bucket_name, destination_file, image_info.image_path,
            )
            image_urls[image_info.image_name] = f"{minio_base_url}/{bucket_name}/{destination_file}"

        return image_urls

    # 替换md文件中图片的链接以及摘要
    def replace_content_in_md(
            self,
            md_path_obj: Path,
            image_urls: Dict[str, str],
            image_summaries: Dict[str, str]
    )->str:
        # 匹配 Markdown 图片语法
        image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        # 匹配代码块开始/结束
        code_block_pattern = re.compile(r"^\s*```")

        md_content = md_path_obj.read_text(encoding="UTF-8")

        in_code_block = False
        new_lines = []

        def replacer(match: re.Match):
            old_alt = match.group(1)
            old_url = match.group(2)

            image_name = Path(old_url).name

            image_url = image_urls.get(image_name)
            if not image_url:
                return match.group(0)

            summary = image_summaries.get(image_name, old_alt)

            return f"![{summary}]({image_url})"

        for line in md_content.splitlines(keepends=True):
            # 遇到 ```，切换代码块状态
            if code_block_pattern.search(line):
                in_code_block = not in_code_block
                new_lines.append(line)
                continue

            # 代码块内部不替换
            if in_code_block:
                new_lines.append(line)
                continue

            # 代码块外部才替换图片链接
            new_line = image_pattern.sub(replacer, line)
            new_lines.append(new_line)

        new_md_content = "".join(new_lines)
        return new_md_content


class MdImageNode(BaseNode):
    name = "md_image_node"

    def __init__(self):
        super().__init__()
        self.md_file_handler = _MdFileHandler(logger=self.logger,node_name=self.name)
        self.image_scanner = _ImageScanner(logger=self.logger, node_name=self.name,config=self.config)
        self.vlm_summarize = _VlmSummarizer(logger=self.logger, node_name=self.name,config=self.config)
        self.image_uploader = _ImageUploader(logger=self.logger, node_name=self.name,config=self.config)

    def process(self, state:ImportGraphState ) -> ImportGraphState:
        #1 读取md文件的内容到内存
        (md_content,md_path_obj,image_dir_obj)=self.md_file_handler.read_md(state["md_path"])

        #2 处理图片信息，包含图片的名称、在md文件中的上下文
        image_info_list = self.image_scanner.scan_image_dir(md_content,image_dir_obj)

        #3 将上述图片信息列表传给vlm模型做处理，得到图片的描述信息 得到一个字典 key是文件名，value是文件的描述信息
        image_summaries = self.vlm_summarize.summarize_all(image_info_list, state["file_title"])

        #4 将图片上传给Minio，并且替换掉原来md文件中的部分内容
        new_md_content = self.image_uploader.upload_and_replace_content(image_info_list,md_path_obj,image_summaries)

        # 仅在显式调试模式下备份处理后的 Markdown。
        if get_settings().import_debug_artifacts:
            self.md_file_handler._backup(md_path_obj, new_md_content)

        #6 更新状态中的md_content
        state["md_content"] = new_md_content

        return state
