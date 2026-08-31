import logging
import shutil
import zipfile

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import ValidationError
from knowledge.processor.import_processor.state import ImportGraphState
from pathlib import Path
import requests
import time


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1 上传并轮询MinerU的解析结果
        zip_url = self._upload_pdf_and_query_result(Path(state.get("pdf_path")))

        # 2 下载ZIP并提取MD文件
        md_path = self._download_extract_md(zip_url,Path(state.get("file_dir")),Path(state.get("pdf_path")))

        # 3 将md_path存入state
        state["md_path"] = md_path

        return state

    def _upload_pdf_and_query_result(self, pdf_path_obj: Path) -> str:
        # 1 检查MinerU的配置
        mineru_api_token = self.config.mineru_api_token
        mineru_base_url = self.config.mineru_base_url
        if not mineru_api_token or not mineru_base_url:
            self.logger.error("mineru_api_token or mineru_base_url not set")
            raise ValidationError(message="mineru_api_token or mineru_base_url not set", node_name="pdf_to_md_node")

        # 2 获取文件上传链接
        # 2.1 构建url
        mineru_upload_url = f"{mineru_base_url}/file-urls/batch"
        # 2.2 构建请求头
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {mineru_api_token}"
        }
        # 2.3 构建请求体
        data = {
            "files": [
                {"name": pdf_path_obj.name, "data_id": "abcd"}
            ],
            "model_version": "vlm"
        }

        # 2.4 发送请求获取上传连接
        try:
            response = requests.post(mineru_upload_url, headers=header, json=data, timeout=40)
        except requests.RequestException as e:
            self.logger.error(f"API call failed: {e}")
            raise RuntimeError(f"API call failed: {e}") from e

        # 2.5 判断请求是否成功
        if response.status_code != 200:
            self.logger.error(f"API call failed,status_code:{response.status_code}")
            raise RuntimeError(f"API call failed,status_code:{response.status_code}")

        result = response.json()
        self.logger.info("MinerU upload URL request succeeded")

        # 2.6 判断业务状态码是否成功
        if result["code"] != 0:
            self.logger.error(f"failed to get upload URL,business status_code:{result['code']}")
            raise RuntimeError(f"failed to get upload URL,business status_code:{result['code']}")

        # 2.7 获取上传链接成功
        batch_id = result["data"]["batch_id"]
        urls = result["data"]["file_urls"]
        upload_url = urls[0]

        self.logger.info("get upload url success,batch_id:%s", batch_id)

        # 3 上传PDF文件到MinerU
        try:
            with open(str(pdf_path_obj), 'rb') as f:
                file_content = f.read()
                res_upload = requests.put(upload_url, data=file_content, timeout=120)
        except OSError as e:
            self.logger.error(f"read PDF file failed: {e}")
            raise RuntimeError(f"read PDF file failed: {e}") from e


        # 3.1 判断状态码是否表示上传成功
        if res_upload.status_code != 200:
            self.logger.error(f"upload failed,status_code:{res_upload.status_code}")
            raise RuntimeError(f"upload failed,status_code:{res_upload.status_code}")

        self.logger.info(f"upload success")

        # 4 轮询查询解析结果，查到成功、失败或超时
        #定义最大超时时间
        max_time = 600

        #定义轮询的时间间隔
        interval_time = 5

        #定义开始时间
        start_time = time.monotonic()
        while True:
            end_time = time.monotonic()
            if end_time - start_time > max_time:
                self.logger.error(f"wait for result timeout")
                raise RuntimeError(f"wait for result timeout")
            pull_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
            pull_res = requests.get(pull_url, headers=header, timeout=20)
            #判断状态码是否是200
            if pull_res.status_code!=200:
                self.logger.warning(f"pull result failed,status_code:{pull_res.status_code}")
                time.sleep(interval_time)
                continue
            #判断业务状态码
            if pull_res.json()["code"]!=0:
                self.logger.warning(f"pull result failed,business status_code:{pull_res.json()['code']}")
                time.sleep(interval_time)
                continue

            extract_result = pull_res.json()["data"]["extract_result"][0]
            if extract_result["state"] == "done":
                #表示Minus真正完成了pdf转换为md的任务
                self.logger.info(f"extract success")
                return extract_result["full_zip_url"]
            elif extract_result["state"] == "failed":
                self.logger.error(f"extract failed,business status_code:{extract_result['code']}")
                raise RuntimeError(f"extract failed,business status_code:{extract_result['code']}")
            else:
                time.sleep(interval_time)
                continue


    def _download_extract_md(self, zip_url: str,file_dir_obj:Path,pdf_path_obj:Path) -> str:
        #1 发送get请求下载zip包
        try:
            response = requests.get(zip_url, timeout=20)
        except requests.RequestException as e:
            self.logger.error(f"Failed to download the zip file: {e}")
            raise RuntimeError(f"Failed to download the zip file: {e}")

        if response.status_code != 200:
            self.logger.error(f"Failed to download the zip file,status_code:{response.status_code}")
            raise RuntimeError(f"Failed to download the zip file,status_code:{response.status_code}")

        #2 指定zip包的存储路径，将下载下来的zip包写入该路径
        #2.1 构建zip包的存储路径
        zip_save_path = file_dir_obj / f"{pdf_path_obj.stem}.zip"

        #2.2 将response中的内容写入zip包的存储路径下
        try:
            with open(zip_save_path, "wb") as f:
                f.write(response.content)
            self.logger.info(f"Successfully downloaded the zip file, saved to {zip_save_path}")
        except OSError as e:
            self.logger.error(f"Failed to save zip file: {e}")
            raise RuntimeError(f"Failed to save zip file: {e}") from e

        #3 解压zip包
        #3.1 构建zip包的解压路径：file_dir/文件名
        extract_dir_obj = file_dir_obj / pdf_path_obj.stem
        self.logger.info(f"Successfully extracted the zip file, saved to {str(extract_dir_obj)}")

        #判断是否有同名路径了，如果有先删除掉路径下的同名路径
        if extract_dir_obj.exists():
            shutil.rmtree(extract_dir_obj)
        #创建解压路径
        extract_dir_obj.mkdir(parents=True, exist_ok=True)

        #3.2 将zip的内容解压到解压路径下
        try:
            with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir_obj)
            self.logger.info(f"Successfully extracted the zip file, extracted to {extract_dir_obj}")
        except OSError as e:
            self.logger.error(f"Failed to extract the zip file: {e}")
            raise RuntimeError(f"Failed to extract the zip file: {e}")

        #删除zip文件
        zip_save_path.unlink()

        #3.3 将md文件重命名  pdf的文件名.md
        #找到md文件
        md_path = extract_dir_obj / "full.md"
        #构造新的名字
        new_md_path = extract_dir_obj / f"{pdf_path_obj.stem}.md"
        #重命名
        md_path.rename(new_md_path)

        #4 返回md文件的路径
        self.logger.info(f"Rename completed, new md file path：{new_md_path}")
        return str(new_md_path)
