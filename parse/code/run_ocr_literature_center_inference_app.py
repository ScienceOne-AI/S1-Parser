# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, Response
import json
import re, os
from datetime import datetime
import logging
import sys
from pydantic import BaseModel
import time
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import tempfile
from PyPDF2 import PdfReader, PdfWriter  # 导入PyPDF2库

# 调整导入路径
#sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append("/code/BedrockOCR-main/")
from magic_pdf.model.taichu_custom_model import TaichuOCR
from self_task import single_task_recognition, parse_pdf
import zipfile
from fastapi.responses import Response
import zipfile
import os
from starlette.concurrency import run_in_threadpool

"""
导入了PyPDF2库的PdfReader和PdfWriter用于 PDF 修复
2. 添加了repair_pdf函数实现 PDF 修复功能
3. 在parse_document接口中：
◦ 保存原始 PDF 后添加了修复步骤
◦ 使用修复后的 PDF 进行解析操作
◦ 增加了修复失败的错误处理
4. 完善了异常处理逻辑，对包含 "object out of range" 的错误给出更友好的提示
5. 增加了修复后临时文件的清理步骤，避免文件残留
"""

# 全局模型实例和线程池
model_parser = None
executor = ThreadPoolExecutor(max_workers=2)


def repair_pdf(input_path, output_path):
    """修复PDF文件函数"""
    try:
        reader = PdfReader(input_path)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        with open(output_path, "wb") as f:
            writer.write(f)
        return True
    except Exception as e:
        print(f"修复失败: {e}")
        return False


def initialize_model():
    """初始化TaichuOCR模型"""
    global model_parser
    if model_parser is None:
        model_config_path = "/code/BedrockOCR-main/configs.yaml"
        config_path = os.getenv("OCR_CONFIG", model_config_path)
        model_parser = TaichuOCR(config_path)
    return model_parser


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期事件处理"""
    # 启动时初始化模型
    try:
        initialize_model()
        print("TaichuOCR model initialized successfully")
    except Exception as e:
        print(f"Failed to initialize TaiChuOCR model: {e}")
        raise

    yield

    # 关闭时清理资源
    global executor
    executor.shutdown(wait=True)
    print("Application shutdown complete")


app = FastAPI(
    title="TaichuOCR API",
    description="OCR and Document Parsing API Using TaichuOCR",
    version="1.0.0",
    lifespan=lifespan
)


# 响应模型
class TaskResponse(BaseModel):
    success: bool
    task_type: str
    content: str
    message: Optional[str] = None


class ParseResponse(BaseModel):
    success: bool
    message: str
    output_dir: Optional[str] = None
    files: Optional[List[str]] = None
    download_url: Optional[str] = None


@app.post("/ocr/text", response_model=TaskResponse)
async def extract_text(file: UploadFile = File(...)):
    """从图像或PDF中提取文本"""
    return await perform_ocr_task(file, "text")


@app.post("/ocr/formula", response_model=TaskResponse)
async def extract_formula(file: UploadFile = File(...)):
    """从图像或PDF中提取公式"""
    return await perform_ocr_task(file, "formula")


@app.post("/ocr/table", response_model=TaskResponse)
async def extract_table(file: UploadFile = File(...)):
    """从图像或PDF中提取表格"""
    return await perform_ocr_task(file, "table")


@app.post('/')
async def index1():
    return {"code": 200, "message": "success"}


@app.post('/parsemessage', response_model=TaskResponse)
async def index2():
    start_time = datetime.now()

    if request.method == 'POST':
        print("########post##########")
        headers = request.headers
        print("#######headers########", headers)  # 打印请求头
        data = request.data
        data = json.loads(data)
        print("#######received_data########", data)  # 打印请求

    file_path = data["file_path"]

    # 提取 prompt 并提供默认值
    prompt: str = data.get("prompt", "")
    if not prompt:
        prompt = "你是一个LaText OCR助手,目标是读取用户输入的信息，转换成LaTex信息"

    # 提取 save_dir 并提供默认值
    save_dir: str = data.get("save_dir", "")
    if not save_dir:
        save_dir = "./result"

    # 模型分析
    results = model_parser.parse_images(
        prompt=prompt,
        base_file_path=file_path,
        save_dir=save_dir
    )

    resdict = {}
    resdict["code"] = 200
    resdict.update(results)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"任务执行时长: {duration.total_seconds()}")

    resdict = json.dumps(resdict, ensure_ascii=False)
    response = Response(resdict, content_type='application/json')
    logging.info("####resdict:{}#".format(resdict))

    return response


async def perform_ocr_task(file: UploadFile, task_type: str) -> TaskResponse:
    """对上传的文件执行OCR任务"""
    if not model_parser:
        raise HTTPException(status_code=500, detail="Model not initialized")

    # 验证文件类型
    allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    # 保存上传的文件到临时位置
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # 创建输出目录
        output_dir = tempfile.mkdtemp(prefix=f"s1_parse_{task_type}_")

        # 在线程池中运行OCR任务
        loop = asyncio.get_event_loop()
        result_dir = await loop.run_in_executor(
            executor,
            single_task_recognition,
            temp_file_path,
            output_dir,
            model_parser,
            task_type
        )

        # 读取结果文件
        result_files = [f for f in os.listdir(result_dir) if f.endswith(f'_{task_type}_result.md')]
        if not result_files:
            raise Exception("No result file generated")

        result_file_path = os.path.join(result_dir, result_files[0])
        with open(result_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return TaskResponse(
            success=True,
            task_type=task_type,
            content=content,
            message=f"{task_type.capitalize()} extraction completed successfully"
        )

    finally:
        # 清理临时文件
        os.unlink(temp_file_path)


@app.post("/parsepdf")
async def parse_document(file: UploadFile = File(...)):
    try:
        if not model_parser:
            raise HTTPException(status_code=500, detail="Model not initialized")

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files supported")

        original_name = Path(file.filename).stem

        # 1. 保存上传的PDF到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(await file.read())
            tmp_pdf_path = tmp_pdf.name

        # 2. 创建修复后的临时文件路径
        repaired_pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

        # 3. 尝试修复PDF
        repair_success = repair_pdf(tmp_pdf_path, repaired_pdf_path)
        if not repair_success:
            raise Exception("PDF文件可能已损坏，修复尝试失败")

        # 4. 创建解析输出目录
        parse_out_dir = tempfile.mkdtemp(prefix="taichuocr_parse_")
        zip_root_name = f"{original_name}_parsed_{int(asyncio.get_event_loop().time())}"
        zip_root_name = "".join(c if ord(c) < 128 else "_" for c in zip_root_name)

        tmp_zip_path = ""
        try:
            # 5. 使用修复后的PDF进行解析
            result_dir = await run_in_threadpool(
                parse_pdf, repaired_pdf_path, parse_out_dir, model_parser
            )

            # 6. 构造zip文件
            tmp_zip_fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip")
            os.close(tmp_zip_fd)

            with zipfile.ZipFile(tmp_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(result_dir):
                    rel_dir = Path(root).relative_to(result_dir)
                    is_images = rel_dir.parts and rel_dir.parts[0] == "images"

                    for fname in files:
                        src_path = Path(root) / fname

                        if fname.endswith(".md"):
                            arc_path = Path(zip_root_name) / f"{original_name}.md"
                            zf.write(src_path, arc_path)
                        elif fname.endswith("_content_list.json"):
                            arc_path = Path(zip_root_name) / "content_list_process.json"
                            zf.write(src_path, arc_path)
                        elif is_images:
                            arc_path = Path(zip_root_name) / rel_dir / fname
                            zf.write(src_path, arc_path)

            # 7. 读取zip内容并返回
            with open(tmp_zip_path, "rb") as f:
                zip_bytes = f.read()
            headers = {"Content-Disposition": f'attachment; filename="{zip_root_name}.zip"'}
            return Response(content=zip_bytes, media_type="application/zip", headers=headers)

        finally:
            # 8. 清理所有临时文件和目录
            try:
                os.unlink(tmp_pdf_path)
                os.unlink(repaired_pdf_path)  # 清理修复后的PDF
                os.unlink(tmp_zip_path)
            except OSError:
                pass
            try:
                import shutil
                shutil.rmtree(parse_out_dir, ignore_errors=True)
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        # 完善异常处理逻辑
        if "object out of range" in str(e):
            detail = "PDF文件可能已损坏，请尝试上传完好的PDF或修复后重试"
        else:
            detail = f"Parsing failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO,
                        datefmt='%I:%M:%S')
    uvicorn.run(app, host="0.0.0.0", port=22)
