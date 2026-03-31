from __future__ import annotations

import logging
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from r2_client import generate_download_url, upload_bytes
from runpod_client import get_job_status, submit_job


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Answer Cropper", version="2.0.0")
logger = logging.getLogger(__name__)
CROP_SCALE = 4.0


def sanitize_download_name(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem or "answer-crops"
    ascii_stem = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "-_") else "_" for ch in stem)
    ascii_stem = ascii_stem.strip("._") or "answer-crops"
    download_name = f"{ascii_stem}_cropped.zip"
    utf8_name = quote(f"{stem}_cropped.zip")
    return download_name, utf8_name


def map_runpod_status(raw_status: str | None) -> str:
    raw = (raw_status or "").upper()
    if raw in {"IN_QUEUE", "QUEUED"}:
        return "queued"
    if raw in {"IN_PROGRESS", "RUNNING"}:
        return "running"
    if raw == "COMPLETED":
        return "completed"
    if raw in {"FAILED", "CANCELLED", "TIMED_OUT"}:
        return "failed"
    return "unknown"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)) -> JSONResponse:
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")

    request_id = uuid.uuid4().hex
    download_name, utf8_name = sanitize_download_name(filename)
    input_bucket_key = f"inputs/{request_id}/input.pdf"
    output_bucket_key = f"outputs/{request_id}/{download_name}"

    try:
        upload_bytes(
            object_key=input_bucket_key,
            data=pdf_bytes,
            content_type="application/pdf",
        )
        job_response = submit_job(
            {
                "request_id": request_id,
                "input_bucket_key": input_bucket_key,
                "output_bucket_key": output_bucket_key,
                "download_name": download_name,
                "utf8_download_name": utf8_name,
                "scale": CROP_SCALE,
            }
        )
    except Exception:
        logger.exception("Failed to create Runpod job for %s", filename)
        raise HTTPException(status_code=500, detail="작업 생성 중 오류가 발생했습니다.")

    job_id = job_response.get("id")
    if not job_id:
        logger.error("Runpod response missing job id: %s", job_response)
        raise HTTPException(status_code=502, detail="작업 ID를 받지 못했습니다.")

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "request_id": request_id,
        },
    )


@app.get("/jobs/{job_id}")
async def read_job(job_id: str) -> JSONResponse:
    try:
        status_response = get_job_status(job_id)
    except Exception:
        logger.exception("Failed to fetch Runpod job status for %s", job_id)
        raise HTTPException(status_code=502, detail="작업 상태를 확인하지 못했습니다.")

    raw_status = status_response.get("status")
    status = map_runpod_status(raw_status)
    output = status_response.get("output") or {}

    response_payload: dict[str, object] = {
        "job_id": job_id,
        "status": status,
        "raw_status": raw_status,
    }

    if status == "completed":
        output_bucket_key = output.get("output_bucket_key")
        if output_bucket_key:
            response_payload["download_url"] = generate_download_url(output_bucket_key)
        response_payload["saved_count"] = output.get("saved_count")

    if status == "failed":
        response_payload["error"] = status_response.get("error") or output.get("error") or "작업이 실패했습니다."

    return JSONResponse(content=response_payload)
