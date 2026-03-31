from __future__ import annotations

import io
import logging
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from answer_cropper2 import collect_entries


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Answer Cropper", version="1.0.0")
logger = logging.getLogger(__name__)
CROP_SCALE = 4.0


def build_zip_bytes(output_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(output_dir.rglob("*")):
            if not file_path.is_file():
                continue
            archive.write(file_path, arcname=file_path.relative_to(output_dir))
    buffer.seek(0)
    return buffer.getvalue()


def sanitize_download_name(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem or "answer-crops"
    ascii_stem = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "-_") else "_" for ch in stem)
    ascii_stem = ascii_stem.strip("._") or "answer-crops"
    download_name = f"{ascii_stem}_cropped.zip"
    utf8_name = quote(f"{stem}_cropped.zip")
    return download_name, utf8_name


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


@app.post("/crop")
async def crop_pdf(file: UploadFile = File(...)) -> StreamingResponse:
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_pdf = temp_path / "input.pdf"
            output_dir = temp_path / "output"
            input_pdf.write_bytes(pdf_bytes)

            saved_count = collect_entries(
                pdf_path=input_pdf,
                output_dir=output_dir,
                scale=CROP_SCALE,
                selected_pages=None,
            )

            if saved_count == 0:
                raise HTTPException(status_code=422, detail="추출된 결과가 없습니다.")

            zip_bytes = build_zip_bytes(output_dir)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Crop request failed for %s", filename)
        raise HTTPException(status_code=500, detail="서버 처리 중 오류가 발생했습니다.")

    download_name, utf8_name = sanitize_download_name(filename)
    headers = {
        "Content-Disposition": (
            f'attachment; filename="{download_name}"; '
            f"filename*=UTF-8''{utf8_name}"
        )
    }
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers=headers,
    )
