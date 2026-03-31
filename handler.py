from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import runpod

from answer_cropper2 import collect_entries
from r2_client import download_file, upload_bytes


def build_zip_bytes(output_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(output_dir.rglob("*")):
            if not file_path.is_file():
                continue
            archive.write(file_path, arcname=file_path.relative_to(output_dir))
    buffer.seek(0)
    return buffer.getvalue()


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input") or {}
    request_id = str(job_input["request_id"])
    input_bucket_key = str(job_input["input_bucket_key"])
    output_bucket_key = str(job_input["output_bucket_key"])
    scale = float(job_input.get("scale", 4.0))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_pdf = temp_path / "input.pdf"
        output_dir = temp_path / "output"

        download_file(input_bucket_key, input_pdf)
        saved_count = collect_entries(
            pdf_path=input_pdf,
            output_dir=output_dir,
            scale=scale,
            selected_pages=None,
        )

        if saved_count == 0:
            return {
                "request_id": request_id,
                "saved_count": 0,
                "output_bucket_key": output_bucket_key,
                "error": "추출된 결과가 없습니다.",
            }

        zip_bytes = build_zip_bytes(output_dir)
        upload_bytes(
            object_key=output_bucket_key,
            data=zip_bytes,
            content_type="application/zip",
        )

    return {
        "request_id": request_id,
        "saved_count": saved_count,
        "output_bucket_key": output_bucket_key,
    }


runpod.serverless.start({"handler": handler})
