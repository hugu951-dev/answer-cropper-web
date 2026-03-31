from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.client import BaseClient


@dataclass(frozen=True)
class R2Config:
    account_id: str
    bucket: str
    access_key_id: str
    secret_access_key: str

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


def load_r2_config() -> R2Config:
    account_id = os.environ["R2_ACCOUNT_ID"]
    bucket = os.environ["R2_BUCKET"]
    access_key_id = os.environ["R2_ACCESS_KEY_ID"]
    secret_access_key = os.environ["R2_SECRET_ACCESS_KEY"]
    return R2Config(
        account_id=account_id,
        bucket=bucket,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
    )


def build_r2_client(config: R2Config | None = None) -> BaseClient:
    config = config or load_r2_config()
    return boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        region_name="auto",
    )


def upload_bytes(
    object_key: str,
    data: bytes,
    content_type: str = "application/octet-stream",
    client: BaseClient | None = None,
    config: R2Config | None = None,
) -> None:
    config = config or load_r2_config()
    client = client or build_r2_client(config)
    client.put_object(
        Bucket=config.bucket,
        Key=object_key,
        Body=data,
        ContentType=content_type,
    )


def upload_file(
    local_path: Path,
    object_key: str,
    content_type: str = "application/octet-stream",
    client: BaseClient | None = None,
    config: R2Config | None = None,
) -> None:
    config = config or load_r2_config()
    client = client or build_r2_client(config)
    with local_path.open("rb") as handle:
        client.upload_fileobj(
            handle,
            config.bucket,
            object_key,
            ExtraArgs={"ContentType": content_type},
        )


def download_bytes(
    object_key: str,
    client: BaseClient | None = None,
    config: R2Config | None = None,
) -> bytes:
    config = config or load_r2_config()
    client = client or build_r2_client(config)
    response = client.get_object(Bucket=config.bucket, Key=object_key)
    return response["Body"].read()


def download_file(
    object_key: str,
    local_path: Path,
    client: BaseClient | None = None,
    config: R2Config | None = None,
) -> None:
    config = config or load_r2_config()
    client = client or build_r2_client(config)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as handle:
        client.download_fileobj(config.bucket, object_key, handle)


def generate_download_url(
    object_key: str,
    expires_in_seconds: int = 900,
    client: BaseClient | None = None,
    config: R2Config | None = None,
) -> str:
    config = config or load_r2_config()
    client = client or build_r2_client(config)
    return client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": config.bucket, "Key": object_key},
        ExpiresIn=expires_in_seconds,
    )
