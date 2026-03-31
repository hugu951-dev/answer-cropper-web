from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


RUNPOD_API_BASE = "https://api.runpod.ai/v2"


@dataclass(frozen=True)
class RunpodConfig:
    api_key: str
    endpoint_id: str

    @property
    def endpoint_base_url(self) -> str:
        return f"{RUNPOD_API_BASE}/{self.endpoint_id}"


def load_runpod_config() -> RunpodConfig:
    return RunpodConfig(
        api_key=os.environ["RUNPOD_API_KEY"],
        endpoint_id=os.environ["RUNPOD_ENDPOINT_ID"],
    )


def build_headers(config: RunpodConfig | None = None) -> dict[str, str]:
    config = config or load_runpod_config()
    return {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }


def submit_job(job_input: dict[str, Any], config: RunpodConfig | None = None) -> dict[str, Any]:
    config = config or load_runpod_config()
    response = requests.post(
        f"{config.endpoint_base_url}/run",
        headers=build_headers(config),
        json={"input": job_input},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_job_status(job_id: str, config: RunpodConfig | None = None) -> dict[str, Any]:
    config = config or load_runpod_config()
    response = requests.get(
        f"{config.endpoint_base_url}/status/{job_id}",
        headers=build_headers(config),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()
