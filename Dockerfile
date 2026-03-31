FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY answer_cropper2.py /app/answer_cropper2.py
COPY handler.py /app/handler.py
COPY r2_client.py /app/r2_client.py

CMD ["python", "-u", "/app/handler.py"]
