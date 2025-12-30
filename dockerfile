FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1


# 시스템 패키지
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# python / pip 기본값
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# pip 업그레이드
RUN pip install --upgrade pip

# CUDA 대응 torch 설치 (중요)
RUN pip install torch==2.2.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 나머지 의존성
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# NumPy 2.0 충돌 방지를 위한 재고정 (중요)
RUN python -m pip install --no-cache-dir "numpy<2.0"

# 코드
COPY *.py /app/
COPY config.yaml /app/

CMD ["python", "main.py"]
