FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1


# 시스템 패키지 (Ubuntu 22.04 기본 python3.10 사용)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# python / pip 기본값
RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# pip 업그레이드
RUN pip install --upgrade pip

# 의존성 설치 (requirements.txt에 torch 2.4.0+cu121 포함됨)
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# NumPy 2.0 충돌 방지를 위한 재고정 (중요)
RUN python -m pip install --no-cache-dir "numpy<2.0"

# 코드
COPY *.py /app/
COPY config.yaml /app/

CMD ["python", "main.py"]
