FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/local/bin/pip3.11 /usr/bin/pip3 \
    && ln -sf /usr/local/bin/pip3.11 /usr/bin/pip

WORKDIR /app

# pip 업그레이드
RUN python3.11 -m pip install --upgrade pip

# 의존성 설치 (단계별 분리)
COPY requirements_base.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements_base.txt

COPY requirements_engines.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements_engines.txt

COPY requirements_app.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements_app.txt

# 코드 복사
COPY *.py /app/
COPY config.yaml /app/

# 모델 미리 다운로드 (빌드 시점에 캐싱)
ARG HF_TOKEN
RUN HF_TOKEN=${HF_TOKEN} python3.11 /app/download_models.py

# 실행
CMD ["python3.11", "main.py"]

