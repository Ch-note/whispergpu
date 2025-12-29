INSTALL.md

This project officially supports Docker-based GPU execution only.
Non-Docker environments (local OS installation, conda, Colab, VM manual setup) are not supported and may fail due to CUDA / cuDNN ABI incompatibilities.

Supported Environment

Execution : Docker only
Base Image : nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04
Python : 3.11
CUDA : 12.1
cuDNN : 9.x (system)
GPU : NVIDIA GPU (Ampere or newer recommended)

Design Principles

This project uses a cuDNN ABI separation strategy to avoid runtime conflicts.

PyTorch:

Installed via pip wheel

Internally uses cuDNN 8.9 ABI

Does not depend on system cuDNN

ctranslate2 / faster-whisper:

Built for CUDA 12

Uses system cuDNN 9.x

This allows multiple CUDA-dependent libraries to coexist in the same process without ABI conflicts or segmentation faults.
This structure is reliably reproducible only in Docker environments.

Prerequisites

NVIDIA Driver

The host machine must have an NVIDIA driver installed.

Command:
nvidia-smi

CUDA Toolkit installation on the host is not required.

Docker & NVIDIA Container Toolkit

Install Docker:
https://docs.docker.com/engine/install/

Install NVIDIA Container Toolkit:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Verify GPU access inside Docker:

Command:
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04 nvidia-smi

Build Docker Image

From the project root:

Command:
docker build -t meeting-ai .

The provided Dockerfile uses the following base image:
FROM nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04

Run Container

Command:
docker run --gpus all -v $(pwd):/app meeting-ai

Run in detached mode if needed:
docker run -d --gpus all meeting-ai

Installation Verification

Inside the running container, execute:

Python snippet:
import torch
print("torch version :", torch.version)
print("CUDA version :", torch.version.cuda)
print("cuDNN used :", torch.backends.cudnn.version())
print("CUDA enabled :", torch.cuda.is_available())

Expected output:
torch version : 2.2.2
CUDA version : 12.1
cuDNN used : 8900
CUDA enabled : True

This confirms that PyTorch is using its internal cuDNN 8.9 ABI while the system provides cuDNN 9.x for other CUDA libraries without conflict.

Unsupported Configurations

The following setups are explicitly unsupported and may cause runtime failures:

Local OS installation without Docker

Conda environments

PyTorch source builds

Mixing system cuDNN 8.x and 9.x

Forcing cuDNN via LD_LIBRARY_PATH

Scope Exclusions

The following environments are intentionally excluded from this document:

Google Colab

Manual VM setup

Local virtual environments

They may be used for experimentation only and are not guaranteed to work.

Summary

This project defines Docker as the single source of truth for execution.
By fixing CUDA 12.1, system cuDNN 9, and PyTorch pip wheels, the runtime environment avoids cuDNN ABI conflicts and ensures stable execution of whisper, diarization, and STT pipelines.

Docker-based execution described in this document is the only supported and guaranteed configuration.




이 프로젝트는 Docker 기반 GPU 실행만을 공식적으로 지원합니다.
Docker가 아닌 환경(로컬 OS 직접 설치, conda, Colab, VM 수동 세팅 등)은 공식적으로 지원하지 않으며 CUDA / cuDNN ABI 충돌로 인해 정상 동작을 보장하지 않습니다.

지원 환경

실행 방식 : Docker 전용
베이스 이미지 : nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04
파이썬 버전 : 3.11
CUDA 버전 : 12.1
cuDNN 버전 : 9.x (시스템 라이브러리)
GPU : NVIDIA GPU (Ampere 이상 권장)

설계 원칙

본 프로젝트는 런타임 충돌을 방지하기 위해 cuDNN ABI 분리 구조를 사용합니다.

PyTorch는 pip wheel 방식으로 설치되며, 내부적으로 cuDNN 8.9 ABI를 사용합니다.
이로 인해 PyTorch는 시스템에 설치된 cuDNN에 의존하지 않습니다.

ctranslate2 및 faster-whisper는 CUDA 12 기반으로 빌드되었으며,
시스템 cuDNN 9.x를 사용합니다.

이 구조를 통해 동일 프로세스 내에서 여러 CUDA 의존 라이브러리가
ABI 충돌이나 segmentation fault 없이 공존할 수 있습니다.
이 구조는 Docker 환경에서만 안정적으로 재현 가능합니다.

사전 요구 사항

NVIDIA 드라이버가 호스트 머신에 설치되어 있어야 합니다.

다음 명령어가 정상 출력되어야 합니다.

nvidia-smi

호스트 머신에 CUDA Toolkit을 별도로 설치할 필요는 없습니다.

Docker 및 NVIDIA Container Toolkit

Docker 설치 방법:
https://docs.docker.com/engine/install/

NVIDIA Container Toolkit 설치 방법:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Docker에서 GPU 접근 확인:

docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04 nvidia-smi

Docker 이미지 빌드

프로젝트 루트 디렉토리에서 다음 명령어를 실행합니다.

docker build -t meeting-ai .

본 프로젝트의 Dockerfile은 다음 베이스 이미지를 사용합니다.

FROM nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04

컨테이너 실행

docker run --gpus all -v $(pwd):/app meeting-ai

백그라운드 실행이 필요한 경우:

docker run -d --gpus all meeting-ai

설치 검증

실행 중인 컨테이너 내부에서 다음 코드를 실행하여 설치 상태를 확인할 수 있습니다.

import torch
print("torch 버전 :", torch.version)
print("CUDA 버전 :", torch.version.cuda)
print("cuDNN 사용 버전 :", torch.backends.cudnn.version())
print("CUDA 사용 가능 여부 :", torch.cuda.is_available())

정상 출력 예시는 다음과 같습니다.

torch 버전 : 2.2.2
CUDA 버전 : 12.1
cuDNN 사용 버전 : 8900
CUDA 사용 가능 여부 : True

이는 PyTorch가 내부 cuDNN 8.9를 사용하고 있으며,
시스템 cuDNN 9.x와 충돌 없이 정상 동작하고 있음을 의미합니다.

지원하지 않는 구성

다음 구성은 공식적으로 지원하지 않으며, 런타임 오류가 발생할 수 있습니다.

Docker 없이 로컬 OS에 직접 설치하는 경우

conda 환경 사용

PyTorch를 source build한 경우

시스템에 cuDNN 8.x와 9.x가 혼재된 경우

LD_LIBRARY_PATH로 cuDNN 로딩 순서를 강제한 경우

범위 외 환경

다음 환경은 실험 목적에 한해서만 사용할 수 있으며,
본 문서에서는 다루지 않습니다.

Google Colab

VM 수동 세팅 환경

로컬 Python 가상환경

요약

본 프로젝트는 Docker를 유일한 실행 기준 환경으로 정의합니다.
CUDA 12.1, 시스템 cuDNN 9, PyTorch pip wheel 구조를 고정함으로써
cuDNN ABI 충돌을 구조적으로 회피하고
Whisper, diarization, STT 파이프라인을 안정적으로 실행합니다.

본 문서에 기술된 Docker 실행 방식이
이 프로젝트에서 유일하게 보장되는 실행 구성입니다.