FROM python:3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    PATH="/home/processing:$PATH"

RUN apt update \
    && apt install --no-install-recommends -y gcc git zip unzip wget curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 libsm6

COPY . /home/

WORKDIR /home

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["processing.py", "--weightRing", "/home/weights/best10px1000eAugEnlargedDataset.pt"]
