FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU

RUN apt update \
    && apt install --no-install-recommends -y gcc libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 libsm6

WORKDIR /home

COPY . /home

RUN pip install -r requirements.txt

ENV PATH="/home/processing:$PATH"
ENV PYTHONPATH="/home:$PYTHONPATH"

ENTRYPOINT ["processing.py", "--weightRing", "/home/weights/best10px1000eAugEnlargedDataset.pt"]