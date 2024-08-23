FROM mambaorg/micromamba:1.5.8

COPY . /home/mambauser/

WORKDIR /home/mambauser

RUN micromamba install -y -n base wget -f /home/mambauser/environment.yml && micromamba clean -y -a

ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENV PATH="/home/mambauser/processing:$MAMBA_ROOT_PREFIX/bin:$PATH"
ENV PYTHONPATH="/home/mambauser:$PYTHONPATH"

ENTRYPOINT ["processing.py", "--weightRing", "/home/mambauser/weights/best10px1000eAugEnlargedDataset.pt"]
