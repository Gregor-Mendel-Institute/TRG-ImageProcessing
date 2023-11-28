FROM mambaorg/micromamba:0.20.0

COPY . /home/mambauser/

WORKDIR /home/mambauser

RUN micromamba install -y -n base wget -f /home/mambauser/environment.yml && micromamba clean -y -a

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN wget https://data.swarts.gmi.oeaw.ac.at/treeringcrackscomb2_onlycracks20210121T2224/mask_rcnn_treeringcrackscomb2_onlycracks_0522.h5 && wget https://data.swarts.gmi.oeaw.ac.at/treeringcrackscomb2_onlyring20210121T1457/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5

ENV PATH "/home/mambauser/postprocessing:$MAMBA_ROOT_PREFIX/bin:$PATH"
ENV PYTHONPATH "/home/mambauser:$PYTHONPATH"

ENTRYPOINT ["postprocessingCracksRings.py", "--weightRing", "/home/mambauser/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5", "--weightCrack", "/home/mambauser/mask_rcnn_treeringcrackscomb2_onlycracks_0522.h5"]
