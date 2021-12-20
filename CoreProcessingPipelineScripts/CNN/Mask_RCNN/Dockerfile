FROM mambaorg/micromamba:0.19.1

COPY . /home/micromamba/

RUN micromamba install -y -n base -f /home/micromamba/environment.yml && micromamba clean -a

ENV PATH "/home/micromamba/postprocessing:$MAMBA_ROOT_PREFIX/bin:$PATH"
ENV PYTHONPATH "/home/micromamba:$PYTHONPATH"

ENTRYPOINT ["postprocessingCracksRings.py"]
