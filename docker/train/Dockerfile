FROM cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py38:0.0.36

USER root
RUN apt-get update --fix-missing && apt-get upgrade -y &&\
 apt-get install -qy\
 python3-dev\
 build-essential\
 libblas-dev\
 libatlas-base-dev\
 libosmesa6-dev\
 libgl1-mesa-glx\
 libglfw3\
 patchelf\
 libgmp-dev\
 libgmpxx4ldbl\
 libglew-dev

ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
WORKDIR /home/user

RUN pip3 install "Cython<3"
RUN pip3 uninstall --yes torch
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir --no-build-isolation \
    ninja \
    setuptools \
    packaging \
    pydantic

RUN pip3 install --no-cache-dir --no-build-isolation \
    flash-attn

RUN pip3 install --no-cache-dir \
    wandb \
    numpy \
    h5py \
    tqdm \
    pyrallis \
    omegaconf

USER user
