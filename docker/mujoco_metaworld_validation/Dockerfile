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

ENV LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH"
RUN mkdir /files
RUN wget -O mujoco210.tar https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && tar -xzf mujoco210.tar -C /files
ENV  MUJOCO_PY_MUJOCO_PATH="/files/mujoco210"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/files/mujoco210/bin"
WORKDIR /home/user

# install torch
RUN pip3 install "Cython<3"
RUN pip3 uninstall --yes torch
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# packages for flash-attn installation
RUN pip3 install --no-cache-dir --no-build-isolation \
    ninja \
    setuptools \
    packaging \
    pydantic

# flash-attn installation
RUN pip3 install --no-cache-dir --no-build-isolation \
    flash-attn

# other packages
RUN pip3 install --no-cache-dir \
    pre-commit \
    wandb \
    numpy \
    rich \
    h5py \
    tqdm \
    pyrallis \
    omegaconf \
    huggingface_hub

# gymnasium and metaworld installation
RUN pip3 install opencv-python==4.9.0.80
RUN pip3 install huggingface_hub
RUN pip3 install gymnasium==1.0.0
RUN pip3 install "gymnasium[accept-rom-license,mujoco]"
RUN pip3 install "metaworld @ git+https://github.com/qgallouedec/Metaworld@gym2.6_register"
RUN pip3 install "mujoco<3"

RUN chmod -R 777 /home/user/conda/lib/python3.8/site-packages/mujoco_py/

USER user
