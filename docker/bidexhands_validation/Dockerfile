FROM cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py38:0.0.36

# Specifies that all subsequent commands must be executed as the root user
USER root

# Update the list of packages in the system and already installed packages
RUN apt-get update --fix-missing && apt-get upgrade -y  &&\
    echo "8" apt-get install -y software-properties-common && apt-get update

## Install the necessary programs and libraries
RUN apt install -qy curl libpython3.8-dev

# Install torch and flashattn
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
WORKDIR /home/user

RUN pip3 install "Cython<3"
RUN pip3 uninstall --yes torch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir --no-build-isolation \
    ninja \
    setuptools \
    packaging \
    pydantic

RUN pip3 install --no-cache-dir --no-build-isolation \
    flash-attn

RUN pip3 install --no-cache-dir \
    lightning-utilities \
    huggingface_hub \
    pre-commit \
    gymnasium \
    matplotlib \
    wandb \
    numpy \
    rich \
    h5py \
    tqdm \
    imageio \
    imageio-ffmpeg \
    pyrallis \
    omegaconf

# Copy patches
RUN mkdir /home/bidexhands_patches
RUN mkdir /home/isaacgym_patches
COPY bidexhands.patch /home/bidexhands_patches
COPY isaacgym_torch_utils.patch /home/isaacgym_patches

# Installing IsaacGym
# WORKDIR /home/user
RUN wget https://developer.nvidia.com/isaac-gym-preview-4
RUN tar -xf isaac-gym-preview-4
RUN cd isaacgym/python && \
    patch isaacgym/torch_utils.py /home/isaacgym_patches/isaacgym_torch_utils.patch && \
    python -m pip install -e .

# Clonning Bi-DEX Hands
RUN git config --global http.postBuffer 1048576000 && \
    git config --global https.postBuffer 1048576000 && \
    git config --global http.version HTTP/1.1
RUN git clone https://github.com/PKU-MARL/DexterousHands.git

# Installing Bi-DEX Hands
RUN cd DexterousHands && \
    patch bidexhands/utils/config.py /home/bidexhands_patches/bidexhands.patch && \
    python -m pip install -e .
RUN python -m pip install importlib-resources wandb

# Debug python
RUN cp /home/user/conda/lib/libpython3.8.so.1.0 /usr/lib/

# Change user to 'user'
USER user
