FROM cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36

# Specifies that all subsequent commands must be executed as the root user
USER root

RUN apt-get update --fix-missing --allow-unauthenticated && apt-get upgrade -y &&\
 apt-get install --fix-missing -qy\
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

# Install torch and flashattn
RUN pip3 install "Cython<3"
RUN pip3 uninstall --yes torch
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir --no-build-isolation \
    ninja \
    setuptools \
    packaging \
    pydantic

RUN python -m pip install --upgrade pip wheel setuptools

RUN pip3 install --no-cache-dir --no-build-isolation \
    flash-attn

RUN pip3 install --no-cache-dir \
    einops \
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

# Install nessesary libraries
RUN pip3 install stable-baselines3
RUN pip3 install wandb
RUN pip3 install gym
RUN pip3 install pyrallis

# Copy patches
RUN mkdir /home/industrial_benchmark_patches
COPY industrial_benchmark_ibgym.patch /home/industrial_benchmark_patches
COPY industrial_benchmark_ids.patch /home/industrial_benchmark_patches

# Clonning Industrial Benchmark
RUN git config --global http.postBuffer 1048576000 && \
    git config --global https.postBuffer 1048576000 && \
    git config --global http.version HTTP/1.1
RUN git clone https://github.com/siemens/industrialbenchmark.git

# Installing Industrial Benchmark
RUN cd industrialbenchmark && \
    patch industrial_benchmark_python/IDS.py /home/industrial_benchmark_patches/industrial_benchmark_ids.patch && \
    patch industrial_benchmark_python/IBGym.py /home/industrial_benchmark_patches/industrial_benchmark_ibgym.patch && \
    python -m pip install -e .

# Change user to 'user'
USER user