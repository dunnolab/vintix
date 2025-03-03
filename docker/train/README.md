# Python environment for training Vintix

## Building docker image
Run the following command to build the image:

```shell
docker build --platform linux/amd64 -t vintix-train-image:version1 .
```

## Manual installation
To set up the Python environment for training Vintix on your own, you must install the following packages:
```shell
numpy>=1.24.4
torch
flash-attn
h5py
tqdm
pyrallis
omegaconf
wandb
```

However, ensure that `flash-attn` is compatible with the installed `torch`
and CUDA version. For example, for CUDA 11.8, you can use the following command:

```shell
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
pip3 install --no-cache-dir --no-build-isolation \
    ninja \
    setuptools \
    packaging \
    pydantic
pip3 install --no-cache-dir --no-build-isolation flash-attn

pip3 install --no-cache-dir \
    wandb \
    numpy \
    h5py \
    tqdm \
    pyrallis \
    omegaconf
```

The versions of the packages the model was trained on are:
- CUDA 11.8
- Python 3.11.5
- Package versions
    ```shell
    torch==2.5.1+cu118
    flash-attn==2.7.0.post2
    numpy==1.24.1
    ```
