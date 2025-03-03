# Python environment for using Vintix

## Building docker image
Run the following command to build the image:

```shell
docker build --platform linux/amd64 -t vintix-val-image:version1 .
```

## Manual installation
For inference Vintix model `numpy>=1.24.4`, `torch` and `flash-attn` packages are required. However, ensure that `flash-attn` is compatible with the installed `torch` and CUDA version. For example, for CUDA 11.8, you can use the following command:
```shell
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
pip3 install --no-cache-dir --no-build-isolation \
    ninja \
    setuptools \
    packaging \
    pydantic \
    wheel
pip3 install --no-cache-dir --no-build-isolation flash-attn
```

`gymnasium>=0.28.0`, `mujoco<3` and `imageio` are required to create MuJoCo tasks.

To create Meta-World tasks using gym.make, you should install [the Meta-World package](https://github.com/qgallouedec/Metaworld/tree/gym2.6_register):
```shell
pip3 install "metaworld @ git+https://github.com/qgallouedec/Metaworld@gym2.6_register"
```

The Meta-World requires the `mujoco_py` package, which depends on MuJoCo binaries: [more information here](https://github.com/openai/mujoco-py?tab=readme-ov-file).
```shell
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
mkdir $HOME/.mujoco
mv mujoco210 $HOME/.mujoco/
```

After installation some environment variables should be updated:

```shell
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
Moreover, to compile this library, you must install `cython<3`. It’s also recommended to install the following Linux packages, as you may encounter compilation errors without them:
```shell
sudo apt-get update --fix-missing && sudo apt-get upgrade -y &&\
 sudo apt-get install -qy\
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
```
For Ubuntu versions newer than 23.10, you need to replace  `libgl1-mesa-glx` with `libgl1` and `libglx-mesa0` in the installation command above.

`metaworld` package should be imported before `gymnasium` to create Meta-World tasks using `gymnasium.make`:
```python
import metaworld
import gymnasium as gym

env = gym.make('assembly-v2')
```

The versions of the packages the model was validated on are:
- CUDA 11.8
- Python 3.11.5
- Package versions
    ```shell
    torch==2.5.1+cu118
    flash-attn==2.7.0.post2
    numpy==1.24.1
    gymnasium==1.0.0
    mujoco==2.3.7
    ```
- Meta-World installed from repository mentioned above

