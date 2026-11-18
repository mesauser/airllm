export USE_ROCM=0
export USE_NCCL=0
export USE_CUDA=1
export USE_CUDNN=1
export USE_MPI=0
export USE_OPENMPI=0
export USE_KINETO=0
export USE_CUPTI=0
export KINETO_BUILD_TYPE=OFF
export USE_OPENMP=0
export PYTORCH_BUILD_VERSION=2.10.0
export PYTORCH_BUILD_NUMBER=0
export TORCH_CUDA_ARCH_LIST="8.9"
# export TORCH_CUDA_ARCH_LIST="8.9+PTX"
# export TORCH_CUDA_ARCH_LIST="10.0"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export CUDA_VISIBLE_DEVICES=0
conda activate env3

cmakepy(){
# git clone  http://github.com/pytorch/pytorch.git
# git checkout -b py210 v2.10.0
# git clone  https://github.com/vllm-project/vllm.git
    git submodule update --init --recursive
    git submodule foreach --recursive 'git checkout -- . && git clean -fd'
    git clone https://github.com/NVIDIA/nccl third_party/nccl
python setup.py develop   
cmake ../ -GNinja -DCMAKE_INSTALL_PREFIX -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF  -DCMAKE_CUDA_ARCHITECTURES=100 -DCMAKE_EXECUTE_PROCESS_COMMAND_ECHO=STDOUT
rm -rf build
pip install -e . --no-build-isolation
python setup.py build   
python setup.py build --cmake #否则不执行cmake
python setup.py   bdist_wheel 
python setup.py clean   

python -m build --no-isolation --wheel
pip install  --no-build-isolation -v . --no-deps

  libpython3-dbg libpython3.10-dbg python3-dbg python3.10-dbg
sysctl kernel.yama.ptrace_scope
sudo sysctl -w kernel.yama.ptrace_scope=0
sudo sysctl -w kernel.yama.ptrace_scope=1

sudo env PATH=$PATH py-spy dump   --nonblocking  --pid 359861 

#允许任何进程ptrace 自己
enable_self_ptrace(){
import ctypes
libc = ctypes.CDLL(None)
PR_SET_PTRACER = 0x59616d61
PR_SET_PTRACER_ANY = -1
libc.prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)
}
# gwj
# set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/pytorch-install" CACHE PATH "" FORCE)
# # unset(CMAKE_INSTALL_PREFIX CACHE)
include_directories(BEFORE /usr/local/cuda-13.0/include)
add_definitions(-D_FILE_OFFSET_BITS=64)
add_compile_options(-Wno-unused-parameter)  # skip 3rdpt warning, disable this unless you known what happens
# # target_compile_options(fbgemm PRIVATE -Wno-unused-parameter)
# # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-unused-parameter")

# use export export TORCH_CUDA_ARCH_LIST="9.0"
# set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures")
# set(TORCH_CUDA_ARCH_LIST "8.9" CACHE STRING "CUDA architectures")
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# Required-by: accelerate, airllm, compressed-tensors, flashinfer-python, quack-kernels, torch_c_dlpack_ext, torchaudio, torchvision, xgrammar
41M     nvidia/cuda_cupti
212M    nvidia/cuda_nvrtc
269M    nvidia/cufft
133M    nvidia/curand
387M    nvidia/cusolver
371M    nvidia/cusparse
432M    nvidia/cusparselt
410M    nvidia/nccl
90M     nvidia/nvjitlink
195M    nvidia/nvshmem
1.2G    flashinfer_cubin
}

cmake_triton(){
export TRITON_OFFLINE_BUILD=1
export LLVM_BUILD_DIR=/data/guowei/git/llvm-project/build/install

LLVM_SYSPATH=$LLVM_BUILD_DIR \
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
JSON_SYSPATH=$PWD/deps/ \
JSON_INCLUDE_DIRS=$PWD/deps/include \
python -m build --wheel --no-isolation
}

cmakevllm(){
python3 setup.py build
export HF_HUB_OFFLINE=1          # 关键：禁止 huggingface_hub 任何网络请求
export HF_HOME=/data/models      # 或把缓存挂到这里

export CUDA_VISIBLE_DEVICES=0
python3 -m vllm.entrypoints.cli.main serve /data/guowei/models/ --host 0.0.0.0 --port 8002 --served-model-name qwen3-3b --max-model-len 4096 --trust-remote-code  --load-format ipc_cache  --model-loader-extra-config '{"fallback":false}' --kv-cache-memory=1G  --gpu-memory-utilization 0.6
python3 -m vllm.entrypoints.cli.main serve /data/guowei/models/ --host 0.0.0.0 --port 8002 --served-model-name qwen3-3b --max-model-len 4096 --trust-remote-code     --kv-cache-memory=1G  --gpu-memory-utilization 0.6
}

debug_airllm(){
b cudaMemcpyAsync
b cudaLaunchKernel
b cudaLaunchKernelExC

expecttest==0.3.0 click==8.3.3 hypothesis==6.170.12 importlib_metadata==10.1.0 lintrunner==0.13.1 optree==0.20.0 sortedcontainers==2.4.0 spin==0.18 uv==0.13.6 zipp==4.1.2
pip install --upgrade "transformers>=5.8"
}

self_cuda_init(){
# 手动初始化 cuda.
torch.cuda.current_device()
  torch._C._cuda_init()
  torch._C._cuda_emptyCache()
}


