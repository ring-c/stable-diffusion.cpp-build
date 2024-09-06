set -ex

#Release
#Debug

#reset
export CUDACXX=/usr/local/cuda-12/bin/nvcc

cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=OFF -DFETCHCONTENT_SOURCE_DIR_SD:PATH=/go/gh/stable-diffusion.cpp
cmake --build . --config Release
mv -f ./bin/libsd-abi.so /go/gh/go-web-diff/pkg/bind/deps/linux/
