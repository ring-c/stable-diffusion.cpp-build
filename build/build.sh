set -ex

#Release
#Debug

#reset
export CUDACXX=/usr/local/cuda-12/bin/nvcc

cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=OFF -DSD_BUILD_EXAMPLES=OFF -DFETCHCONTENT_SOURCE_DIR_SD:PATH=/code/github/ring-c/stable-diffusion.cpp
cmake --build . --config Release -j 10
mv -f ./bin/libsd-abi.so /code/github/ring-c/go-web-diff/pkg/bind/deps/linux/
