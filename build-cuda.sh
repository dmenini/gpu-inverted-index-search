#!/bin/bash
#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli

#Titan X: compute_52, TX1: compute_53, GTX1080Ti: compute_61, TX2: compute_62

nvccflags="-O3 --use_fast_math -g -G -std=c++11 -Xcompiler '-fopenmp' --gpu-architecture=compute_61 --compiler-options -fPIC --linker-options --no-undefined"
nvcc -o bin/query-cu.out query.cu $nvccflags

./bin/query-cu.out ./out/inv_dictionary.int ./out/query.txt ./out/files.txt ./out

# nvccflags="-O3 --use_fast_math -std=c++11 -Xcompiler '-fopenmp' --shared --gpu-architecture=compute_61 --compiler-options -fPIC --linker-options --no-undefined"
# nvcc -o cbconv2d_cg_half_backend_$(uname -i).so cbconv2d_cg_half_backend.cu $nvccflags