# Installation on CSD3 Ampere GPU Nodes

### first steps
mkdir lammps-mace-gpu
cd lammps-mace-gpu
git clone https://github.com/ACEsuit/mace
git clone --branch mace-kokkos --depth=1 https://github.com/ACEsuit/lammps
wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip
mv libtorch libtorch-gpu

### request interactive node
sintr -t 4:0:0 --exclusive -A CSANYI-SL2-GPU -p ampere

### prepare the environment
conda deactivate
module purge
module load intel-mkl-2017.4-gcc-5.4.0-2tzpyn7
module load rhel8/default-amp
module load cudnn

### compile lammps
mkdir build
cd build
cmake \
    -D CMAKE_INSTALL_PREFIX=$(pwd) \
    -D BUILD_MPI=yes \
    -D BUILD_OMP=no \
    -D PKG_KOKKOS=yes \
    -D Kokkos_ARCH_AMDAVX=yes \
    -D Kokkos_ARCH_AMPERE100=yes \
    -D Kokkos_ENABLE_CUDA=yes \
    -D Kokkos_ENABLE_OPENMP=no \
    -D CMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
    -D PKG_ML-MACE=yes \
    -D CMAKE_PREFIX_PATH=$(pwd)/../../libtorch-gpu \
    -D CMAKE_BUILD_TYPE=Release \
    -D Kokkos_ENABLE_DEBUG=no \
    -D Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=no \
    -D Kokkos_ENABLE_CUDA_UVM=no \
    ../cmake
 make -j 4
 
