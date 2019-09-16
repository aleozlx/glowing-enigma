# Ways to get dependencies

## Arch Linux

Use system packages for all dependencies

```
pacman -Sy cmake clang cuda opencv tensorflow-cuda postgresql libpqxx
pacman -Sy glew soil glfw-x11
pacman -Sy boost boost-libs
```

Spfreq2

```
# git-clone the repo
mkdir build && cd build
cmake ..
make
cp -fv spfreq2_op.so /usr/lib/
```

## CentOS

OpenCV

```sh
# git-clone the opencv and contrib repo, then
git checkout 4.1.1
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D ENABLE_CXX11=ON ..
make -j$(nproc)
make install
```

TensorFlow

```
yum install centos-release-scl
yum install rh-python36
scl enable rh-python36 bash
pip install tensorflow-gpu
```

PostgreSQL

```
yum install postgresql-server postgresql-contrib
```

gcc8

```
yum install centos-release-scl
yum install devtoolset-8-gcc devtoolset-8-gcc-c++
scl enable devtoolset-8 bash
```

cmake

```sh
# git-clone the cmake repo
./bootstrap --system-curl
make -j$(nproc)
make install
```

libpqxx

```sh
# git-clone the cmake repo
mkdir build && cd build
cmake ..
make -j$(nproc)
make install
```

clang++

```
yum install lzo lzop mbuffer perl-Config-IniFiles pv perl-Data-Dumper
# git-clone the llvm-project repo
git checkout llvmorg-8.0.1
mkdir build && cd build
# USE GCC8!
cmake -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi;openmp" \
      -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
make install
```

Spfreq2: see Arch Linux ^

## Ubuntu

OpenCV: see CentOS ^

TensorFlow

```
pip install tensorflow-gpu
```

PostgreSQL

```
apt install postgresql postgresql-contrib
```

libpqxx

```
apt install libpqxx-dev
```

clang++

```
apt install build-essential xz-utils curl
# download from releases.llvm.org and decompress and add to PATH
```

Spfreq2: see Arch Linux ^

GUI

```
apt install libglew-dev libsoil-dev libglfw3 libglfw3-dev
```

Boost

```
apt install  libboost-all-dev
```
