## Toolchain

* cmake ^3.8
* clang++ 8.0.1
* OpenMP
* CUDA 10.1
* cuDNN ^7.6

## Dependencies

* OpenCV ^4.1.1
* TensorFlow 1.14
  * libtensorflow_cc.so
  * libtensorflow_framework.so
* PostgreSQL ^11.5
* libpqxx ^6.4
* Spfreq2 (dynamically linked) - Superpixel feature extraction (CUDA)
  * spfreq2_op.so
  * requires libtensorflow_framework.so
  * toolchain: cmake ^3.8, gcc ^8, CUDA ^10
* fpconv (submoduled) - Fast float to char[] conversion
* gSLICr (included) - Superpixel segmentation (CUDA)
* ImGUI (submoduled) - Provides all GUI (OpenGL)

## Optional Dependencies

* GUI: OpenGL, GLEW, SOIL, GLFW
* Testing: Boost
* Extra modules: (submoduled @extras/)

## Build

Edit  `include/build_vars.hpp` first, some parameters are defined as macros and built into the executables,
so that we do not have to pass them around all the time.

```sh
mkdir build && cd build
CC=$(which clang) CXX=$(which clang++) cmake ..
```

## Make targets

```
make superpixel_process
make superpixel_analyzer # (GUI)
```
