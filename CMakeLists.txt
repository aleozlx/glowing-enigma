cmake_minimum_required(VERSION 3.3)
project(glowing-enigma)
set(OpenGL_GL_PREFERENCE GLVND)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

# GUI Packages
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(SOIL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

# Computer Vision Packages
find_package(OpenCV REQUIRED)
# find_package(libusb-1.0 REQUIRED)

# Machine Learning Packages
find_package(TensorFlow)
find_package(CUDA 10)

# More Packages
find_package(Boost)
find_package(Spfreq2)

# Feature definitions
add_definitions("-DVER_OPENCV=${OpenCV_VERSION}")
if(CUDA_FOUND)
    add_definitions("-DHAS_CUDA=${CUDA_VERSION_STRING}")
endif()
if(TENSORFLOW_FOUND)
    add_definitions("-DHAS_TF=${TensorFlow_VERSION}")
endif()

# ImGui Library
set(root ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
set(phdrs ${root}/imgui.h)
set(hdrs ${root}/imstb_rectpack.h ${root}/imstb_textedit.h ${root}/imstb_truetype.h
    ${root}/imgui_internal.h)
set(srcs ${root}/imgui.cpp
    ${root}/imgui_demo.cpp ${root}/imgui_draw.cpp ${root}/imgui_widgets.cpp)

add_library(imgui STATIC ${phdrs} ${hdrs} ${srcs})
target_include_directories(imgui PUBLIC
    $<BUILD_INTERFACE:${root}>
    $<INSTALL_INTERFACE:include>)
target_compile_definitions(imgui PUBLIC IMGUI_DISABLE_INCLUDE_IMCONFIG_H)

# gSLIC Library (optional)
if(CUDA_FOUND)
    set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
    set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(GSLICR_LIB
        gSLICr/gSLICr_Lib/engines/gSLICr_core_engine.h
        gSLICr/gSLICr_Lib/engines/gSLICr_seg_engine.h
        gSLICr/gSLICr_Lib/engines/gSLICr_seg_engine_GPU.h
        gSLICr/gSLICr_Lib/engines/gSLICr_seg_engine_shared.h
        gSLICr/gSLICr_Lib/engines/gSLICr_core_engine.cpp
        gSLICr/gSLICr_Lib/engines/gSLICr_seg_engine.cpp
        gSLICr/gSLICr_Lib/engines/gSLICr_seg_engine_GPU.cu
        gSLICr/gSLICr_Lib/objects/gSLICr_settings.h
        gSLICr/gSLICr_Lib/objects/gSLICr_spixel_info.h
        gSLICr/gSLICr_Lib/gSLICr_defines.h
        gSLICr/gSLICr_Lib/gSLICr.h
    )
    cuda_add_library(gSLICr
        ${GSLICR_LIB}
        OPTIONS
            -gencode arch=compute_30,code=compute_30
            -gencode arch=compute_61,code=compute_61)
    target_link_libraries(gSLICr ${CUDA_LIBRARY})
endif()

# Global Config
include_directories(SYSTEM "/usr/include/opencv4")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/gSLICr)

# ImGui Demo
add_executable(gui_demo ${CMAKE_CURRENT_SOURCE_DIR}/demos/imgui.cpp ${root}/examples/imgui_impl_glfw.cpp ${root}/examples/imgui_impl_opengl3.cpp)
target_compile_definitions(gui_demo PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)
target_include_directories(gui_demo PUBLIC ${root}/examples)
target_link_libraries(gui_demo imgui ${SOIL_LIBRARY} ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})

# Superpixel Analyzer
add_executable(superpixel_analyzer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/app.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/superpixel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dcnn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/misc_ocv.cpp
    ${root}/examples/imgui_impl_glfw.cpp ${root}/examples/imgui_impl_opengl3.cpp)
target_compile_definitions(superpixel_analyzer PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)
if(CUDA_FOUND)
    target_compile_definitions(superpixel_analyzer PUBLIC HAS_LIBGSLIC)
    target_link_libraries(superpixel_analyzer gSLICr)
endif()
if(TENSORFLOW_FOUND)
    target_include_directories(superpixel_analyzer PUBLIC ${TensorFlow_INCLUDE_DIR})
    target_link_libraries(superpixel_analyzer -Wl,--allow-multiple-definition -Wl,--whole-archive ${TensorFlow_C_LIBRARY} -Wl,--no-whole-archive)
    target_link_libraries(superpixel_analyzer -Wl,--allow-multiple-definition -Wl,--whole-archive ${TensorFlow_LIBRARY} -Wl,--no-whole-archive)
endif()
target_include_directories(superpixel_analyzer PUBLIC ${root}/examples)
target_link_libraries(superpixel_analyzer imgui ${SOIL_LIBRARY} ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES}
    opencv_core
    opencv_dnn
    opencv_imgproc
    opencv_ximgproc
    opencv_imgcodecs
    opencv_videoio
    opencv_video
)

# Superpixel Processor
add_executable(superpixel_process
    ${CMAKE_CURRENT_SOURCE_DIR}/src/process.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/superpixel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dcnn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/misc_ocv.cpp
)
if(CUDA_FOUND)
    target_compile_definitions(superpixel_process PUBLIC HAS_LIBGSLIC)
    target_link_libraries(superpixel_process gSLICr)
endif()
if(TENSORFLOW_FOUND)
    target_include_directories(superpixel_process PUBLIC ${TensorFlow_INCLUDE_DIR})
    target_link_libraries(superpixel_process -Wl,--allow-multiple-definition -Wl,--whole-archive ${TensorFlow_C_LIBRARY} -Wl,--no-whole-archive)
    target_link_libraries(superpixel_process -Wl,--allow-multiple-definition -Wl,--whole-archive ${TensorFlow_LIBRARY} -Wl,--no-whole-archive)
endif()
target_link_libraries(superpixel_process
    opencv_core
    opencv_dnn
    opencv_imgproc
    opencv_ximgproc
    opencv_imgcodecs
    opencv_videoio
    opencv_video
)

# gSLIC Demo
add_executable(gslic_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/demos/gslic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/app.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/superpixel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/misc_ocv.cpp
    ${root}/examples/imgui_impl_glfw.cpp ${root}/examples/imgui_impl_opengl3.cpp)
target_compile_definitions(gslic_demo PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)
if(CUDA_FOUND)
    target_compile_definitions(gslic_demo PUBLIC HAS_LIBGSLIC)
    target_link_libraries(gslic_demo gSLICr)
endif()
target_include_directories(gslic_demo PUBLIC ${root}/examples)
target_link_libraries(gslic_demo imgui ${SOIL_LIBRARY} ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES}
    opencv_core
    opencv_imgproc
    opencv_ximgproc
    opencv_imgcodecs
    opencv_videoio
    opencv_video
)

# set(targets imgui)
# set(headers_to_install ${phdrs})

# install(TARGETS ${targets}
#     EXPORT imgui-targets DESTINATION lib)
# install(EXPORT imgui-targets
#     FILE imgui-config.cmake
#     NAMESPACE imgui::
#     DESTINATION lib/cmake/imgui)
# install(FILES ${headers_to_install}
#     DESTINATION include)
