cmake_minimum_required(VERSION 3.3)
project(imgui)
set(OpenGL_GL_PREFERENCE GLVND)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(SOIL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

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

add_executable(my_demo ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${root}/examples/imgui_impl_glfw.cpp ${root}/examples/imgui_impl_opengl3.cpp)
target_compile_definitions(my_demo PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)
target_include_directories(my_demo PUBLIC ${root}/examples)
target_link_libraries(my_demo imgui ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})

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
