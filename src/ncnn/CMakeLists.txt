cmake_minimum_required(VERSION 3.20) # 修正版本号，4.1 尚未发布
project(ncnntest)

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    add_compile_options(/utf-8 /DNOMINMAX)
endif()

add_library(NcnnDet SHARED NcnnDet.cpp NcnnDet.h)


find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

target_include_directories(NcnnDet PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${ncnn_INCLUDE_DIRS}
)

target_link_libraries(NcnnDet PUBLIC
        ${OpenCV_LIBS}
        ncnn
)

set_target_properties(NcnnDet PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/x64/DllRelease"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/x64/DllRelease"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/x64/DllRelease"
)

install(TARGETS NcnnDet
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)
install(FILES NcnnDet.h DESTINATION include)

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libopencv-dev, libncnn-dev")
include(CPack)