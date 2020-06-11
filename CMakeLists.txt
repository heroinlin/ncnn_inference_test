#1.cmake verson，指定cmake的最小版本号 
cmake_minimum_required(VERSION 2.8)

#2.project name，指定项目的名称，一般和项目的文件夹名称对应
project(ncnn_test C CXX)

#3.set environment variable，设置环境变量
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(NCNN_PATH /home/heroin/workspace/thirdparty/ncnn)
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
#4.include头文件目录 
include_directories(${NCNN_PATH}/include
                    ${CMAKE_CURRENT_LIST_DIR}/src)

#5.library目录及name名称
link_directories(${NCNN_PATH}/lib)
list(APPEND LINKER_LIBS ncnn)


#6.source directory源文件目录
file(GLOB SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.h
                    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
set(COMPILE_CODE ${SRC})

#7.1.add executable file，编译为可执行文件
add_executable(ncnn_test ${COMPILE_CODE})
#7.2.add library file，编译为动态库
# add_library(ncnn_test SHARED ${COMPILE_CODE})

#8.add link library，添加工程所依赖的库
target_link_libraries(ncnn_test ${LINKER_LIBS} -lz)
