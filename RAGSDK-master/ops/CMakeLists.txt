cmake_minimum_required(VERSION 3.8)
project("mxRAGEmbFramework")
set(CMAKE_CXX_STANDARD 17)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/models/include
        ${PROJECT_SOURCE_DIR}/models
        $ENV{ATB_HOME_PATH}/include
        ${PROJECT_SOURCE_DIR}/../opensource/json/include
        $ENV{ASCEND_HOME_PATH}/include
        $ENV{ASCEND_HOME_PATH}/include/aclnn
        )

link_directories(
        $ENV{ASCEND_HOME_PATH}/lib64
        $ENV{ATB_HOME_PATH}/lib
	    ${PROJECT_SOURCE_DIR}/lib
        $ENV{ASCEND_TOOLKIT_HOME}/lib64
        )

#options, these can be pass in by cmake argument
option(USE_CXX11_ABI "USE_CXX11_ABI" OFF)
option(USE_OPTIMIZE "USE_OPTIMIZE" ON)
option(TORCH_HIGHER_THAN_PTA6 "TORCH_HIGHER_THAN_PTA6" ON)

set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fstack-protector-strong")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -fno-common")

if(USE_OPTIMIZE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic -fpie -Wl,--build-id=none")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common")

set(LD_FLAGS_GLOBAL "-shared -rdynamic -ldl -Wl,-z,relro \
    -Wl,-z,now -Wl,-z,noexecstack -s -Wl,--build-id=none")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} \
    ${LD_FLAGS_GLOBAL} -fexceptions")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LD_FLAGS_GLOBAL} \
    -pie -fPIE")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

if (TARGET_PLATFORM STREQUAL "Ascend310P")
    message(STATUS "target platform is Ascend310P")
    add_definitions(-D Ascend310P)
else()
    message(STATUS "target platform is Ascend910B")
    add_definitions(-D Ascend910B)
endif()

if(TORCH_HIGHER_THAN_PTA6)
    message(STATUS "define TORCH_HIGHER_THAN_PTA6")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_HIGHER_THAN_PTA6")
endif()

add_subdirectory(models)
add_subdirectory(operations)
add_subdirectory(adapter)

install(TARGETS atb_torch atb_models atb_operations DESTINATION lib)