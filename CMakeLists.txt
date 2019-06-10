cmake_minimum_required(VERSION 3.5)

project(zimiao)
set(OpenCV_DIR /usr/share/opencv)
set(CMAKE_CXX_STANDARD 11)
SET(OpenCV_DIR /usr/local/opencv3/share/OpenCV)
SET(CMAKE_BUILD_TYPE "Debug") 
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

include_directories(${PROJECT_SOURCE_DIR}/include)
#include_directories(${PROJECT_SOURCE_DIR}/libHardWare/serialPort/include)

#include_directories(${PROJECT_SOURCE_DIR}/libHardWare/mindVision/include)



#ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/libHardWare)


AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

add_executable(example ${DIR_SRCS})

#target_link_libraries(example SerialPort  MindVision)
#target_link_libraries(example MVSDK)

find_package(OpenCV 3.3.1 REQUIRED)
target_link_libraries(example ${OpenCV_LIBS})
