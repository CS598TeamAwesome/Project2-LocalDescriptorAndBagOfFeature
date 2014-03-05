cmake_minimum_required (VERSION 2.6)

project(LocalDescriptorAndBagOfFeature)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE)
set(HEADERS)

add_subdirectory(Extraction)
add_subdirectory(Util)

add_library(LocalDescriptorAndBagOfFeature ${SOURCE} ${HEADERS})
target_link_libraries(LocalDescriptorAndBagOfFeature ${OpenCV_LIBS})

add_executable(Test Test.cpp)
target_link_libraries(Test LocalDescriptorAndBagOfFeature)
