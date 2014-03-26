cmake_minimum_required (VERSION 2.6)

project(LocalDescriptorAndBagOfFeature)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE)
set(HEADERS)

add_subdirectory(Classification)
add_subdirectory(Extraction)
add_subdirectory(Quantization)
add_subdirectory(Util)
add_subdirectory(BagOfFeatures)
add_subdirectory(GMM)
add_subdirectory(SVM)

add_library(LocalDescriptorAndBagOfFeature ${SOURCE} ${HEADERS})
target_link_libraries(LocalDescriptorAndBagOfFeature ${OpenCV_LIBS})

add_executable(Test Test.cpp)
add_executable(BuildCodebook Codebook.cpp)
add_executable(TrainClassifier Classifier.cpp)
add_executable(PerformCategorization Categorize.cpp)
target_link_libraries(Test LocalDescriptorAndBagOfFeature)
target_link_libraries(BuildCodebook LocalDescriptorAndBagOfFeature)
target_link_libraries(TrainClassifier LocalDescriptorAndBagOfFeature)
target_link_libraries(PerformCategorization LocalDescriptorAndBagOfFeature)
