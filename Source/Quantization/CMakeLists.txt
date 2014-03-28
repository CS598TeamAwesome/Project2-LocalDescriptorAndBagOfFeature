set(SOURCE
    ${SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/CodewordUncertainty.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HardAssignment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Quantization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VocabularyTreeQuantization.cpp
    PARENT_SCOPE
)

set(HEADERS
    ${HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/CodewordUncertainty.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HardAssignment.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Quantization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VocabularyTreeQuantization.hpp
    PARENT_SCOPE
)
