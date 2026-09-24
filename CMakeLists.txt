cmake_minimum_required(VERSION 3.12)
project(riso_approximation_tool LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

if(APPLE)
    set(ONNXRUNTIME_ROOT "/opt/homebrew" CACHE PATH "onnxruntime install prefix (Homebrew)")
else()
    set(ONNXRUNTIME_ROOT "/usr/local" CACHE PATH "onnxruntime install prefix")
endif()
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARY_DIRS "${ONNXRUNTIME_ROOT}/lib")
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_LIBRARY_DIRS})

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "onnxruntime library not found under ${ONNXRUNTIME_LIBRARY_DIRS}. Install onnxruntime or set ONNXRUNTIME_ROOT.")
endif()

add_executable(riso_tool
    main.cpp
    scoring.cpp
    image_utils.cpp
    riso_output.cpp
    filter.cpp
    faceDetect.cpp
    img_pipeline.cpp
)

target_include_directories(riso_tool PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_directories(riso_tool PRIVATE ${ONNXRUNTIME_LIBRARY_DIRS})
target_link_libraries(riso_tool PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

# Running from build/ uses ./haarcascade_frontalface_alt2.xml by default.
add_custom_command(TARGET riso_tool POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/haarcascade_frontalface_alt2.xml"
        "$<TARGET_FILE_DIR:riso_tool>/haarcascade_frontalface_alt2.xml"
)
