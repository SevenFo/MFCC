cmake_minimum_required(VERSION 3.21)
project(MFCC)

set(CMAKE_CXX_STANDARD 11)
set(ONNX_DIR onnxruntime-linux-x64-1.10.0)

set(EIGEN_DIR "eigen-3.4.0")
link_directories("${ONNX_DIR}/lib")
include_directories(
        "include",
        "${ONNX_DIR}/include",
        ${EIGEN_DIR}
)

add_executable(MFCC main.cpp CsvPraser.cpp CsvPraser.h test.cpp)
add_executable(test test.cpp CsvPraser.cpp CsvPraser.h test.cpp)

target_link_libraries(MFCC "onnxruntime")
target_link_libraries(test "onnxruntime")
