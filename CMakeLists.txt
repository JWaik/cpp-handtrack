cmake_minimum_required(VERSION 3.10)
project(cpp-handtrack)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(cpp_handtrack ${SOURCES})
target_link_libraries(cpp_handtrack ${OpenCV_LIBS})