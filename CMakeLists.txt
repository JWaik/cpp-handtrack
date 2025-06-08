cmake_minimum_required(VERSION 3.10)
project(cpp-handtrack)

set(CMAKE_CXX_STANDARD 17)

# ----------- OpenCV -----------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ----------- Sources and Includes -----------
include_directories(include)
file(GLOB SOURCES "src/*.cpp")

# ----------- cxxopts via submodule -----------
# May change to FetchConetent later
add_subdirectory(third_party/cxxopts/)

# ----------- Build Target -----------
add_executable(cpp_handtrack ${SOURCES})
target_link_libraries(cpp_handtrack PRIVATE ${OpenCV_LIBS})
target_include_directories(cpp_handtrack PUBLIC third_party/cxxopts/include/)