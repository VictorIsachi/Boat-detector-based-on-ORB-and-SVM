cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(boat_detector)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(include)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(boat_detector
			   src/boat_detector_main.cpp
			   src/image_boxer.cpp
			   src/feature_classifier.cpp)

target_link_libraries(boat_detector ${OpenCV_LIBS})