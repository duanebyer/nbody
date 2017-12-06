cmake_minimum_required(VERSION 3.1)

project(nbody)

set(CMAKE_CXX_STANDARD 14)
# Compile as strictly as possible under GCC.
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
		-Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations \
		-Wold-style-cast -pedantic")
endif()
enable_testing()

set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})

set(
	SOURCES
	src/main.cpp
	src/open_cl_simulation.cpp
	src/naive_simulation.cpp)
set(
	KERNEL_SOURCES
	include/nbody/device/types.h
	src/verify.cl
	src/moment.cl
	src/interaction.cl
	src/field.cl
	src/force.cl)

find_package(OpenCL 1.2 REQUIRED)
find_package(GladeLib REQUIRED NO_MODULE)

add_executable(NBody ${SOURCES})
foreach(KERNEL_SOURCE ${KERNEL_SOURCES})
	get_filename_component(KERNEL_TARGET ${KERNEL_SOURCE} NAME)
	add_custom_command(
		OUTPUT ${KERNEL_SOURCE}
		COMMAND ${CMAKE_COMMAND}
		ARGS -E copy ${PROJECT_SOURCE_DIR}/${KERNEL_SOURCE} ${PROJECT_BINARY_DIR}
		COMMENT "Copying kernel file ${KERNEL_TARGET} to build directory")
	add_custom_target(
		${KERNEL_TARGET}
		DEPENDS ${PROJECT_BINARY_DIR}/${KERNEL_SOURCE})
	add_dependencies(
		NBody
		${KERNEL_TARGET})
endforeach(KERNEL_SOURCE)

target_include_directories(
	NBody PRIVATE
	${PROJECT_SOURCE_DIR}/include)
target_include_directories(
	NBody SYSTEM PRIVATE
	${GladeLib_INCLUDE_DIRS}
	${OpenCL_INCLUDE_DIRS})

target_link_libraries(
	NBody
	GladeLib
	${OpenCL_LIBRARIES})

