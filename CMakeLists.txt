cmake_minimum_required(VERSION 3.17)

set(PROJECT_NAME undistort_test)
project(${PROJECT_NAME} LANGUAGES CXX C VERSION 0.0.1 DESCRIPTION "undistort_test")

set(CMAKE_VERBOSE_MAKEFILE TRUE)

# .lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# .dll and .exe
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin) 

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

if (WIN32)
# option for Visual Studio
# -EHsc (enable proper Exxeption Handling) needs to avoid C4530
# -Wall is too noisy so that set -W4. 
# https://docs.microsoft.com/en-us/cpp/build/reference/compiler-option-warning-level?view=vs-2017
# "However, for a new project, it may be best to use /W4 in all compilations;
# this will ensure the fewest possible hard-to-find code defects."
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /EHsc /MP /bigobj")

add_definitions(-DNOMINMAX)

else()
# g++/clang option for *nix
set(CMAKE_CXX_EXTENSIONS OFF) #Set this to ON if you want to use GNU++
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -O2")
endif()

get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  message(STATUS "Has a parent scope.")
else()
  message(STATUS "Doesn't have a parent scope.")
endif()

function(set_as_cache ARG_NAME ARG_PATH ARG_TEXT)
    set(${ARG_NAME} ${ARG_PATH} CACHE PATH ${ARG_TEXT} FORCE)
endfunction(set_as_cache)

find_package(OpenCV REQUIRED)

add_executable(undistort_test main.cc)
target_include_directories(undistort_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(undistort_test PRIVATE ${OpenCV_LIBS})

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_LIST})