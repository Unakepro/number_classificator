cmake_minimum_required(VERSION 3.16)
project(number_classificator)

set (CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

find_package(SFML 2.5 COMPONENTS system window graphics network audio REQUIRED)
set(EIGEN3_INCLUDE_DIR "eigen-3.3.9")

add_executable(ProjExec main.cpp regression/mnist_regression.cpp)
target_include_directories(ProjExec PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(ProjExec sfml-system sfml-window sfml-graphics sfml-network sfml-audio)

# include(FetchContent)
# FetchContent_Declare(SFML
#     GIT_REPOSITORY https://github.com/SFML/SFML.git
#     GIT_TAG 2.6.x
#     GIT_SHALLOW ON
#     EXCLUDE_FROM_ALL
#     SYSTEM)
# FetchContent_MakeAvailable(SFML)

# add_executable(main main.cpp)
# target_link_libraries(main PRIVATE sfml-graphics)
# target_compile_features(main PRIVATE cxx_std_17)

# if(WIN32)
#     add_custom_command(
#         TARGET main
#         COMMENT "Copy OpenAL DLL"
#         PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${SFML_SOURCE_DIR}/extlibs/bin/$<IF:$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>,x64,x86>/openal32.dll $<TARGET_FILE_DIR:main>
#         VERBATIM)
# endif()