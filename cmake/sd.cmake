include(FetchContent)
# FetchContent_MakeAvailable was not added until CMake 3.14
if (${CMAKE_VERSION} VERSION_LESS 3.14)
    include(add_FetchContent_MakeAvailable.cmake)
endif ()

set(SD_GIT_TAG becb26df0d2427b19480b982e0fcf4b1f7112b67)
set(SD_GIT_URL https://github.com/ring-c/stable-diffusion.cpp)
#set(BUILD_SHARED_LIBS OFF)

FetchContent_Declare(
        sd
        GIT_REPOSITORY ${SD_GIT_URL}
        GIT_TAG ${SD_GIT_TAG}
)
FetchContent_MakeAvailable(sd)

set(GGML_AVX512 OFF)
set(GGML_AVX2 OFF)
set(GGML_AVX OFF)
