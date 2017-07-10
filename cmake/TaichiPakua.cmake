set(PAKUA_LIBRARY_NAME pakua)

file(GLOB PAKUA_SOURCE
        "src/*/*/*/*.cpp" "src/*/*/*.cpp" "src/*/*.cpp" "src/*.cpp"
        "src/*/*/*/*.h" "src/*/*/*.h" "src/*/*.h" "src/*.h"
        "include/taichi/*/*/*/*.cpp" "include/taichi/*/*/*.cpp" "include/taichi/*/*.cpp"
        "include/taichi/*/*/*/*.h" "include/taichi/*/*/*.h" "include/taichi/*/*.h")

add_library(${PAKUA_LIBRARY_NAME} SHARED ${PAKUA_SOURCE})

include_directories(pakua/include)
include_directories(pakua/external/include)
