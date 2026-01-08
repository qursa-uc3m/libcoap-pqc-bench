cmake_minimum_required(VERSION 3.16)
project(pq-mqtt-sn-clients VERSION 1.0.0 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Define build types if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find wolfMQTT library and headers
find_library(WOLFMQTT_LIBRARY wolfmqtt REQUIRED HINTS /usr/local/lib)
find_path(WOLFMQTT_INCLUDE_DIR wolfmqtt/mqtt_client.h PATHS /usr/local/include REQUIRED)

# Find wolfSSL library and headers
find_library(WOLFSSL_LIBRARY wolfssl REQUIRED HINTS /usr/local/lib)
find_path(WOLFSSL_INCLUDE_DIR wolfssl/ssl.h PATHS /usr/local/include REQUIRED)

# Include directories
include_directories(
    ${WOLFMQTT_INCLUDE_DIR}
    ${WOLFSSL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common
)

# Common source files
set(COMMON_SOURCES
    src/common/mqttexample.c
    src/common/mqttnet.c
    src/common/mqttport.c
)

# Add executables
add_executable(sn-client src/sn-client.c ${COMMON_SOURCES})
add_executable(sn-pub src/sn-pub.c ${COMMON_SOURCES})
add_executable(sn-sub src/sn-sub.c ${COMMON_SOURCES})

# Link executables
foreach(TARGET sn-client sn-pub sn-sub)
    target_link_libraries(${TARGET} PRIVATE ${WOLFMQTT_LIBRARY} ${WOLFSSL_LIBRARY})
endforeach()

# Installation rules
include(GNUInstallDirs)

install(TARGETS sn-client sn-pub sn-sub
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
)