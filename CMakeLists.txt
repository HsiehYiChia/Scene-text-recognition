cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
project(scene_text_recognition)

function(get_msvc_version OUT_VERSION)
    set(V ${CMAKE_GENERATOR})
    separate_arguments(V)
    list(GET V 2 V)
    set(${OUT_VERSION} ${V} PARENT_SCOPE)
endfunction()

function(create_wrapper_script target dest_dir)
    message("--== Creating wrapper script")
    get_msvc_version(OPENCV_MSVC_SUFFIX)
    if (${OPENCV_MSVC_SUFFIX} GREATER_EQUAL 15)
        set(OPENCV_MSVC_SUFFIX 15)
    endif()
    
    if(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set(OPENCV_PLT_SUFFIX "x64")
    else()
        set(OPENCV_PLT_SUFFIX "x86")
    endif()
    
    set(OPENCV_LIB_DIR "${WIN_OPENCV_CONFIG_PATH}\\${OPENCV_PLT_SUFFIX}\\vc${OPENCV_MSVC_SUFFIX}\\bin")
    set(CONTENTS "setlocal\nset path=%path%" "${OPENCV_LIB_DIR}\n${target} %*\nendlocal")
    file(WRITE "${dest_dir}\\${target}.bat" "${CONTENTS}")
endfunction()


# Set additional paths to CMake scripts.
# Tested with OpenCV 4.1 through 4.3.
# If needed, change the path to where OpenCVConfig.cmake is.
if (MSVC)
    set(WIN_OPENCV_CONFIG_PATH "C:\\tools\\opencv\\build")
    list(APPEND CMAKE_PREFIX_PATH ${WIN_OPENCV_CONFIG_PATH})
endif()

# Find required packages
find_package(OpenCV REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS           "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Gather platform-specific definitions
if (MSVC)
    set(TARGET_CXX_DEFS
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTDC_NO_DEPRECATE
        UNICODE
        _UNICODE
    )
else()
    set(TARGET_CXX_DEFS)
endif()



# Gather source and header files
set(STR_TARGET_SOURCES
    src/adaboost.cpp 
    src/ER.cpp 
    src/main.cpp 
    src/OCR.cpp 
    src/SpellingCorrector.cpp 
    src/svm.cpp
    src/utils.cpp
)

set(STR_TARGET_HEADERS
    inc/adaboost.h 
    inc/ER.h 
    inc/getopt.h
    inc/OCR.h 
    inc/SpellingCorrector.h 
    inc/svm.h
    inc/utils.h
)

# Add target for scene text recognition
add_executable(scene_text_recognition
    ${STR_TARGET_HEADERS}
    ${STR_TARGET_SOURCES}
)

# Add target settings
target_include_directories(scene_text_recognition PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(scene_text_recognition PRIVATE ${OpenCV_LIBS})
target_compile_definitions(scene_text_recognition PRIVATE ${TARGET_CXX_DEFS})

# Add source file groups
source_group(inc FILES ${STR_TARGET_HEADERS})
source_group(src FILES ${STR_TARGET_SOURCES})

# Create run file
if (MSVC)
    create_wrapper_script(scene_text_recognition ${CMAKE_CURRENT_LIST_DIR})
endif()



# Gather source and header files
set(SVM_TARGET_SOURCES
    src/svm-train.cpp
    src/svm.cpp
)

set(SVM_TARGET_HEADERS
    inc/svm.h
)

# Add target svm-train from libsvm
add_executable(svm-train
    ${SVM_TARGET_HEADERS}
    ${SVM_TARGET_SOURCES}
)

# Add target settings
target_compile_definitions(svm-train PRIVATE ${TARGET_CXX_DEFS})

# Add source file groups
source_group(inc FILES ${SVM_TARGET_HEADERS})
source_group(src FILES ${SVM_TARGET_SOURCES})

# Post-build events
add_custom_command(TARGET scene_text_recognition POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:scene_text_recognition>
        ${CMAKE_CURRENT_LIST_DIR}
)

add_custom_command(TARGET svm-train POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:svm-train>
        ${CMAKE_CURRENT_LIST_DIR}
)
