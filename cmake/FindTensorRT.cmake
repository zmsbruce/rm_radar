#[=======================================================================[
FindTensorRT
--------

Find the TensorRT libraries (libnvinfer.so, libonnxparser.so, libnvinfer_plugin.so).

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` targets:

``TensorRT::TensorRT``
  The TensorRT library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``TensorRT_FOUND``
  true if the TensorRT headers and libraries were found
``TensorRT_INCLUDE_DIR``
  the directory containing the TensorRT headers
``TensorRT_INCLUDE_DIRS``
  the directory containing the TensorRT headers
``TensorRT_LIBRARY``
  TensorRT libraries to be linked
``TensorRT_LIBRARIES``
  TensorRT libraries to be linked
#]=======================================================================]


# Add common TensorRT paths
file(GLOB TensorRT_PATHS_ "/usr/local/TensorRT*" "/opt/TensorRT*")
list(APPEND TensorRT_PATHS "/usr" "${TensorRT_PATHS_}")

# Find TensorRT include directory
foreach(search ${TensorRT_PATHS})
    find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h PATHS ${search}/include)
    if (TensorRT_INCLUDE_DIR)
        break()
    endif()
endforeach()

# Find TensorRT libraries
foreach(search ${TensorRT_PATHS})
    find_library(NVONNX_PARSER_LIB NAMES nvonnxparser PATHS ${search}/lib)
    find_library(NVINFER_LIB NAMES nvinfer PATHS ${search}/lib)
    find_library(NVINFER_PLUGIN_LIB NAMES nvinfer_plugin PATHS ${search}/lib)
    if (NVONNX_PARSER_LIB AND NVINFER_LIB AND NVINFER_PLUGIN_LIB)
        break()
    endif()
endforeach()
list(APPEND TensorRT_LIBRARY ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB} ${NVONNX_PARSER_LIB})

# Extract TensorRT version
if (TensorRT_INCLUDE_DIR)
    file(READ "${TensorRT_INCLUDE_DIR}/NvInfer.h" NVINFER_CONTENTS)

    string(REGEX MATCH "#define NV_TENSORRT_MAJOR ([0-9]+)" TensorRT_MAJOR_MATCH "${NVINFER_CONTENTS}")
    string(REGEX MATCH "#define NV_TENSORRT_MINOR ([0-9]+)" TensorRT_MINOR_MATCH "${NVINFER_CONTENTS}")
    string(REGEX MATCH "#define NV_TENSORRT_PATCH ([0-9]+)" TensorRT_PATCH_MATCH "${NVINFER_CONTENTS}")

    if (TensorRT_MAJOR_MATCH AND TensorRT_MINOR_MATCH AND TensorRT_PATCH_MATCH)
        set(TensorRT_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
    endif()
endif()

# Handle find_package(TensorRT REQUIRED)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR VERSION_VAR TensorRT_VERSION)

# Set some variables may be used
set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
set(TensorRT_LIBRARIES ${TensorRT_LIBRARY})

# Create an IMPORTED target
add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
set_target_properties(TensorRT::TensorRT PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}")
set_property(TARGET TensorRT::TensorRT APPEND PROPERTY IMPORTED_LOCATION "${TensorRT_LIBRARY}")