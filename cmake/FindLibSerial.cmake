# Locate LibSerial
#
# This module defines
#  LibSerial_FOUND, if false, do not try to link to LibSerial
#  LibSerial_LIBRARY, where to find LibSerial
#  LibSerial_INCLUDE_DIR, where to find SerialPort.h

# find the LibSerial include directory
find_path(
    LibSerial_INCLUDE_DIR 
    NAMES SerialPort.h
    PATHS /usr/include/libserial /usr/local/include/libserial
)

# find the LibSerial library
find_library(
    LibSerial_LIBRARY
    NAMES libserial.so
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)

# handle the QUIETLY and REQUIRED arguments and set LibSerial_FOUND to TRUE if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LibSerial DEFAULT_MSG LibSerial_INCLUDE_DIR LibSerial_LIBRARY)
mark_as_advanced(LibSerial_INCLUDE_DIR LibSerial_LIBRARY)

if (LibSerial_FOUND)
    message(STATUS "LibSerial found (include: ${LibSerial_INCLUDE_DIR}, lib: ${LibSerial_LIBRARY})")
endif()