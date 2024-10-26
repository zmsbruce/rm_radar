set(Hikvision_INCLUDE_DIR /opt/MVS/include)
set(lib_names FormatConversion
    MvCameraControl
    MvCameraControlWrapper
    MVRender
    MvUsb3vTL
)

set(path_names /opt/MVS/lib/64 /opt/MVS/lib/32 /opt/MVS/lib)

foreach(lib_name IN LISTS lib_names)
    unset(Hikvision_LIB)  
  
    find_library(Hikvision_LIB NAMES ${lib_name} PATHS ${path_names})

    if(Hikvision_LIB)
        list(APPEND Hikvision_LIBS ${Hikvision_LIB})
        if(NOT hikvision_find_quietly)
            message(STATUS "Hikvision found: ${Hikvision_LIB}")
        endif()
    elseif(hikvision_find_required)
        message(FATAL_ERROR "Could not find Hikvision library: ${lib_name}")
    endif()
endforeach()

unset(Hikvision_LIB CACHE)