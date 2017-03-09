project (fast_bilateral_space_stereo)

cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.0)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(Eigen)
find_package(OpenCV 3)
find_package(Ceres)

# Source files
SET(FAST_BIL_SOURCES
    src/main.cpp
    src/bilateral_grid_simplified.cpp
    src/bilateral_grid_simplified.h
    src/fast_bilateral_solver.cpp
    src/fast_bilateral_solver.h
    src/stereo_matcher_birchfield_tomasi.cpp
    src/stereo_matcher_birchfield_tomasi.h
)

include_directories(${EIGEN_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${GLOG_INCLUDE_DIR} ${CERES_INCLUDE_DIRS})

# Create the executable
add_executable(fast_bilateral_space_stereo ${FAST_BIL_SOURCES})
if(MSVC)
    set(LINK_LIBRARY ${CERES_LIBRARIES} ${OpenCV_LIBS} ${GLOG_LIBRARIES} shlwapi.lib)
else()
    set(LINK_LIBRARY ${CERES_LIBRARIES} ${OpenCV_LIBS} ${GLOG_LIBRARIES})
endif()
target_link_libraries(fast_bilateral_space_stereo ${LINK_LIBRARY})