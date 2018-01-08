SET(PROJECT_NAME DSO)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
#set(CMAKE_VERBOSE_MAKEFILE ON)


set(BUILD_TYPE Release)
#set(BUILD_TYPE RelWithDebInfo)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# required libraries
find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread) 

# optional libraries
find_package(LibZip QUIET)
find_package(Pangolin 0.2 QUIET)
find_package(OpenCV QUIET)

# flags
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O3 -g -std=c++0x -march=native"
#   "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
)

if (MSVC)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif (MSVC)

# Sources files
set(dso_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystem.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptimize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptPoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemDebugStuff.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemMarginalize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/Residuals.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseTracker.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseInitializer.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/ImmaturePoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/HessianBlocks.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/PixelSelector2.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.cpp
  ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
  ${PROJECT_SOURCE_DIR}/src/util/Undistort.cpp
  ${PROJECT_SOURCE_DIR}/src/util/globalCalib.cpp

  ${PROJECT_SOURCE_DIR}/src/calibration/geometric_calibration.cpp
  ${PROJECT_SOURCE_DIR}/src/calibration/photometric_calibration.cpp
  ${PROJECT_SOURCE_DIR}/src/monovo/system.cpp
)



include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  ${PROJECT_SOURCE_DIR}/thirdparty/sse2neon
  ${EIGEN3_INCLUDE_DIR}
) 


# decide if we have pangolin
if (Pangolin_FOUND)
	message("--- found PANGOLIN, compiling dso_pangolin library.")
	include_directories( ${Pangolin_INCLUDE_DIRS} ) 
	set(dso_pangolin_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.cpp)
	set(HAS_PANGOLIN 1)
else ()
	message("--- could not find PANGOLIN, not compiling dso_pangolin library.")
	message("    this means there will be no 3D display / GUI available for dso_dataset.")
	set(dso_pangolin_SOURCE_FILES )
	set(HAS_PANGOLIN 0)
endif ()

# decide if we have openCV
set(OpenCV_FOUND TRUE)
file(GLOB OPENCV_LIBRARIES_MINE "/usr/local/Cellar/opencv3/3.3.1_1/lib/*dylib")
if (OpenCV_FOUND)
	message("--- found OpenCV, compiling dso_opencv library.")
	# include_directories( ${OpenCV_INCLUDE_DIRS} )
	include_directories( /usr/local/Cellar/opencv3/3.3.1_1/include )
	set(dso_opencv_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp)
	set(HAS_OPENCV 1)
else ()
	message("--- could not find OpenCV, not compiling dso_opencv library.")
	message("    this means there will be no image display, and image read / load functionality.")
	set(dso_opencv_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageDisplay_dummy.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageRW_dummy.cpp)
	set(HAS_OPENCV 0)
endif ()

# decide if we have ziplib.
if (LIBZIP_LIBRARY)
	message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
	add_definitions(-DHAS_ZIPLIB=1)
	include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} ) 
else()
	message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
	set(LIBZIP_LIBRARY "")
endif()


# compile main library.
include_directories( ${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}) 
add_library(dso ${dso_SOURCE_FILES} ${dso_opencv_SOURCE_FILES} ${dso_pangolin_SOURCE_FILES})

#set_property( TARGET dso APPEND_STRING PROPERTY COMPILE_FLAGS -Wall )


if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # OSX
    set(BOOST_THREAD_LIBRARY boost_thread-mt)
else()
    set(BOOST_THREAD_LIBRARY boost_thread)
endif()


# boost
include_directories( /usr/local/Cellar/boost/1.65.1/include )
file(GLOB BOOST_LIBRARIES_MINE "/usr/local/Cellar/boost/1.65.1/lib/*" )

# suite sparse
include_directories( /usr/local/Cellar/suite-sparse/4.5.4/include/ )
file(GLOB SUITESPARSE_LIBRARIES_MINE "/usr/local/Cellar/suite-sparse/4.5.4/lib/*" )

# mynteye
include_directories( /Users/yinr/ComputerVision/SLAM/workspace/MYNT-EYE-SDK/1.x/1.7/mynteye-1.7-mac-x64-opencv-3.2.0/include )
file(GLOB MYNTEYE_LIBRARIES_1
		"/Users/yinr/ComputerVision/SLAM/workspace/MYNT-EYE-SDK/1.x/1.7/mynteye-1.7-mac-x64-opencv-3.2.0/lib/libmynteye_core.dylib")
file(GLOB OPENCV_LIBRARIES_MYNTEYE "/Users/yinr/ComputerVision/SLAM/workspace/opencv/build/lib/*dylib")



# build main executable (only if we have both OpenCV and Pangolin)
if (OpenCV_FOUND AND Pangolin_FOUND)
	message("--- compiling dso_dataset.")
	add_executable(dso_dataset ${PROJECT_SOURCE_DIR}/src/main_dso_pangolin.cpp )
    target_link_libraries(dso_dataset dso ${BOOST_LIBRARIES_MINE} ${SUITESPARSE_LIBRARIES_MINE} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OPENCV_LIBRARIES_MINE})

	add_executable(mynteye_async ${PROJECT_SOURCE_DIR}/src/main_mynteye_async.cpp)
	target_link_libraries(mynteye_async dso ${BOOST_LIBRARIES_MINE} ${SUITESPARSE_LIBRARIES_MINE} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OPENCV_LIBRARIES_MYNTEYE} ${MYNTEYE_LIBRARIES_1})

	add_executable(mynteye ${PROJECT_SOURCE_DIR}/src/main_mynteye.cpp)
	target_link_libraries(mynteye dso ${BOOST_LIBRARIES_MINE} ${SUITESPARSE_LIBRARIES_MINE} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OPENCV_LIBRARIES_MYNTEYE} ${MYNTEYE_LIBRARIES_1})

	add_executable(mynteye_dataset ${PROJECT_SOURCE_DIR}/src/main_mynteye_dataset.cpp)
	target_link_libraries(mynteye_dataset dso ${BOOST_LIBRARIES_MINE} ${SUITESPARSE_LIBRARIES_MINE} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OPENCV_LIBRARIES_MYNTEYE} ${MYNTEYE_LIBRARIES_1})

	add_executable(mynteye_dataset_nogui ${PROJECT_SOURCE_DIR}/src/main_mynteye_dataset_nogui.cpp)
	target_link_libraries(mynteye_dataset_nogui dso ${BOOST_LIBRARIES_MINE} ${SUITESPARSE_LIBRARIES_MINE} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OPENCV_LIBRARIES_MYNTEYE} ${MYNTEYE_LIBRARIES_1})

	add_executable(mynteye_recorder ${PROJECT_SOURCE_DIR}/src/main_mynteye_recorder.cpp)
	target_link_libraries(mynteye_recorder ${BOOST_LIBRARIES_MINE} ${OPENCV_LIBRARIES_MYNTEYE} ${MYNTEYE_LIBRARIES_1})

	add_executable(mynteye_async_recorder ${PROJECT_SOURCE_DIR}/src/main_mynteye_async_recorder.cpp)
	target_link_libraries(mynteye_async_recorder ${BOOST_LIBRARIES_MINE} ${OPENCV_LIBRARIES_MYNTEYE} ${MYNTEYE_LIBRARIES_1})

	add_executable(breakdown_initializer ${PROJECT_SOURCE_DIR}/src/main_breakdown_initializer.cpp)
	target_link_libraries(breakdown_initializer dso ${BOOST_LIBRARIES_MINE} ${SUITESPARSE_LIBRARIES_MINE} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OPENCV_LIBRARIES_MYNTEYE} ${MYNTEYE_LIBRARIES_1})


else()
	message("--- not building dso_dataset, since either don't have openCV or Pangolin.")
endif()

