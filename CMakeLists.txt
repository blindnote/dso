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
#find_package(OpenCV REQUIRED PATHS "/usr/local/Cellar/opencv/HEAD-f109c01/")
set(BUILD_TYPE Release)
#set(BUILD_TYPE RelWithDebInfo)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# required libraries
#find_package(SuiteParse REQUIRED)
#find_package(Eigen3 REQUIRED)
#find_package(Boost COMPONENTS system thread)

# optional libraries
#find_package(LibZip QUIET)
#find_package(Pangolin 0.2 QUIET)
#find_package(OpenCV QUIET)

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

  ${PROJECT_SOURCE_DIR}/src/Initialization/iCommon.cpp
  ${PROJECT_SOURCE_DIR}/src/Initialization/iEpipolarSolver.cpp
  ${PROJECT_SOURCE_DIR}/src/Initialization/iFeatureExtractor.cpp
  ${PROJECT_SOURCE_DIR}/src/Initialization/iRichFeatureMatcher.cpp
  ${PROJECT_SOURCE_DIR}/src/Initialization/iFrame.cpp
  ${PROJECT_SOURCE_DIR}/src/Initialization/Initializer.cpp
)


#set(CMAKE_PREFIX_PATH /usr/local/Cellar/opencv3/HEAD-f109c01/include)

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
#file(GLOB OPENCV_LIBRARIES_MINE "/usr/local/Cellar/opencv/HEAD-f109c01/lib/*dylib")
file(GLOB OPENCV_LIBRARIES_MINE "/usr/local/Cellar/opencv/3.3.1_1/lib/*dylib")
if (OpenCV_FOUND)
	message("--- found OpenCV, compiling dso_opencv library.")
#	message("--- found OpenCV, compiling dso_opencv library.")
#	include_directories( ${OpenCV_INCLUDE_DIRS} )
#    set(OpenCV_INCLUDE_DIRS /usr/local/Cellar/opencv3/HEAD-f109c01/include)
#	include_directories( /usr/local/Cellar/opencv3/HEAD-f109c01/include )
	include_directories( /usr/local/Cellar/opencv3/3.3.1_1/include )
	set(dso_opencv_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp src/Initialization/Initializer.cpp src/Initialization/Initializer.h src/Initialization/iFrame.cpp src/Initialization/iFrame.h src/Initialization/iFeatureExtractor.cpp src/Initialization/iFeatureExtractor.h src/Initialization/iCommon.cpp src/Initialization/iCommon.h src/Initialization/iEpipolarSolver.cpp src/Initialization/iEpipolarSolver.h src/Initialization/iFeatureMatcher.h src/Initialization/iRichFeatureMatcher.cpp src/Initialization/iRichFeatureMatcher.h)
	set(HAS_OPENCV 1)


else ()
	message("--- could not find OpenCV, not compiling dso_opencv library.")
	message("    this means there will be no image display, and image read / load functionality.")
#	set(dso_opencv_SOURCE_FILES 
#	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageDisplay_dummy.cpp
#	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageRW_dummy.cpp)
#	set(HAS_OPENCV 0)
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

#file(GLOB Pangolin_LIBRARIES_MINE "/Users/yinr/ComputerVision/3rdParty/Pangolin/build/src/*dylib")

#include_directories( /usr/local/Cellar/boost/1.63.0/include )
#file(GLOB BOOST_LIBRARIES_MINE "/usr/local/Cellar/boost/1.63.0/lib/*" )
include_directories( /usr/local/Cellar/boost/1.65.1/include )
file(GLOB BOOST_LIBRARIES_MINE "/usr/local/Cellar/boost/1.65.1/lib/*" )

include_directories( /usr/local/Cellar/suite-sparse/4.5.4/include/ )
file(GLOB SUITESPARSE_LIBRARIES_MINE "/usr/local/Cellar/suite-sparse/4.5.4/lib/*" )

# realsense
#include_directories( /usr/local/include/ )
#include_directories( /Users/yinr/ComputerVision/librealsense/examples )
#file(GLOB REALSENSE_LIBRARIES "/usr/local/lib/librealsense.dylib")

# glew
include_directories( /usr/local/Cellar/glew/2.0.0/include/ )
file(GLOB GLEW_LIBRARIES "/usr/local/Cellar/glew/2.0.0/lib/*")

# glfw
include_directories( /usr/local/Cellar/glfw/3.2.1/include/ )
file(GLOB GLFW_LIBRARIES "/usr/local/Cellar/glfw/3.2.1/lib/*")

# mynteye
include_directories( /Users/yinr/ComputerVision/SLAM/workspace/MYNT-EYE-SDK/1.x/1.6/mynteye-1.6-mac-x64-opencv-3.2.0/include )
file(GLOB MYNTEYE_LIBRARIES_1
		"/Users/yinr/ComputerVision/SLAM/workspace/MYNT-EYE-SDK/1.x/1.6/mynteye-1.6-mac-x64-opencv-3.2.0/lib/libmynteye_core.dylib")
file(GLOB MYNTEYE_LIBRARIES_2
		"/Users/yinr/ComputerVision/SLAM/workspace/MYNT-EYE-SDK/1.x/1.6/mynteye-1.6-mac-x64-opencv-3.2.0/lib/libmynteye_features.dylib")
file(GLOB MYNTEYE_LIBRARIES_3
		"/Users/yinr/ComputerVision/SLAM/workspace/MYNT-EYE-SDK/1.x/1.6/mynteye-1.6-mac-x64-opencv-3.2.0/samples/lib/*dylib")
file(GLOB OPENCV_LIBRARIES_MYNTEYE "/Users/yinr/ComputerVision/SLAM/workspace/opencv/build/lib/*dylib")


# build main executable (only if we have both OpenCV and Pangolin)
if (OpenCV_FOUND AND Pangolin_FOUND)
	message("--- compiling dso_dataset.")
	add_executable(dso_dataset ${PROJECT_SOURCE_DIR}/src/main_dso_pangolin.cpp)
	#add_executable(dso_r200 ${PROJECT_SOURCE_DIR}/src/main_realtime_r200.cpp)
	#add_executable(dso_iphone6s ${PROJECT_SOURCE_DIR}/src/main_iphone6s_images.cpp)
	add_executable(sse_test ${PROJECT_SOURCE_DIR}/src/main_sse_test.cpp)

	add_executable(mynteye_test ${PROJECT_SOURCE_DIR}/src/main_mynteye_test.cpp)
	add_executable(mynteye ${PROJECT_SOURCE_DIR}/src/main_mynteye.cpp)
	add_executable(mynteye_async ${PROJECT_SOURCE_DIR}/src/main_mynteye_async.cpp)
	add_executable(mynteye_dataset ${PROJECT_SOURCE_DIR}/src/main_mynteye_dataset.cpp)

    #target_link_libraries(dso_dataset dso boost_system boost_thread cxsparse ${LIBZIP_LIBRARY} ${OPENCV_LIBRARIES_MINE} ${Pangolin_LIBRARIES} ${OpenCV_LIBS})
    target_link_libraries(dso_dataset dso ${BOOST_LIBRARIES_MINE} ${SUITESPARSE_LIBRARIES_MINE} ${LIBZIP_LIBRARY} ${OPENCV_LIBRARIES_MINE} ${Pangolin_LIBRARIES} ${REALSENSE_LIBRARIES})
	#target_link_libraries(dso_r200 dso ${BOOST_LIBRARIES_MINE} ${SUITESPARSE_LIBRARIES_MINE} ${LIBZIP_LIBRARY} ${OPENCV_LIBRARIES_MINE} ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${REALSENSE_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW_LIBRARIES})
	#target_link_libraries(dso_iphone6s dso ${BOOST_LIBRARIES_MINE} ${SUITESPARSE_LIBRARIES_MINE} ${LIBZIP_LIBRARY} ${OPENCV_LIBRARIES_MINE} ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${REALSENSE_LIBRARIES})
	target_link_libraries(sse_test dso ${BOOST_LIBRARIES_MINE} ${SUITESPARSE_LIBRARIES_MINE} ${LIBZIP_LIBRARY} ${OPENCV_LIBRARIES_MINE} ${Pangolin_LIBRARIES} ${REALSENSE_LIBRARIES})

	target_link_libraries(mynteye_test ${MYNTEYE_LIBRARIES_1} ${OPENCV_LIBRARIES_MYNTEYE} dso ${BOOST_LIBRARIES_MINE} )
	target_link_libraries(mynteye ${MYNTEYE_LIBRARIES_1} dso ${BOOST_LIBRARIES_MINE} ${SUITESPARSE_LIBRARIES_MINE} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OPENCV_LIBRARIES_MYNTEYE})
	target_link_libraries(mynteye_async ${MYNTEYE_LIBRARIES_1} dso ${BOOST_LIBRARIES_MINE} ${SUITESPARSE_LIBRARIES_MINE} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OPENCV_LIBRARIES_MYNTEYE})
	target_link_libraries(mynteye_dataset ${MYNTEYE_LIBRARIES_1} dso ${BOOST_LIBRARIES_MINE} ${SUITESPARSE_LIBRARIES_MINE} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OPENCV_LIBRARIES_MYNTEYE})

else()
	message("--- not building dso_dataset, since either don't have openCV or Pangolin.")
endif()


