@echo off

if not exist "C:\opencv" mkdir "C:\opencv"
if not exist "C:\opencv\build" mkdir "C:\opencv\build"

echo Downloading OpenCV sources
echo.
echo For monitoring the download progress please check the C:\opencv directory.
echo.


echo This script will download OpenCV 4.12.0 and opencv_contrib 4.12.0.
echo It will take a while, please be patient.
echo.
echo Downloading: opencv-4.12.0.zip [91MB]
powershell -command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri https://github.com/opencv/opencv/archive/refs/tags/4.12.0.zip -OutFile c:\opencv\opencv-4.12.0.zip"
echo Extracting...
powershell -command "$ProgressPreference = 'SilentlyContinue'; Expand-Archive -Path c:\opencv\opencv-4.12.0.zip -DestinationPath c:\opencv"
del c:\opencv\opencv-4.12.0.zip /q
echo.
echo Downloading: opencv_contrib-4.12.0.zip [58MB]
powershell -command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri https://github.com/opencv/opencv_contrib/archive/refs/tags/4.12.0.zip -OutFile c:\opencv\opencv_contrib-4.12.0.zip"
echo Extracting...
powershell -command "$ProgressPreference = 'SilentlyContinue'; Expand-Archive -Path c:\opencv\opencv_contrib-4.12.0.zip -DestinationPath c:\opencv"
del c:\opencv\opencv_contrib-4.12.0.zip /q
echo.
echo OpenCV sources downloaded and extracted successfully.


cmake C:\opencv\opencv-4.12.0 -G "MinGW Makefiles" -BC:\opencv\build -DENABLE_CXX11=ON -DOPENCV_EXTRA_MODULES_PATH=C:\opencv\opencv_contrib-4.12.0\modules -DBUILD_SHARED_LIBS=%enable_shared% -DWITH_IPP=OFF -DWITH_MSMF=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_DOCS=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DBUILD_opencv_saliency=OFF -DBUILD_opencv_wechat_qrcode=ON -DCPU_DISPATCH= -DOPENCV_GENERATE_PKGCONFIG=ON -DWITH_OPENCL_D3D11_NV=OFF -DOPENCV_ALLOCATOR_STATS_COUNTER_TYPE=int64_t -Wno-dev

mingw32-make -j %NUMBER_OF_PROCESSORS%

mingw32-make install