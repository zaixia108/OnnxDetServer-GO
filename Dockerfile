FROM debian:bookworm
LABEL authors="zaixia108"

ENV http_proxy=http://192.168.31.10:7890
ENV https_proxy=http://192.168.31.10:7890

RUN apt update && apt install -y curl zip unzip tar autoconf autoconf-archive automake libtool wget

RUN cd /tmp && \
    wget https://go.dev/dl/go1.25.5.linux-amd64.tar.gz && \
    rm -rf /usr/local/go && tar -C /usr/local -xzf go1.25.5.linux-amd64.tar.gz && \
    export PATH=$PATH:/usr/local/go/bin


RUN apt-get update && apt-get -y install \
      autoconf automake libass-dev libgnutls28-dev \
      libmp3lame-dev libtool libvorbis-dev \
      meson ninja-build pkg-config \
      texinfo wget yasm \
      zlib1g-dev libx264-dev libvpx-dev \
      libopus-dev libdav1d-dev \
      git build-essential cmake pkg-config unzip \
      curl ca-certificates libcurl4-openssl-dev libssl-dev \
      libharfbuzz-dev libfreetype6-dev \
      nasm libatlas-base-dev libblas-dev liblapack-dev libvtk9-dev python3.11-venv bison libxtst-dev

RUN cd /opt && \
    git clone https://github.com/microsoft/vcpkg.git

WORKDIR /opt/vcpkg

RUN ./bootstrap-vcpkg.sh && \
    ./vcpkg install ncnn[vulkan]:x64-linux

ARG OPENCV_VERSION="4.13.0"
ENV OPENCV_VERSION $OPENCV_VERSION

ARG OPENCV_FILE="https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip"
ENV OPENCV_FILE $OPENCV_FILE

ARG OPENCV_CONTRIB_FILE="https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip"
ENV OPENCV_CONTRIB_FILE $OPENCV_CONTRIB_FILE

RUN curl -Lo opencv.zip ${OPENCV_FILE} && \
      unzip -q opencv.zip && \
      curl -Lo opencv_contrib.zip ${OPENCV_CONTRIB_FILE} && \
      unzip -q opencv_contrib.zip && \
      rm opencv.zip opencv_contrib.zip

RUN cd opencv-${OPENCV_VERSION} && \
      mkdir build && cd build && \
      cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D WITH_IPP=ON \
      -D BUILD_WITH_DYNAMIC_IPP=OFF \
      -D BUILD_IPP_IW=ON \
      -D WITH_OPENGL=ON \
      -D BUILD_OPENGL=ON \
      -D WITH_QT=OFF \
      -D WITH_FREETYPE=ON \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D BUILD_SHARED_LIBS=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D WITH_JASPER=OFF \
      -D WITH_TBB=ON \
      -D BUILD_TBB=ON \
      -D BUILD_JPEG=ON \
      -D WITH_SIMD=ON \
      -D ENABLE_LIBJPEG_TURBO_SIMD=OFF \
      -D WITH_QUIRC=ON \
      -D WITH_GTK=OFF \
      -D WITH_FFMPEG=ON \
      -D WITH_1394=ON \
      -D BUILD_1394=ON \
      -D WITH_WEBP=ON \
      -D BUILD_WEBP=ON \
      -D WITH_OPENJPEG=ON \
      -D BUILD_OPENJPEG=ON \
      -D WITH_TIFF=ON \
      -D BUILD_TIFF=ON \
      -D BUILD_DOCS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_opencv_java=NO \
      -D BUILD_opencv_python=NO \
      -D BUILD_opencv_python2=NO \
      -D BUILD_opencv_python3=NO \
      -D WITH_EIGEN=OFF \
      -D WITH_VTK=OFF \
      -D BUILD_opencv_wechat_qrcode=OFF \
      -D CMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
      -D WITH_ICONV=OFF \
      -D OPENCV_GENERATE_PKGCONFIG=ON .. && \
      make -j $(( $(nproc) - 1 )) && \
      make preinstall && make install && ldconfig && \
      cd / && rm -rf opencv*

RUN mkdir -p /tmp/build && mkdir -p /opt/server && \
    cd /tmp/build && \
    git clone https://github.com/zaixia108/OnnxDetServer-GO.git && \
    cd OnnxDetServer-GO && \
    git checkout DevMultiPlatform

WORKDIR /tmp/build/OnnxDetServer-GO
RUN cd src/ncnn && mkdir build && cd build && \
    cmake -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && make install && cd ../../
WORKDIR /tmp/build/OnnxDetServer-GO
ENV CGO_ENABLED=1
ENV PATH="/usr/local/go/bin:${PATH}"

RUN apt-get install -y build-essential

RUN go mod tidy
COPY src/NcnnDet.h /usr/local/include/NcnnDet.h
RUN go build -o /opt/server/DetServer OnnxDetServer-Go.go
COPY config.yaml /opt/server/config.yaml
RUN ldconfig

RUN apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /etc/apt/sources.list.d/*

WORKDIR /opt/server
EXPOSE 50051-50054/tcp
CMD ["./DetServer"]