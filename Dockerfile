FROM debian:bookworm
LABEL authors="zaixia108"

RUN cd /tmp && \
    wget https://go.dev/dl/go1.25.5.linux-amd64.tar.gz && \
    rm -rf /usr/local/go && tar -C /usr/local -xzf go1.25.5.linux-amd64.tar.gz && \
    export PATH=$PATH:/usr/local/go/bin

ENV http_proxy=http://192.168.31.10:7890
ENV https_proxy=http://192.168.31.10:7890
RUN apt update && apt install -y curl zip unzip tar autoconf autoconf-archive automake libtool

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
      nasm libatlas-base-dev libblas-dev liblapack-dev libvtk7-dev python3.11-venv bison libxtst-dev

RUN apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /etc/apt/sources.list.d/*

RUN cd /opt && \
    git clone https://github.com/microsoft/vcpkg.git

WORKDIR /opt/vcpkg

RUN ./bootstrap-vcpkg.sh && \
    ./vcpkg install opencv:x64-linux && \
    ./vcpkg install ncnn[vulkan]:x64-linux

RUN mkdir -p /tmp/build && mkdir -p /opt/server && \
    cd /tmp/build && \
    git clone https://github.com/zaixia108/OnnxDetServer-GO.git && \
    git checkout DevMultiPlatform

WORKDIR /tmp/build/OnnxDetServer-GO
RUN cd src/ncnn && mkdir build && cd build && \
    cmake -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && make install && cd ../../

RUN go build -o /opt/server/DetServer ./cmd/DetServer/OnnxDetServer-Go.go
COPY config.yaml /opt/server/config.yaml

WORKDIR /opt/server
EXPOSE 50051-50054/tcp
CMD ["./DetServer"]