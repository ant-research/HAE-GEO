FROM ubuntu:22.04 AS engine-build
ARG XUNSEARCH_URL=https://www.xunsearch.com/download/xunsearch-full-latest.tar.bz2
ARG XUNSEARCH_SHA256=2ed66562f5cc6350e084c50edffa970c5306d348ae7c1da7ba9aa140afec7743
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential bzip2 zlib1g-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN curl -fL --retry 3 "$XUNSEARCH_URL" -o source.tar.bz2 \
    && printf '%s  source.tar.bz2\n' "$XUNSEARCH_SHA256" | sha256sum -c - \
    && mkdir source && tar -xjf source.tar.bz2 --strip-components=1 -C source \
    && cd source && CXX="g++ -std=gnu++11" sh setup.sh --prefix=/opt/xunsearch --jobs=2

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 php-cli php-mbstring libstdc++6 libuuid1 zlib1g ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=engine-build /opt/xunsearch /opt/xunsearch
COPY . /app/deep_researchbehind
ENV LD_LIBRARY_PATH=/opt/xunsearch/lib
WORKDIR /app/deep_researchbehind
CMD ["python3", "server.py", "--manifest", "/data/index/manifest.json", "--host", "0.0.0.0"]
