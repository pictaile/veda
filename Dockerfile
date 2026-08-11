FROM gcc:latest

RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    gdb \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN cmake -G Ninja -S . -B build && cmake --build build

CMD ["./build/veda"]