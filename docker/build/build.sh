docker build --build-arg http_proxy=192.168.1.11:8118 --build-arg https_proxy=192.168.1.11:8118 \
  -t oneflow-build -f docker/build/Dockerfile .
