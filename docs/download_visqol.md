# How to install ViSQOL

## 1. Install bazel
```
// ↓ select url that matches your machine from https://github.com/bazelbuild/bazelisk/releases.
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
mv bazelisk-linux-amd64 ~/.local/bin/bazel
```

## 2. Clone ViSQOL and build
```
git clone https://github.com/google/visqol.git
cd visqol
bazel build :visqol
mv ./bazel-bin/visqol ~/.local/bin/
```