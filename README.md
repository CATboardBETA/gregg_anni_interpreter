# Gregg Anniversary Edition Interpreter

# Special setup

This requires a special installation of `libtorch` to work.  The `libtorch` library
is not included in this repository.  You can download it from the
[PyTorch website](https://pytorch.org/get-started/locally/), e.g, for macOS:
```shell
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.13.1.zip
```

You will also always need to run these commands:
```shell
cd libtorch/include/torch
mkdir include
mkdir lib
mv *.h include
mv csrc include/csrc
cd ../../lib
mv *.so ../../include/torch/lib
```