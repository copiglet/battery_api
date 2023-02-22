#!/bin/bash

# IMG="pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel"
IMG="huvio:v1"

DATA_IN=${PWD}"/input"
DATA_OUT=${PWD}"/output"


# docker run --name yolo-test --gpus all -it --rm -v ${PWD}:/pmx $IMG bash
# docker run --name huvio_test --gpus all --rm -it \
docker run --name huvio_test --gpus all --rm -d -t \
  -v ${PWD}:/workspace:rw \
  -v ${DATA_IN}:/workspace/inference/images:rw \
  -v ${DATA_OUT}:/workspace/runs/detect:rw \
  --workdir /workspace \
  ${IMG}