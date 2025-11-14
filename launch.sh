#!/bin/bash
docker run --rm -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility grpo
