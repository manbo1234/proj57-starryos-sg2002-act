ARG TPU_MLIR_IMAGE=sophgo/tpuc_dev:latest
FROM ${TPU_MLIR_IMAGE}

WORKDIR /workspace/proj57
ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
