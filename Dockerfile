ARG BASE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE}
# uv from its official image; baked deps so loop/prep never reinstall.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv pip install --system --break-system-packages --no-cache "kernels==0.14.1" transformers pillow pyyaml \
 && python3 -c "import kernels,transformers,PIL,yaml;print('baked kernels',kernels.__version__,'transformers',transformers.__version__)"
