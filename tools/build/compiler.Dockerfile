# Public Jetson CUDA 12.6/PyTorch 2.9.1 image; no private parent or model/cache copy.
FROM ghcr.io/nvidia-ai-iot/vllm@sha256:2a90817f4d760094a25c546126a48aa8e8ae4fae1ae41bf72445fbc2c8bc2a7f
COPY tools/build/compiler-requirements.txt /tmp/orin-requirements.txt
RUN PIP_CONSTRAINT=/dev/null python3 -m pip install --no-cache-dir --index-url https://pypi.org/simple --no-deps uv==0.12.23 \
 && uv pip uninstall --python /opt/venv/bin/python vllm pygobject nvidia-cutlass \
 && UV_CONSTRAINT=/dev/null UV_BUILD_CONSTRAINT=/dev/null uv pip install --python /opt/venv/bin/python --no-cache --index-url https://pypi.org/simple --no-deps -r /tmp/orin-requirements.txt \
 && uv pip check --python /opt/venv/bin/python \
 && python3 -c "import importlib.metadata as m; assert m.version('torch') == '2.9.1'; assert m.version('tilelang') == '0.1.15'" \
 && rm /tmp/orin-requirements.txt
ENV PYTHONPATH=/opt/venv/lib/python3.10/site-packages
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
WORKDIR /work
ENTRYPOINT ["/usr/bin/python3"]
