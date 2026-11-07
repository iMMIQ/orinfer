ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# The runner resolves the selected base to an immutable image ID and pins the wheel.
COPY tilelang-0.1.4+ubuntu.24.3.cann910-cp312-cp312-linux_aarch64.whl /tmp/
RUN python3 -c 'import hashlib,pathlib,subprocess; p=pathlib.Path("/tmp/tilelang-0.1.4+ubuntu.24.3.cann910-cp312-cp312-linux_aarch64.whl"); assert hashlib.sha256(p.read_bytes()).hexdigest()=="6545d48550959eef4208c934d09c5a529ce720ff8a8bcd3e0f9e5a7687138922"; subprocess.check_call(["python3","-m","pip","install","--no-deps","--no-cache-dir","--disable-pip-version-check",str(p)]); p.unlink()'

ENV TL_PLATFORM=A2
WORKDIR /workspace/orinfer
ENTRYPOINT ["/bin/bash"]
