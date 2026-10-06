# open-terminal (Open WebUI code/terminal sandbox) plus extra Python packages.
# The sandbox's home is tmpfs and the reaper restarts it hourly, so anything
# pip-installed at runtime is lost; packages must be baked in here.
FROM ghcr.io/open-webui/open-terminal:latest
USER root
RUN pip install --no-cache-dir cryptography
USER user
