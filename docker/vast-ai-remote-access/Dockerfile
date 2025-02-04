# From base directory: docker build -f ./docker/vast-ai-image/Dockerfile -t vast-ai-janus .
FROM python:3.9

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY . /app
RUN pip install --no-cache-dir -e .
