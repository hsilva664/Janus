# From base directory: docker build -f ./docker/vast-ai-image/Dockerfile -t vast-ai-janus .
FROM python:3.9

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY . /app
RUN pip install --no-cache-dir -e .

# Installing directory for transformers debug

# Clone the transformers repository and checkout the llavas branch
RUN mkdir /custom && \
    git clone https://github.com/hsilva664/transformers.git /custom/transformers && \
    cd /custom/transformers && \
    git checkout yaswanth19-add-janus-model && \
    python -m venv my-venv && \
    source my-venv/bin/activate && \
    pip install --upgrade pip && \
    sudo apt-get update && \
    sudo apt-get install -y cmake g++ make && \
    sudo apt-get install -y libboost-all-dev && \
    pip install -e ".[dev-torch]"
