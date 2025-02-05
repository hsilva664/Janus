# From base directory: docker build -f ./docker/vast-ai-remote-access/Dockerfile -t vast-ai-remote-access .
# Run with: docker run -it -p 9999:5901 --gpus all --runtime nvidia vast-ai-remote-access bash
# Use vncsession root :1 to start the vnc server
FROM python:3.9

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Copy and install dependencies
RUN apt update && \
    apt install git -y && \
    git clone https://github.com/hsilva664/Janus
COPY ./data /app/Janus/data
COPY ./cache /app/Janus/cache
COPY ./docker/vast-ai-remote-access/install /app/install
RUN cd Janus && pip install --no-cache-dir -e .

RUN apt update &&\
    apt install xfce4 xfce4-goodies nano chromium systemd dbus-x11 -y

# Install tigervnc from /app/install/tigervnc-1.14.1.x86_64.tar.gz
RUN tar -xvzf /app/install/tigervnc-1.14.1.x86_64.tar.gz -C / --strip-components=1 && \
    mkdir -p /root/.config/tigervnc &&\
    cp /app/install/vnc_config/vncserver.users /etc/tigervnc/vncserver.users &&\
    cp /app/install/vnc_config/passwd /root/.config/tigervnc/passwd &&\
    cp /app/install/vnc_config/config /root/.config/tigervnc/config

# Start it with: vncsession root :1

# Install pycharm
RUN tar xzf /app/install/pycharm-community-2024.3.2.tar.gz -C /opt/ && \
    chmod +x /opt/pycharm-community-2024.3.2/bin/pycharm.sh && \
    ln -s /opt/pycharm-community-2024.3.2/bin/pycharm.sh /usr/local/bin/pycharm
