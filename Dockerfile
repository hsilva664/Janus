FROM python:3.9

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# Set working directory
WORKDIR /app

# Install SSH server
RUN apt-get update && apt-get install -y openssh-server
# Create SSH directory
RUN mkdir /var/run/sshd
# Create custom SSH config with proper newlines
RUN echo "PermitEmptyPasswords yes" > /etc/ssh/sshd_config.d/custom.conf && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config.d/custom.conf && \
    echo "UsePAM no" >> /etc/ssh/sshd_config.d/custom.conf && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config.d/custom.conf
# Remove root password
RUN passwd -d root

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep container running
CMD ["/usr/sbin/sshd", "-D"]
