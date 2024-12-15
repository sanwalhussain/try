# Use the official Python image as a base
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for MetaDrive and other Python packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-xcb1 \
    libx11-6 \
    libglu1-mesa \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the MetaDrive repository
RUN git clone https://github.com/metadriverse/metadrive.git /app/metadrive

# Copy all local files into the MetaDrive directory
COPY . /app/metadrive

# Install Python dependencies
RUN pip install --upgrade pip && pip install \
    shapely \
    scipy \
    pygments \
    gltf \
    seaborn \
    opencv-python \
    progressbar \
    gymnasium

# Set environment variables
ENV MODEL_PATH=/app/metadrive/model_path

# Ensure the model_path directory exists
RUN mkdir -p $MODEL_PATH

# Expose any required ports (if needed for the application)
EXPOSE 8080

# Set the command to run the application
CMD ["python", "/app/metadrive/main.py"]
