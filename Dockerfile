# Use the official Python image as a base
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Ensure the model_path directory exists for saving trained models
RUN mkdir -p /app/model_path

# Install system dependencies required for MetaDrive and other packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-xcb1 \
    libx11-6 \
    libglu1-mesa \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install \
       torch \
       numpy \
       matplotlib \
       pygame \
       panda3d \
       shapely \
       scipy \
       pygments \
       gltf \
       seaborn \
       opencv-python \
       tqdm \
       gymnasium \
       imageio \
       metadrive

# Set environment variables for saving models
ENV MODEL_PATH=/app/model_path

# Expose any required ports (if needed for the application)
EXPOSE 8080

# Set the command to run the application
CMD ["python", "main.py"]
