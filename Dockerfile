# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Expose the port the app runs on
EXPOSE 8253

# Run the application
CMD ["python", "diffusers_server.py", "--host", "0.0.0.0", "--port", "8253", "--model", "v2ray/stable-diffusion-3-medium-diffusers", "--output_dir", "/app/outputs", "--models_dir", "/app/models", "--verbose"]
