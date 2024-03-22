# Use an official Python image
FROM python:3.11-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir --default-timeout=120 -r requirements.txt

# Copy the TTS model directory from your local system into the container
# Adjust the destination path according to your application's expectations

# Copy the rest of your application
COPY . .

EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--timeout-keep-alive", "999999"]
