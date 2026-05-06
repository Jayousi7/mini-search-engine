# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for NLTK and some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (required for preprocessing)
RUN python -m nltk.downloader punkt punkt_tab stopwords

# Copy the rest of the application code
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "server.py"]
