# Use an official Python runtime as a parent image
FROM python:3.11-alpine

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some Python packages (e.g., for image processing)
# Add any other build-time dependencies here if needed by your Python packages
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY bot.py .
# If you have other local modules or files (e.g. a "utils" folder), copy them too:
# COPY utils/ ./utils/

# Expose the port the app runs on (not strictly necessary for polling bots, but good practice if you ever switch to webhooks)
# EXPOSE 8080 # Or whatever port your webhook might use

# Command to run the application
# The environment variables (BOT_TOKEN, etc.) will be passed in via docker-compose.yml or `docker run -e`
CMD ["python", "bot.py"]
