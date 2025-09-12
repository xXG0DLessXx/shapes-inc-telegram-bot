# builder stage
FROM python:3.11-alpine AS builder
WORKDIR /app

# System dependencies for building Python packages
RUN apk add --no-cache \
    build-base \
    cmake \
    make \
    ninja \
    linux-headers \
    ffmpeg-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
FROM python:3.11-alpine

# System dependencies for running the application
RUN apk add --no-cache \
    ffmpeg-libs \
    libjpeg-turbo \
    libpng \
    tiff

WORKDIR /app

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY bot.py .

CMD ["python", "bot.py"]