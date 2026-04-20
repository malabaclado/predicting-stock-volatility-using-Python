# Use the official Python slim image (lighter and better for cross-platform)
FROM python:3.10-slim AS builder

# Set the working directory
WORKDIR /app

# Install build-essential (needed for some ML libraries like 'arch')
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies from requirements.txt
# This is much more reliable for Docker than a Windows-specific Conda file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Building the final image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the installed packages from the builder's system directory
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]