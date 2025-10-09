# Use an official lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files & buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port Render will use
EXPOSE 8000

# Command to run the app with Gunicorn
CMD ["waitress-serve ", "--host"," 127.0.0.1", "--call", "app:create_app"]
