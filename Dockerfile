FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and source code
COPY pyproject.toml .
COPY augagent /app/augagent

# Install dependencies (including optional api/web/memory)
RUN pip install --no-cache-dir -e .[api,web,memory]

# Expose API port
EXPOSE 8000

# Start API server
CMD ["augagent", "serve", "--host", "0.0.0.0", "--port", "8000"]
