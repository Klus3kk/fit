FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev,examples]"

# Copy source code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash fit
RUN chown -R fit:fit /app
USER fit

# Default command
CMD ["python", "-m", "pytest", "tests/"]