FROM python:3.11-slim

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENVIRONMENT=production

# Set work directory
WORKDIR /app

# Install system dependencies for production
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        python3-sklearn \
        curl \
        build-essential \
        libssl-dev \
        libffi-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r bci && useradd -r -g bci -d /app -s /bin/bash bci

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY requirements-dev.txt ./

# Install production dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        numpy scipy scikit-learn matplotlib \
        anthropic fastapi uvicorn pydantic \
        h5py websockets asyncio-mqtt python-dotenv \
        tqdm cryptography requests redis psutil

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Install the package
RUN pip install -e .

# Create directories for production data
RUN mkdir -p /app/audit_logs /app/compliance_data /app/logs /app/monitoring

# Change ownership to non-root user
RUN chown -R bci:bci /app

# Switch to non-root user
USER bci

# Expose port for API server
EXPOSE 8000

# Health check for production
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Production command - start API server
CMD ["python", "-m", "bci_agent_bridge", "--server", "--log-level", "INFO"]