FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        libffi-dev \
        libblas-dev \
        liblapack-dev \
        gfortran \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r bci && useradd -r -g bci bci

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -e .[dev] \
    && pip install -r requirements-dev.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Change ownership to non-root user
RUN chown -R bci:bci /app

# Switch to non-root user
USER bci

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import bci_agent_bridge; print('BCI Bridge is healthy')" || exit 1

# Default command
CMD ["python", "-m", "bci_agent_bridge"]