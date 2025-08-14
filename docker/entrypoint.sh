#!/bin/bash
set -euo pipefail

# BCI-Agent-Bridge Production Entrypoint
# Medical-grade startup script with comprehensive validation and safety checks

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Pre-startup validation
validate_environment() {
    log "🔍 Validating environment configuration..."
    
    # Required environment variables
    required_vars=(
        "BCI_MODE"
        "PRIVACY_MODE"
        "DATABASE_URL"
        "CLAUDE_API_KEY"
        "JWT_SECRET"
        "ENCRYPTION_KEY"
    )
    
    missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        error "Missing required environment variables: ${missing_vars[*]}"
        exit 1
    fi
    
    # Validate BCI mode
    if [[ "$BCI_MODE" != "production" && "$BCI_MODE" != "clinical" && "$BCI_MODE" != "staging" ]]; then
        error "Invalid BCI_MODE: $BCI_MODE. Must be 'production', 'clinical', or 'staging'"
        exit 1
    fi
    
    # Validate privacy mode for medical deployments
    if [[ "$PRIVACY_MODE" == "medical" ]]; then
        if [[ -z "${HIPAA_ENABLED:-}" || "$HIPAA_ENABLED" != "true" ]]; then
            warn "HIPAA_ENABLED not set to 'true' in medical privacy mode"
        fi
    fi
    
    log "✅ Environment validation passed"
}

# Database connectivity check
check_database() {
    log "🗄️ Checking database connectivity..."
    
    # Extract database host from URL for connection testing
    if [[ "$DATABASE_URL" =~ postgresql://[^@]+@([^:/]+) ]]; then
        db_host="${BASH_REMATCH[1]}"
        
        # Wait for database to be ready
        max_attempts=30
        attempt=0
        
        while [[ $attempt -lt $max_attempts ]]; do
            if python3 -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.close()
    print('Database connection successful')
    exit(0)
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
" 2>/dev/null; then
                log "✅ Database connection established"
                return 0
            fi
            
            attempt=$((attempt + 1))
            warn "Database not ready, attempt $attempt/$max_attempts. Retrying in 5 seconds..."
            sleep 5
        done
        
        error "Failed to connect to database after $max_attempts attempts"
        exit 1
    else
        error "Invalid DATABASE_URL format"
        exit 1
    fi
}

# Security and compliance validation
validate_security() {
    log "🔒 Validating security configuration..."
    
    # Check file permissions
    if [[ -d "/app/audit" ]]; then
        audit_perms=$(stat -c "%a" /app/audit)
        if [[ "$audit_perms" != "750" && "$audit_perms" != "755" ]]; then
            warn "Audit directory permissions may be too permissive: $audit_perms"
        fi
    fi
    
    # Validate encryption keys
    if [[ ${#ENCRYPTION_KEY} -lt 32 ]]; then
        error "ENCRYPTION_KEY must be at least 32 characters long"
        exit 1
    fi
    
    if [[ ${#JWT_SECRET} -lt 32 ]]; then
        error "JWT_SECRET must be at least 32 characters long"
        exit 1
    fi
    
    # Check for secure communication requirements
    if [[ "${ENABLE_TLS:-true}" == "true" ]]; then
        if [[ ! -f "/app/config/ssl/server.crt" || ! -f "/app/config/ssl/server.key" ]]; then
            warn "TLS enabled but SSL certificates not found"
        fi
    fi
    
    log "✅ Security validation passed"
}

# Medical device compliance checks
validate_medical_compliance() {
    if [[ "$PRIVACY_MODE" == "medical" || "$COMPLIANCE_MODE" == "clinical" ]]; then
        log "🏥 Validating medical device compliance..."
        
        # Check for required compliance features
        compliance_features=(
            "AUDIT_LOGGING"
            "DATA_RETENTION_DAYS"
            "HIPAA_ENABLED"
        )
        
        for feature in "${compliance_features[@]}"; do
            if [[ -z "${!feature:-}" ]]; then
                warn "Medical compliance feature not configured: $feature"
            fi
        done
        
        # Validate audit logging directory
        if [[ ! -d "/app/audit" ]]; then
            error "Audit logging directory not found: /app/audit"
            exit 1
        fi
        
        # Check data retention policy
        if [[ -n "${DATA_RETENTION_DAYS:-}" ]]; then
            if [[ "$DATA_RETENTION_DAYS" -lt 2555 ]]; then  # 7 years minimum for medical
                warn "Data retention period may be insufficient for medical compliance: $DATA_RETENTION_DAYS days"
            fi
        fi
        
        log "✅ Medical compliance validation passed"
    fi
}

# Initialize directories and permissions
initialize_directories() {
    log "📁 Initializing application directories..."
    
    # Create required directories
    directories=(
        "/app/logs"
        "/app/data"
        "/app/storage"
        "/app/audit"
        "/tmp/bci"
        "/var/log/bci"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            info "Created directory: $dir"
        fi
        
        # Set appropriate permissions
        if [[ "$dir" == "/app/audit" ]]; then
            chmod 750 "$dir"  # Restrictive permissions for audit logs
        else
            chmod 755 "$dir"
        fi
        
        # Ensure proper ownership
        chown bci:bci "$dir" 2>/dev/null || true
    done
    
    log "✅ Directory initialization completed"
}

# Neural processing system checks
validate_neural_system() {
    log "🧠 Validating neural processing system..."
    
    # Check Python neural processing dependencies
    python3 -c "
import sys
try:
    import numpy
    import scipy
    print('✅ Core scientific libraries available')
except ImportError as e:
    print(f'❌ Missing scientific library: {e}')
    sys.exit(1)

try:
    from bci_agent_bridge.core.bridge import BCIBridge
    print('✅ BCI Bridge module loadable')
except ImportError as e:
    print(f'❌ BCI Bridge import failed: {e}')
    sys.exit(1)
"
    
    if [[ $? -eq 0 ]]; then
        log "✅ Neural processing system validation passed"
    else
        error "Neural processing system validation failed"
        exit 1
    fi
}

# Performance and monitoring setup
setup_monitoring() {
    log "📊 Setting up monitoring and telemetry..."
    
    # Ensure monitoring endpoints are accessible
    if [[ "${PROMETHEUS_ENABLED:-false}" == "true" ]]; then
        info "Prometheus metrics enabled on port 9090"
    fi
    
    if [[ "${JAEGER_ENABLED:-false}" == "true" ]]; then
        info "Jaeger tracing enabled"
    fi
    
    # Setup log rotation for production
    if [[ "$BCI_MODE" == "production" ]]; then
        cat > /app/config/logrotate.conf << 'EOF'
/app/logs/*.log {
    daily
    rotate 90
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    postrotate
        /bin/kill -HUP $(cat /app/logs/bci.pid 2>/dev/null) 2>/dev/null || true
    endscript
}
EOF
        info "Log rotation configured for production"
    fi
    
    log "✅ Monitoring setup completed"
}

# Signal handlers for graceful shutdown
setup_signal_handlers() {
    log "🛡️ Setting up signal handlers for graceful shutdown..."
    
    # Create shutdown script
    cat > /tmp/shutdown.sh << 'EOF'
#!/bin/bash
echo "$(date): Received shutdown signal, initiating graceful shutdown..."

# Send SIGTERM to main process
if [[ -f /app/logs/bci.pid ]]; then
    pid=$(cat /app/logs/bci.pid)
    if kill -0 "$pid" 2>/dev/null; then
        echo "$(date): Sending SIGTERM to process $pid"
        kill -TERM "$pid"
        
        # Wait for graceful shutdown
        for i in {1..30}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "$(date): Process shut down gracefully"
                exit 0
            fi
            sleep 1
        done
        
        echo "$(date): Forcing shutdown with SIGKILL"
        kill -KILL "$pid" 2>/dev/null || true
    fi
fi
exit 0
EOF
    
    chmod +x /tmp/shutdown.sh
    
    # Set up signal traps
    trap '/tmp/shutdown.sh' TERM INT
    
    log "✅ Signal handlers configured"
}

# Main startup sequence
main() {
    log "🚀 Starting BCI-Agent-Bridge Production Environment"
    log "Version: ${BCI_VERSION:-unknown}"
    log "Mode: ${BCI_MODE:-unknown}"
    log "Privacy: ${PRIVACY_MODE:-unknown}"
    
    # Run all validation and setup steps
    validate_environment
    validate_security
    validate_medical_compliance
    check_database
    initialize_directories
    validate_neural_system
    setup_monitoring
    setup_signal_handlers
    
    # Switch to bci user for running the application
    if [[ "$(id -u)" == "0" ]]; then
        log "🔄 Switching to non-root user 'bci'"
        
        # Ensure proper ownership
        chown -R bci:bci /app/logs /app/data /app/storage /app/audit /tmp/bci /var/log/bci 2>/dev/null || true
        
        # Execute the application as bci user
        exec gosu bci "$@"
    else
        log "🔄 Running as user: $(whoami)"
        exec "$@"
    fi
}

# Health check mode
if [[ "${1:-}" == "health-check" ]]; then
    # Simple health check for container orchestration
    curl -f http://localhost:8001/health >/dev/null 2>&1
    exit $?
fi

# Run main startup sequence
main "$@"