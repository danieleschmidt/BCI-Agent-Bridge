#!/bin/bash
# BCI-Agent-Bridge Health Check Script
# Comprehensive health validation for medical-grade deployment

set -euo pipefail

# Configuration
HEALTH_ENDPOINT="http://localhost:8001/health"
API_ENDPOINT="http://localhost:8000/api/v1/health"
TIMEOUT=10
MAX_RETRIES=3

# Exit codes
EXIT_SUCCESS=0
EXIT_FAILURE=1
EXIT_WARNING=2

# Logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >&2
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# Check if curl is available
if ! command -v curl >/dev/null 2>&1; then
    error "curl command not found"
    exit $EXIT_FAILURE
fi

# Function to check HTTP endpoint
check_endpoint() {
    local endpoint="$1"
    local name="$2"
    local retry_count=0
    
    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        if curl -f -s --max-time $TIMEOUT "$endpoint" >/dev/null 2>&1; then
            log "$name endpoint healthy"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        if [[ $retry_count -lt $MAX_RETRIES ]]; then
            log "$name endpoint check failed, retrying ($retry_count/$MAX_RETRIES)..."
            sleep 2
        fi
    done
    
    error "$name endpoint unhealthy after $MAX_RETRIES attempts"
    return 1
}

# Function to check detailed health status
check_detailed_health() {
    local response
    if response=$(curl -f -s --max-time $TIMEOUT "$API_ENDPOINT" 2>/dev/null); then
        # Parse JSON response
        local status
        if command -v jq >/dev/null 2>&1; then
            status=$(echo "$response" | jq -r '.status // "unknown"')
            local bci_status=$(echo "$response" | jq -r '.components.bci_bridge // "unknown"')
            local claude_status=$(echo "$response" | jq -r '.components.claude_adapter // "unknown"')
            
            log "Detailed health - Status: $status, BCI: $bci_status, Claude: $claude_status"
            
            if [[ "$status" == "healthy" ]]; then
                return 0
            elif [[ "$status" == "degraded" ]]; then
                log "System is degraded but operational"
                return $EXIT_WARNING
            else
                error "System status is: $status"
                return 1
            fi
        else
            # Basic string check without jq
            if echo "$response" | grep -q '"status":"healthy"'; then
                log "System reports healthy status"
                return 0
            elif echo "$response" | grep -q '"status":"degraded"'; then
                log "System reports degraded status"
                return $EXIT_WARNING
            else
                error "System reports unhealthy status"
                return 1
            fi
        fi
    else
        error "Failed to get detailed health status"
        return 1
    fi
}

# Function to check process health
check_process_health() {
    # Check if main process is running
    if [[ -f /app/logs/bci.pid ]]; then
        local pid
        pid=$(cat /app/logs/bci.pid 2>/dev/null || echo "")
        
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log "Main process (PID: $pid) is running"
            return 0
        else
            error "Main process not found or not responding"
            return 1
        fi
    else
        # If no PID file, check by process name
        if pgrep -f "bci_agent_bridge" >/dev/null 2>&1; then
            log "BCI process detected"
            return 0
        else
            error "No BCI process found"
            return 1
        fi
    fi
}

# Function to check file system health
check_filesystem_health() {
    local required_dirs=(
        "/app/logs"
        "/app/data"
        "/app/storage"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            error "Required directory missing: $dir"
            return 1
        fi
        
        # Check if directory is writable
        if [[ ! -w "$dir" ]]; then
            error "Directory not writable: $dir"
            return 1
        fi
    done
    
    # Check disk space (warn if less than 1GB free)
    local available_space
    available_space=$(df /app | tail -1 | awk '{print $4}')
    if [[ "$available_space" -lt 1048576 ]]; then  # 1GB in KB
        error "Low disk space: ${available_space}KB available"
        return 1
    fi
    
    log "File system health check passed"
    return 0
}

# Function to check memory usage
check_memory_health() {
    # Get memory usage of current process
    local mem_usage
    if command -v ps >/dev/null 2>&1; then
        mem_usage=$(ps -o pid,pmem,rss -p $$ | tail -1 | awk '{print $3}')
        
        # Convert to MB
        local mem_mb=$((mem_usage / 1024))
        
        # Warn if using more than 6GB (for 8GB limit)
        if [[ $mem_mb -gt 6144 ]]; then
            error "High memory usage: ${mem_mb}MB"
            return 1
        fi
        
        log "Memory usage: ${mem_mb}MB"
    fi
    
    return 0
}

# Main health check function
main_health_check() {
    local exit_code=0
    local warnings=0
    
    log "Starting comprehensive health check..."
    
    # Basic endpoint checks
    if ! check_endpoint "$HEALTH_ENDPOINT" "Health"; then
        exit_code=$EXIT_FAILURE
    fi
    
    if ! check_endpoint "$API_ENDPOINT" "API"; then
        exit_code=$EXIT_FAILURE
    fi
    
    # Detailed health check
    local detailed_result
    check_detailed_health
    detailed_result=$?
    
    if [[ $detailed_result -eq 1 ]]; then
        exit_code=$EXIT_FAILURE
    elif [[ $detailed_result -eq $EXIT_WARNING ]]; then
        warnings=1
    fi
    
    # Process health check
    if ! check_process_health; then
        exit_code=$EXIT_FAILURE
    fi
    
    # File system health check
    if ! check_filesystem_health; then
        exit_code=$EXIT_FAILURE
    fi
    
    # Memory health check
    if ! check_memory_health; then
        exit_code=$EXIT_FAILURE
    fi
    
    # Final status determination
    if [[ $exit_code -eq $EXIT_FAILURE ]]; then
        error "Health check FAILED"
        return $EXIT_FAILURE
    elif [[ $warnings -eq 1 ]]; then
        log "Health check PASSED with warnings"
        return $EXIT_WARNING
    else
        log "Health check PASSED"
        return $EXIT_SUCCESS
    fi
}

# Execute main health check
main_health_check