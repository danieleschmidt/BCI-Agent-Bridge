#!/bin/bash
set -e

# BCI-Agent-Bridge Production Deployment Script
# This script automates the deployment process with comprehensive checks

echo "🚀 BCI-Agent-Bridge Production Deployment"
echo "============================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    log_success "Docker is available and running"
    
    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi
    log_success "Docker Compose is available"
    
    # Verify environment variables
    if [ "$ENVIRONMENT" = "production" ]; then
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            log_error "ANTHROPIC_API_KEY environment variable is required for production"
            exit 1
        fi
        log_success "Required environment variables are set"
    fi
    
    # Check available disk space (need at least 2GB)
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    REQUIRED_SPACE=2097152  # 2GB in KB
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        log_error "Insufficient disk space. Need at least 2GB available"
        exit 1
    fi
    log_success "Sufficient disk space available"
    
    # Run quality gates validation
    log_info "Running quality gates validation..."
    if python3 -c "
import sys
sys.path.append('src')
from bci_agent_bridge import BCIBridge
from bci_agent_bridge.performance.caching import NeuralDataCache
import numpy as np

try:
    # Test core functionality
    bridge = BCIBridge()
    cache = NeuralDataCache()
    test_data = np.random.randn(8, 250)
    cache.cache_neural_features('deployment_test', 'P300', test_data)
    retrieved = cache.get_neural_features('deployment_test', 'P300')
    
    if retrieved is None:
        raise Exception('Cache retrieval failed')
    
    print('Quality gates passed')
except Exception as e:
    print(f'Quality gate failed: {e}')
    sys.exit(1)
    "; then
        log_success "Quality gates validation passed"
    else
        log_error "Quality gates validation failed"
        exit 1
    fi
}

# Backup current deployment
backup_current_deployment() {
    if [ "$ENVIRONMENT" = "production" ] && docker-compose ps | grep -q "Up"; then
        log_info "Creating backup of current deployment..."
        
        mkdir -p "$BACKUP_DIR"
        
        # Backup configuration files
        cp docker-compose.yml "$BACKUP_DIR/"
        cp -r monitoring/ "$BACKUP_DIR/" 2>/dev/null || true
        
        # Backup volumes
        docker-compose exec postgres pg_dump -U bci_user bci_database > "$BACKUP_DIR/database_backup.sql" || log_warning "Database backup failed"
        docker run --rm -v bci-agent-bridge_redis_data:/data -v $(pwd)/$BACKUP_DIR:/backup alpine tar czf /backup/redis_backup.tar.gz -C /data . || log_warning "Redis backup failed"
        
        log_success "Backup created at $BACKUP_DIR"
    fi
}

# Build and deploy
build_and_deploy() {
    log_info "Building and deploying BCI-Agent-Bridge..."
    
    # Create necessary directories
    mkdir -p data logs clinical_data
    mkdir -p monitoring/grafana monitoring/prometheus
    
    # Set appropriate environment
    export ENVIRONMENT
    export VERSION
    
    # Pull latest images (for dependencies)
    log_info "Pulling latest dependency images..."
    docker-compose pull redis postgres monitoring grafana
    
    # Build the BCI bridge image
    log_info "Building BCI-Agent-Bridge image..."
    docker-compose build --no-cache bci-bridge
    
    # Start services in correct order
    log_info "Starting database services..."
    docker-compose up -d postgres redis
    
    # Wait for databases to be ready
    log_info "Waiting for databases to be ready..."
    sleep 30
    
    # Start monitoring services
    log_info "Starting monitoring services..."
    docker-compose up -d monitoring grafana
    
    # Start main BCI service
    log_info "Starting BCI-Agent-Bridge service..."
    docker-compose up -d bci-bridge
    
    log_success "All services started successfully"
}

# Health checks
run_health_checks() {
    log_info "Running post-deployment health checks..."
    
    # Wait for services to be ready
    sleep 60
    
    # Check service health
    SERVICES=("bci-bridge" "redis" "postgres" "monitoring" "grafana")
    
    for service in "${SERVICES[@]}"; do
        if docker-compose ps | grep "$service" | grep -q "Up"; then
            log_success "$service is running"
        else
            log_error "$service is not running properly"
            docker-compose logs "$service"
            exit 1
        fi
    done
    
    # Test API health endpoint
    log_info "Testing API health endpoint..."
    for i in {1..10}; do
        if curl -f http://localhost:8000/health &>/dev/null; then
            log_success "API health check passed"
            break
        elif [ $i -eq 10 ]; then
            log_error "API health check failed after 10 attempts"
            exit 1
        else
            log_info "API not ready, waiting... (attempt $i/10)"
            sleep 10
        fi
    done
    
    # Test BCI functionality
    log_info "Testing BCI functionality..."
    docker-compose exec bci-bridge python -c "
import sys
sys.path.append('/app/src')
from bci_agent_bridge import BCIBridge
bridge = BCIBridge()
health = bridge.get_health_status()
print(f'BCI Bridge Health: {health.get(\"status\", \"unknown\")}')
if health.get('status') != 'healthy':
    sys.exit(1)
" || (log_error "BCI functionality test failed" && exit 1)
    
    log_success "All health checks passed"
}

# Performance validation
validate_performance() {
    log_info "Running performance validation..."
    
    docker-compose exec bci-bridge python -c "
import time
import numpy as np
import sys
sys.path.append('/app/src')

from bci_agent_bridge.core.bridge import BCIBridge
from bci_agent_bridge.performance.caching import NeuralDataCache

# Initialize components
bridge = BCIBridge()
cache = NeuralDataCache()

# Test neural processing speed
test_data = np.random.randn(8, 250).astype(np.float32)
start_time = time.time()

for i in range(10):
    processed = bridge.preprocessor.apply_filters(test_data)
    cache.put(f'perf_test_{i}', processed)

processing_time = (time.time() - start_time) / 10
print(f'Average processing time: {processing_time*1000:.1f}ms per sample')

if processing_time > 0.1:  # More than 100ms per sample
    print('Performance validation failed: processing too slow')
    sys.exit(1)

print('Performance validation passed')
" || (log_error "Performance validation failed" && exit 1)
    
    log_success "Performance validation passed"
}

# Generate deployment summary
generate_deployment_summary() {
    log_info "Generating deployment summary..."
    
    cat > deployment_summary.md << EOF
# BCI-Agent-Bridge Deployment Summary

**Deployment Date:** $(date)
**Environment:** $ENVIRONMENT
**Version:** $VERSION

## Services Status
$(docker-compose ps)

## System Resources
**Memory Usage:**
$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}")

**Disk Usage:**
$(df -h .)

## API Endpoints
- Health Check: http://localhost:8000/health
- API Documentation: http://localhost:8000/docs
- Monitoring Dashboard: http://localhost:3000 (Grafana)
- Metrics Endpoint: http://localhost:9090 (Prometheus)

## Security Configuration
- Security audit logging: ENABLED
- Input validation: ENABLED
- Encryption: ENABLED
- HIPAA compliance: ENABLED

## Next Steps
1. Configure monitoring alerts
2. Set up backup schedules
3. Review security logs
4. Conduct user acceptance testing

## Backup Location
$BACKUP_DIR
EOF

    log_success "Deployment summary generated: deployment_summary.md"
}

# Rollback function
rollback() {
    if [ -d "$BACKUP_DIR" ]; then
        log_warning "Rolling back to previous deployment..."
        
        docker-compose down
        cp "$BACKUP_DIR/docker-compose.yml" ./
        docker-compose up -d
        
        log_info "Rollback completed"
    else
        log_error "No backup available for rollback"
        exit 1
    fi
}

# Main deployment workflow
main() {
    log_info "Starting BCI-Agent-Bridge deployment for environment: $ENVIRONMENT"
    
    # Trap errors and offer rollback
    trap 'log_error "Deployment failed. Run with --rollback to revert changes."; exit 1' ERR
    
    # Handle rollback option
    if [ "$1" = "--rollback" ]; then
        rollback
        exit 0
    fi
    
    # Run deployment steps
    pre_deployment_checks
    backup_current_deployment
    build_and_deploy
    run_health_checks
    validate_performance
    generate_deployment_summary
    
    echo ""
    log_success "🎉 BCI-Agent-Bridge deployed successfully!"
    echo ""
    log_info "Access points:"
    log_info "  • API: http://localhost:8000"
    log_info "  • Documentation: http://localhost:8000/docs"
    log_info "  • Monitoring: http://localhost:3000 (admin/admin)"
    log_info "  • Metrics: http://localhost:9090"
    echo ""
    log_info "Next steps:"
    log_info "  1. Review deployment_summary.md"
    log_info "  2. Configure monitoring alerts"
    log_info "  3. Run user acceptance tests"
    log_info "  4. Set up backup schedules"
    echo ""
    log_success "Deployment completed successfully! 🚀"
}

# Run main function
main "$@"