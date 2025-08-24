#!/bin/bash
# Generation 10 Health Check Script

echo "🔍 Checking Generation 10 system health..."

check_endpoint() {
    local endpoint=$1
    local service=$2
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint/health" 2>/dev/null || echo "000")
    
    if [ "$response" = "200" ]; then
        echo "✅ $service: Healthy"
        return 0
    else
        echo "❌ $service: Unhealthy (HTTP $response)"
        return 1
    fi
}

# Check all services
healthy=0
check_endpoint "http://localhost:8080" "Consciousness Service" && ((healthy++))
check_endpoint "http://localhost:8081" "Performance Engine" && ((healthy++))
check_endpoint "http://localhost:8082" "Symbiosis Service" && ((healthy++))

echo "📊 Health Summary: $healthy/3 services healthy"

if [ $healthy -eq 3 ]; then
    echo "🎯 Generation 10 system is fully operational!"
    exit 0
else
    echo "⚠️ Generation 10 system has issues"
    exit 1
fi
