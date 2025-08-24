#!/bin/bash
# Generation 10 Docker Deployment Script

set -e

echo "🚀 Deploying Generation 10 Ultra-Autonomous System..."

# Build Docker image
docker build -t generation10-system:latest .

# Create network
docker network create generation10-network || true

# Deploy consciousness processing service
docker run -d \
  --name generation10-consciousness \
  --network generation10-network \
  -p 8080:8080 \
  -e SERVICE_TYPE=consciousness \
  -e REGION=$DEPLOYMENT_REGION \
  generation10-system:latest

# Deploy performance engine service  
docker run -d \
  --name generation10-performance \
  --network generation10-network \
  -p 8081:8081 \
  -e SERVICE_TYPE=performance \
  -e REGION=$DEPLOYMENT_REGION \
  generation10-system:latest

# Deploy symbiosis service
docker run -d \
  --name generation10-symbiosis \
  --network generation10-network \
  -p 8082:8082 \
  -e SERVICE_TYPE=symbiosis \
  -e REGION=$DEPLOYMENT_REGION \
  generation10-system:latest

# Deploy load balancer
docker run -d \
  --name generation10-loadbalancer \
  --network generation10-network \
  -p 80:80 \
  -p 443:443 \
  -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf \
  nginx:alpine

echo "✅ Generation 10 deployment complete!"
echo "🌍 Region: $DEPLOYMENT_REGION"
echo "🔗 Access URL: https://generation10-${DEPLOYMENT_REGION}.terragon.ai"
