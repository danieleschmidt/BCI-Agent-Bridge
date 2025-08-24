#!/bin/bash
# Generation 10 Kubernetes Deployment Script

set -e

echo "☸️ Deploying Generation 10 to Kubernetes..."

# Apply namespace
kubectl apply -f k8s/namespace.yaml

# Apply configmaps
kubectl apply -f k8s/configmaps/

# Apply secrets
kubectl apply -f k8s/secrets/

# Deploy consciousness processing service
kubectl apply -f k8s/consciousness-deployment.yaml
kubectl apply -f k8s/consciousness-service.yaml

# Deploy performance engine service
kubectl apply -f k8s/performance-deployment.yaml
kubectl apply -f k8s/performance-service.yaml

# Deploy symbiosis service
kubectl apply -f k8s/symbiosis-deployment.yaml
kubectl apply -f k8s/symbiosis-service.yaml

# Deploy ingress controller
kubectl apply -f k8s/ingress.yaml

# Wait for deployments to be ready
kubectl wait --for=condition=available --timeout=600s deployment/generation10-consciousness -n generation10
kubectl wait --for=condition=available --timeout=600s deployment/generation10-performance -n generation10
kubectl wait --for=condition=available --timeout=600s deployment/generation10-symbiosis -n generation10

echo "✅ Generation 10 Kubernetes deployment complete!"
echo "🌍 Cluster: $(kubectl config current-context)"
echo "📊 Status: kubectl get pods -n generation10"
