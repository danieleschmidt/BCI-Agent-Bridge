# BCI-Agent-Bridge Deployment Guide

## Production Deployment

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM
- 2+ CPU cores
- 50GB+ storage
- Linux/Windows/macOS

### Quick Start

1. **Clone and Configure**
   ```bash
   git clone https://github.com/danieleschmidt/quantum-inspired-task-planner.git
   cd quantum-inspired-task-planner
   cp .env.example .env
   ```

2. **Set Required Environment Variables**
   ```bash
   # Edit .env file
   ANTHROPIC_API_KEY=your_api_key_here
   POSTGRES_PASSWORD=secure_password
   SECRET_KEY=your_secret_key
   ```

3. **Deploy with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Verify Deployment**
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

### Environment Configuration

#### Core Settings
```bash
# BCI Hardware
BCI_DEVICE=Simulation        # Simulation, OpenBCI, Emotiv, etc.
BCI_CHANNELS=8              # Number of EEG channels
BCI_SAMPLING_RATE=250       # Sampling rate in Hz
BCI_PARADIGM=P300          # P300, MotorImagery, SSVEP

# Privacy & Security
PRIVACY_EPSILON=1.0         # Differential privacy parameter
PRIVACY_DELTA=1e-5         # Privacy failure probability
PRIVACY_MODE=medical       # medical, standard, research
```

#### Clinical Configuration
```bash
# Clinical Trial Settings
IRB_APPROVAL=IRB-2025-0142
PROTOCOL_ID=BCI-LLM-2025-001
CLINICAL_DATA_DIR=./clinical_data
```

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BCI Hardware  │ -> │  BCI-Bridge     │ -> │  Claude API     │
│   (OpenBCI/Sim) │    │  (Processing)   │    │  (LLM Engine)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │ <- │   FastAPI       │ -> │   Redis Cache   │
│   (Clinical DB) │    │   (REST API)    │    │   (Sessions)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              v
┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │ <- │   Grafana       │
│   (Metrics)     │    │   (Dashboard)   │
└─────────────────┘    └─────────────────┘
```

### Service Endpoints

| Service | Port | Endpoint | Purpose |
|---------|------|----------|---------|
| API | 8000 | `/api/v1/` | REST API |
| WebSocket | 8080 | `/ws/stream` | Real-time data |
| Grafana | 3000 | `/dashboard` | Monitoring |
| Prometheus | 9090 | `/metrics` | Metrics |
| PostgreSQL | 5432 | N/A | Database |
| Redis | 6379 | N/A | Cache |

### Health Checks

The system includes comprehensive health monitoring:

```bash
# API Health
curl http://localhost:8000/api/v1/health

# System Status
curl http://localhost:8000/api/v1/status

# Prometheus Metrics
curl http://localhost:8000/api/v1/metrics
```

### Security Configuration

#### SSL/TLS Setup
```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Update docker-compose.yml
volumes:
  - ./cert.pem:/app/cert.pem
  - ./key.pem:/app/key.pem
```

#### Firewall Rules
```bash
# Allow only necessary ports
ufw allow 8000/tcp  # API
ufw allow 8080/tcp  # WebSocket
ufw allow 3000/tcp  # Grafana (optional)
ufw deny 5432/tcp   # PostgreSQL (internal only)
ufw deny 6379/tcp   # Redis (internal only)
```

### Scaling Configuration

#### Horizontal Scaling
```yaml
# docker-compose.override.yml
services:
  bci-bridge:
    deploy:
      replicas: 3
    environment:
      - WORKER_PROCESSES=2
```

#### Resource Limits
```yaml
services:
  bci-bridge:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Backup Strategy

#### Database Backup
```bash
# Automated backup script
#!/bin/bash
docker exec bci-postgres pg_dump -U bci_user bci_database > backup_$(date +%Y%m%d).sql
```

#### Clinical Data Backup
```bash
# HIPAA-compliant encrypted backup
tar -czf clinical_backup_$(date +%Y%m%d).tar.gz clinical_data/
gpg --cipher-algo AES256 --compress-algo 1 --s2k-cipher-algo AES256 --s2k-digest-algo SHA512 --s2k-mode 3 --s2k-count 65536 --symmetric clinical_backup_*.tar.gz
```

### Monitoring Alerts

Critical alerts are configured for:

- High processing latency (>100ms)
- Low signal quality (<70%)
- Privacy budget exhaustion
- Adverse events in clinical sessions
- System resource usage
- Service availability

### Disaster Recovery

1. **Data Recovery**
   ```bash
   # Restore database
   docker exec -i bci-postgres psql -U bci_user bci_database < backup.sql
   
   # Restore clinical data
   gpg --decrypt clinical_backup.tar.gz.gpg | tar -xzf -
   ```

2. **Service Recovery**
   ```bash
   # Restart all services
   docker-compose down
   docker-compose up -d
   
   # Check service health
   docker-compose ps
   ```

### Performance Tuning

#### PostgreSQL Optimization
```sql
-- postgresql.conf optimizations
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
```

#### Redis Optimization
```bash
# redis.conf
maxmemory 512mb
maxmemory-policy allkeys-lru
```

### Troubleshooting

#### Common Issues

1. **High Latency**
   - Check CPU/memory usage
   - Verify network connectivity
   - Review processing pipeline

2. **Connection Errors**
   - Verify service health
   - Check firewall rules
   - Review authentication

3. **Data Quality Issues**
   - Verify BCI hardware connection
   - Check signal preprocessing
   - Review calibration status

#### Log Analysis
```bash
# View application logs
docker-compose logs -f bci-bridge

# View specific service logs
docker-compose logs postgres
docker-compose logs redis
```

### Compliance & Validation

#### HIPAA Compliance
- Encrypted data at rest and in transit
- Access logging and audit trails
- Secure backup procedures
- Regular security assessments

#### FDA Validation
- Clinical trial management
- Data integrity verification
- Adverse event reporting
- Regulatory documentation

### Maintenance Schedule

- **Daily**: Health checks, log review
- **Weekly**: Performance analysis, backup verification
- **Monthly**: Security updates, system optimization
- **Quarterly**: Compliance audit, disaster recovery testing