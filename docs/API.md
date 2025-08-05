# BCI-Agent-Bridge API Documentation

## Overview

The BCI-Agent-Bridge provides a comprehensive REST API for brain-computer interface interactions with Claude AI. All endpoints support JSON request/response format and include medical-grade privacy protection.

**Base URL**: `http://localhost:8000/api/v1`

## Authentication

Currently using API key authentication. Include your API key in headers:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8000/api/v1/health
```

## Health & Status

### GET /health

Check system health status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": 1691234567.89,
  "version": "0.1.0",
  "components": {
    "bci_bridge": "healthy",
    "claude_adapter": "healthy",
    "database": "healthy",
    "privacy": "healthy"
  }
}
```

### GET /status

Get detailed system status.

**Response**:
```json
{
  "bci_system": {
    "device": "Simulation",
    "channels": 8,
    "sampling_rate": 250,
    "paradigm": "P300",
    "connected": true,
    "streaming": false
  },
  "streaming": {
    "active": false,
    "buffer_size": 0
  },
  "privacy": {},
  "claude_adapter": {
    "model": "claude-3-sonnet-20240229",
    "safety_mode": "medical"
  },
  "uptime": 1691234567.89,
  "version": "0.1.0"
}
```

## BCI Data Management

### POST /bci/start

Start BCI data streaming.

**Response**:
```json
{
  "message": "BCI streaming started",
  "status": "active"
}
```

### POST /bci/stop

Stop BCI data streaming.

**Response**:
```json
{
  "message": "BCI streaming stopped", 
  "status": "stopped"
}
```

### GET /bci/data

Get recent BCI data from buffer.

**Parameters**:
- `samples` (optional): Number of samples to retrieve (default: 250)

**Response**:
```json
{
  "data": [[1.2, 0.8, ...], [0.9, 1.1, ...]],
  "shape": [8, 250],
  "timestamp": 1691234567.89,
  "channels": 8,
  "sampling_rate": 250
}
```

## Neural Intention Decoding

### POST /decode/intention

Decode intention from current neural data.

**Request**:
```json
{
  "channels": 8,
  "sampling_rate": 250,
  "paradigm": "P300",
  "duration_seconds": 1.0
}
```

**Response**:
```json
{
  "command": "Select current item",
  "confidence": 0.85,
  "context": {
    "paradigm": "P300",
    "prediction": 1,
    "timestamp": 1691234567.89
  },
  "timestamp": 1691234567.89,
  "paradigm": "P300"
}
```

## Claude AI Integration

### POST /claude/execute

Execute neural intention through Claude AI.

**Request**:
```json
{
  "command": "Select current item",
  "confidence": 0.85,
  "context": {
    "paradigm": "P300",
    "user_context": "Menu navigation"
  }
}
```

**Response**:
```json
{
  "response": "I understand you want to select the current menu item. Executing selection now.",
  "reasoning": "High confidence P300 signal detected for selection task",
  "confidence": 0.85,
  "safety_flags": [],
  "processing_time_ms": 245.6,
  "tokens_used": 42
}
```

## Real-time Processing

### POST /realtime/process

Start real-time BCI processing pipeline.

**Response**:
```json
{
  "message": "Real-time processing started",
  "status": "active"
}
```

## Calibration

### POST /calibrate

Calibrate BCI decoder for specific paradigm.

**Parameters**:
- `paradigm` (optional): P300, MotorImagery, SSVEP (default: P300)
- `trials` (optional): Number of calibration trials (default: 50)

**Response**:
```json
{
  "message": "Calibration completed for P300",
  "paradigm": "P300",
  "trials": 50,
  "status": "calibrated",
  "timestamp": 1691234567.89
}
```

## WebSocket API

### WS /ws/stream

Real-time neural data streaming via WebSocket.

**Connection**: `ws://localhost:8000/api/v1/ws/stream`

**Message Format**:
```json
{
  "timestamp": 1691234567.89,
  "data": [[1.2, 0.8], [0.9, 1.1]],
  "channels": ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"],
  "sampling_rate": 250
}
```

## Monitoring

### GET /metrics

Get Prometheus-compatible metrics.

**Response** (text/plain):
```
bci_connected 1
bci_streaming 0
buffer_size 0
channels 8
sampling_rate 250
uptime 1691234567.89
conversation_history_length 5
```

## Error Handling

### Standard Error Response

```json
{
  "detail": "Error message describing what went wrong",
  "status_code": 400
}
```

### Common HTTP Status Codes

- `200`: Success
- `400`: Bad Request - Invalid input parameters
- `401`: Unauthorized - Missing or invalid API key
- `403`: Forbidden - Insufficient permissions
- `404`: Not Found - Endpoint doesn't exist
- `500`: Internal Server Error - System error
- `503`: Service Unavailable - System not ready

## Rate Limiting

API requests are limited to prevent abuse:

- **General endpoints**: 100 requests/minute
- **Streaming endpoints**: 10 requests/minute
- **Real-time processing**: 5 requests/minute

Rate limit headers included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1691234567
```

## Clinical Mode Features

When operating in clinical mode (`PRIVACY_MODE=medical`), additional endpoints and safeguards are available:

### Enhanced Privacy Protection
- Automatic differential privacy application
- Clinical audit logging
- HIPAA-compliant data handling

### Clinical Trial Integration
- Subject enrollment tracking
- Session management
- Adverse event reporting
- Regulatory compliance tools

## Code Examples

### Python Client

```python
import requests
import asyncio
import websockets
import json

class BCIClient:
    def __init__(self, base_url="http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    def health_check(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def start_streaming(self):
        response = requests.post(f"{self.base_url}/bci/start")
        return response.json()
    
    def decode_intention(self, channels=8, duration=1.0):
        data = {
            "channels": channels,
            "sampling_rate": 250,
            "paradigm": "P300",
            "duration_seconds": duration
        }
        response = requests.post(
            f"{self.base_url}/decode/intention", 
            json=data,
            headers=self.headers
        )
        return response.json()
    
    async def stream_realtime(self):
        uri = "ws://localhost:8000/api/v1/ws/stream"
        async with websockets.connect(uri) as websocket:
            while True:
                data = await websocket.recv()
                neural_data = json.loads(data)
                print(f"Received data: {neural_data['timestamp']}")

# Usage
client = BCIClient()
print(client.health_check())
print(client.start_streaming())
intention = client.decode_intention()
print(f"Decoded: {intention['command']} (confidence: {intention['confidence']})")
```

### JavaScript Client

```javascript
class BCIClient {
    constructor(baseUrl = 'http://localhost:8000/api/v1') {
        this.baseUrl = baseUrl;
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        return response.json();
    }
    
    async startStreaming() {
        const response = await fetch(`${this.baseUrl}/bci/start`, {
            method: 'POST'
        });
        return response.json();
    }
    
    async decodeIntention(channels = 8, duration = 1.0) {
        const data = {
            channels,
            sampling_rate: 250,
            paradigm: 'P300',
            duration_seconds: duration
        };
        
        const response = await fetch(`${this.baseUrl}/decode/intention`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        
        return response.json();
    }
    
    streamRealtime() {
        const ws = new WebSocket('ws://localhost:8000/api/v1/ws/stream');
        
        ws.onmessage = (event) => {
            const neuralData = JSON.parse(event.data);
            console.log('Received data:', neuralData.timestamp);
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        return ws;
    }
}

// Usage
const client = new BCIClient();

client.healthCheck().then(health => {
    console.log('System health:', health);
});

client.startStreaming().then(result => {
    console.log('Streaming started:', result);
});

const ws = client.streamRealtime();
```

## Privacy & Security

### Data Protection
- All neural data is processed with differential privacy
- Clinical sessions maintain HIPAA compliance
- Audit trails for all data access
- Encrypted data transmission

### Safety Features
- Confidence thresholds for medical applications
- Adverse event detection
- Emergency stop mechanisms
- Real-time quality monitoring

For additional details on privacy implementation, see the [Privacy Guide](PRIVACY.md).