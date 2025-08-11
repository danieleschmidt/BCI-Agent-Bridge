"""
Real-time dashboard data feeds with WebSocket support for BCI monitoring.
Provides live streaming of metrics, alerts, and health data to monitoring dashboards.
"""

import asyncio
import json
import time
import logging
import uuid
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import websockets
from websockets.server import WebSocketServerProtocol
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

from .alert_manager import AlertManager, Alert, AlertSeverity
from .health_monitor import HealthMonitor, HealthCheck
from .metrics_collector import MetricsCollector, Metric


class FeedType(Enum):
    METRICS = "metrics"
    ALERTS = "alerts" 
    HEALTH = "health"
    SYSTEM_STATUS = "system_status"
    ANOMALIES = "anomalies"
    LOGS = "logs"
    ALL = "all"


class MessageType(Enum):
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    DATA = "data"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    AUTHENTICATION = "authentication"


@dataclass
class DashboardMessage:
    """Message structure for dashboard communication."""
    message_type: MessageType
    feed_type: Optional[FeedType] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_id: Optional[str] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(asdict(self), default=str, separators=(',', ':'))

    @classmethod
    def from_json(cls, json_str: str) -> 'DashboardMessage':
        """Create message from JSON string."""
        try:
            data = json.loads(json_str)
            return cls(
                message_type=MessageType(data['message_type']),
                feed_type=FeedType(data['feed_type']) if data.get('feed_type') else None,
                data=data.get('data'),
                timestamp=data.get('timestamp', time.time()),
                message_id=data.get('message_id', str(uuid.uuid4())),
                client_id=data.get('client_id'),
                error=data.get('error')
            )
        except Exception as e:
            raise ValueError(f"Invalid message format: {e}")


@dataclass
class ConnectedClient:
    """Represents a connected dashboard client."""
    client_id: str
    websocket: WebSocketServerProtocol
    subscriptions: Set[FeedType] = field(default_factory=set)
    last_heartbeat: float = field(default_factory=time.time)
    authenticated: bool = False
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    connected_at: float = field(default_factory=time.time)
    message_count: int = 0
    error_count: int = 0


class DashboardFeedManager:
    """
    Manages real-time data feeds for monitoring dashboards with WebSocket support.
    """
    
    def __init__(self, 
                 alert_manager: AlertManager,
                 health_monitor: HealthMonitor,
                 metrics_collector: MetricsCollector,
                 host: str = "localhost",
                 port: int = 8765,
                 enable_authentication: bool = True,
                 max_message_queue: int = 1000):
        
        self.alert_manager = alert_manager
        self.health_monitor = health_monitor
        self.metrics_collector = metrics_collector
        self.host = host
        self.port = port
        self.enable_authentication = enable_authentication
        self.max_message_queue = max_message_queue
        
        self.logger = logging.getLogger(__name__)
        
        # Client management
        self.connected_clients: Dict[str, ConnectedClient] = {}
        self.client_lock = threading.Lock()
        
        # Message queuing
        self.message_queues: Dict[FeedType, deque] = {
            feed_type: deque(maxlen=max_message_queue) 
            for feed_type in FeedType if feed_type != FeedType.ALL
        }
        
        # WebSocket server
        self.server: Optional[websockets.WebSocketServer] = None
        self.server_task: Optional[asyncio.Task] = None
        
        # Background tasks
        self.feed_tasks: Dict[FeedType, asyncio.Task] = {}
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.client_timeout = 300  # seconds
        self.update_intervals = {
            FeedType.METRICS: 1.0,      # 1 second
            FeedType.ALERTS: 0.5,       # 500ms
            FeedType.HEALTH: 5.0,       # 5 seconds
            FeedType.SYSTEM_STATUS: 10.0, # 10 seconds
            FeedType.ANOMALIES: 2.0,    # 2 seconds
            FeedType.LOGS: 1.0          # 1 second
        }
        
        # Statistics
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'start_time': time.time(),
            'last_activity': time.time()
        }
        
        # Authentication tokens (in production, use proper auth system)
        self.valid_tokens = {
            "dashboard_admin": "admin",
            "dashboard_viewer": "viewer",
            "dashboard_operator": "operator"
        }
        
        # Thread pool for background processing
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="dashboard-feed")

    async def start_server(self) -> None:
        """Start the WebSocket server and all feed tasks."""
        try:
            # Start WebSocket server
            self.server = await websockets.serve(
                self._handle_client_connection,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.logger.info(f"Dashboard feed server started on {self.host}:{self.port}")
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.stats['start_time'] = time.time()
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard server: {e}")
            raise

    async def stop_server(self) -> None:
        """Stop the WebSocket server and all tasks."""
        try:
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Close all client connections
            with self.client_lock:
                for client in list(self.connected_clients.values()):
                    await self._disconnect_client(client.client_id, "Server shutting down")
            
            # Stop WebSocket server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                self.server = None
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Dashboard feed server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping dashboard server: {e}")

    async def _start_background_tasks(self) -> None:
        """Start all background feed tasks."""
        # Start individual feed tasks
        for feed_type in FeedType:
            if feed_type != FeedType.ALL:
                self.feed_tasks[feed_type] = asyncio.create_task(
                    self._feed_loop(feed_type)
                )
        
        # Start utility tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Dashboard background tasks started")

    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks."""
        # Stop feed tasks
        for task in self.feed_tasks.values():
            if not task.done():
                task.cancel()
        
        # Stop utility tasks
        for task in [self.heartbeat_task, self.cleanup_task]:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        all_tasks = list(self.feed_tasks.values())
        if self.heartbeat_task:
            all_tasks.append(self.heartbeat_task)
        if self.cleanup_task:
            all_tasks.append(self.cleanup_task)
        
        for task in all_tasks:
            if not task.done():
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Dashboard background tasks stopped")

    async def _handle_client_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle new client connection."""
        client_id = str(uuid.uuid4())
        client_info = websocket.request_headers
        
        client = ConnectedClient(
            client_id=client_id,
            websocket=websocket,
            user_agent=client_info.get('User-Agent'),
            ip_address=websocket.remote_address[0] if websocket.remote_address else None
        )
        
        with self.client_lock:
            self.connected_clients[client_id] = client
            self.stats['total_connections'] += 1
            self.stats['active_connections'] += 1
        
        self.logger.info(f"New dashboard client connected: {client_id} from {client.ip_address}")
        
        try:
            # Send welcome message
            welcome_msg = DashboardMessage(
                message_type=MessageType.DATA,
                data={
                    "type": "welcome",
                    "client_id": client_id,
                    "server_info": {
                        "version": "1.0.0",
                        "features": ["metrics", "alerts", "health", "anomalies"],
                        "authentication_required": self.enable_authentication
                    }
                }
            )
            await self._send_message_to_client(client_id, welcome_msg)
            
            # Handle client messages
            async for message in websocket:
                await self._handle_client_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
        finally:
            await self._disconnect_client(client_id, "Connection closed")

    async def _handle_client_message(self, client_id: str, message: str) -> None:
        """Handle message from client."""
        try:
            client = self.connected_clients.get(client_id)
            if not client:
                return
            
            client.message_count += 1
            self.stats['messages_received'] += 1
            self.stats['last_activity'] = time.time()
            
            # Parse message
            try:
                dashboard_msg = DashboardMessage.from_json(message)
            except ValueError as e:
                await self._send_error(client_id, f"Invalid message format: {e}")
                return
            
            # Handle different message types
            if dashboard_msg.message_type == MessageType.AUTHENTICATION:
                await self._handle_authentication(client_id, dashboard_msg)
            
            elif dashboard_msg.message_type == MessageType.SUBSCRIBE:
                await self._handle_subscription(client_id, dashboard_msg)
            
            elif dashboard_msg.message_type == MessageType.UNSUBSCRIBE:
                await self._handle_unsubscription(client_id, dashboard_msg)
            
            elif dashboard_msg.message_type == MessageType.HEARTBEAT:
                client.last_heartbeat = time.time()
                # Echo heartbeat back
                response = DashboardMessage(
                    message_type=MessageType.HEARTBEAT,
                    data={"status": "alive", "server_time": time.time()}
                )
                await self._send_message_to_client(client_id, response)
            
            else:
                await self._send_error(client_id, f"Unknown message type: {dashboard_msg.message_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing message from {client_id}: {e}")
            await self._send_error(client_id, "Internal server error")

    async def _handle_authentication(self, client_id: str, message: DashboardMessage) -> None:
        """Handle client authentication."""
        client = self.connected_clients.get(client_id)
        if not client:
            return
        
        if not self.enable_authentication:
            client.authenticated = True
            response = DashboardMessage(
                message_type=MessageType.AUTHENTICATION,
                data={"status": "success", "message": "Authentication disabled"}
            )
            await self._send_message_to_client(client_id, response)
            return
        
        token = message.data.get('token') if message.data else None
        if not token:
            await self._send_error(client_id, "Authentication token required")
            return
        
        # Validate token (in production, use proper auth system)
        if token in self.valid_tokens:
            client.authenticated = True
            role = self.valid_tokens[token]
            response = DashboardMessage(
                message_type=MessageType.AUTHENTICATION,
                data={"status": "success", "role": role}
            )
            await self._send_message_to_client(client_id, response)
            self.logger.info(f"Client {client_id} authenticated as {role}")
        else:
            await self._send_error(client_id, "Invalid authentication token")

    async def _handle_subscription(self, client_id: str, message: DashboardMessage) -> None:
        """Handle client subscription request."""
        client = self.connected_clients.get(client_id)
        if not client:
            return
        
        if self.enable_authentication and not client.authenticated:
            await self._send_error(client_id, "Authentication required")
            return
        
        feed_type = message.feed_type
        if not feed_type:
            await self._send_error(client_id, "Feed type required for subscription")
            return
        
        with self.client_lock:
            if feed_type == FeedType.ALL:
                # Subscribe to all feeds
                for ft in FeedType:
                    if ft != FeedType.ALL:
                        client.subscriptions.add(ft)
            else:
                client.subscriptions.add(feed_type)
        
        response = DashboardMessage(
            message_type=MessageType.DATA,
            data={
                "type": "subscription_confirmed", 
                "feed_type": feed_type.value,
                "active_subscriptions": [ft.value for ft in client.subscriptions]
            }
        )
        await self._send_message_to_client(client_id, response)
        
        self.logger.info(f"Client {client_id} subscribed to {feed_type.value}")
        
        # Send current state for the subscribed feed
        await self._send_current_state(client_id, feed_type)

    async def _handle_unsubscription(self, client_id: str, message: DashboardMessage) -> None:
        """Handle client unsubscription request."""
        client = self.connected_clients.get(client_id)
        if not client:
            return
        
        feed_type = message.feed_type
        if not feed_type:
            await self._send_error(client_id, "Feed type required for unsubscription")
            return
        
        with self.client_lock:
            if feed_type == FeedType.ALL:
                client.subscriptions.clear()
            else:
                client.subscriptions.discard(feed_type)
        
        response = DashboardMessage(
            message_type=MessageType.DATA,
            data={
                "type": "unsubscription_confirmed",
                "feed_type": feed_type.value,
                "active_subscriptions": [ft.value for ft in client.subscriptions]
            }
        )
        await self._send_message_to_client(client_id, response)
        
        self.logger.info(f"Client {client_id} unsubscribed from {feed_type.value}")

    async def _send_current_state(self, client_id: str, feed_type: FeedType) -> None:
        """Send current state for a feed type to client."""
        try:
            if feed_type == FeedType.METRICS:
                data = await self._get_current_metrics()
            elif feed_type == FeedType.ALERTS:
                data = await self._get_current_alerts()
            elif feed_type == FeedType.HEALTH:
                data = await self._get_current_health()
            elif feed_type == FeedType.SYSTEM_STATUS:
                data = await self._get_system_status()
            elif feed_type == FeedType.ANOMALIES:
                data = await self._get_current_anomalies()
            elif feed_type == FeedType.ALL:
                # Send all current states
                for ft in FeedType:
                    if ft != FeedType.ALL:
                        await self._send_current_state(client_id, ft)
                return
            else:
                return
            
            message = DashboardMessage(
                message_type=MessageType.DATA,
                feed_type=feed_type,
                data={"type": "current_state", "content": data}
            )
            await self._send_message_to_client(client_id, message)
            
        except Exception as e:
            self.logger.error(f"Error sending current state to {client_id}: {e}")

    async def _send_message_to_client(self, client_id: str, message: DashboardMessage) -> None:
        """Send message to specific client."""
        client = self.connected_clients.get(client_id)
        if not client:
            return
        
        try:
            message.client_id = client_id
            await client.websocket.send(message.to_json())
            self.stats['messages_sent'] += 1
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_id} connection closed during send")
            await self._disconnect_client(client_id, "Connection closed")
        except Exception as e:
            self.logger.error(f"Error sending message to {client_id}: {e}")
            client.error_count += 1
            self.stats['errors'] += 1

    async def _broadcast_message(self, message: DashboardMessage, feed_type: FeedType) -> None:
        """Broadcast message to all clients subscribed to feed type."""
        if not self.connected_clients:
            return
        
        # Add to message queue
        if feed_type != FeedType.ALL:
            self.message_queues[feed_type].append(message)
        
        # Send to subscribed clients
        clients_to_send = []
        with self.client_lock:
            for client in self.connected_clients.values():
                if feed_type in client.subscriptions:
                    clients_to_send.append(client.client_id)
        
        # Send messages concurrently
        tasks = [
            self._send_message_to_client(client_id, message)
            for client_id in clients_to_send
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_error(self, client_id: str, error_message: str) -> None:
        """Send error message to client."""
        error_msg = DashboardMessage(
            message_type=MessageType.ERROR,
            error=error_message
        )
        await self._send_message_to_client(client_id, error_msg)

    async def _disconnect_client(self, client_id: str, reason: str) -> None:
        """Disconnect and clean up client."""
        with self.client_lock:
            client = self.connected_clients.pop(client_id, None)
            if client:
                self.stats['active_connections'] -= 1
        
        if client:
            try:
                await client.websocket.close()
            except Exception:
                pass  # Connection might already be closed
            
            self.logger.info(f"Client {client_id} disconnected: {reason}")

    # Feed data generation methods
    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics data."""
        summaries = self.metrics_collector.get_all_summaries()
        current_time = time.time()
        
        # Get recent metrics for trending
        recent_metrics = {}
        for name, summary in summaries.items():
            recent_data = self.metrics_collector.get_recent_metrics(name, seconds=60)
            recent_metrics[name] = {
                "summary": {
                    "count": summary.count,
                    "min": summary.min_value,
                    "max": summary.max_value,
                    "avg": summary.avg_value,
                    "last": summary.last_value,
                    "unit": summary.unit,
                    "tags": summary.tags
                },
                "recent_data": [
                    {"timestamp": m.timestamp, "value": m.value}
                    for m in recent_data[-20:]  # Last 20 data points
                ]
            }
        
        return {
            "timestamp": current_time,
            "metrics": recent_metrics,
            "counters": dict(self.metrics_collector.counters),
            "gauges": dict(self.metrics_collector.gauges)
        }

    async def _get_current_alerts(self) -> Dict[str, Any]:
        """Get current alerts data."""
        active_alerts = self.alert_manager.get_active_alerts()
        alert_summary = self.alert_manager.get_alert_summary()
        
        return {
            "timestamp": time.time(),
            "active_alerts": [alert.to_dict() for alert in active_alerts],
            "summary": alert_summary
        }

    async def _get_current_health(self) -> Dict[str, Any]:
        """Get current health data."""
        health_summary = self.health_monitor.get_comprehensive_health_summary()
        return {
            "timestamp": time.time(),
            **health_summary
        }

    async def _get_system_status(self) -> Dict[str, Any]:
        """Get system status data."""
        return {
            "timestamp": time.time(),
            "dashboard_server": {
                "active_connections": self.stats['active_connections'],
                "total_connections": self.stats['total_connections'],
                "messages_sent": self.stats['messages_sent'],
                "messages_received": self.stats['messages_received'],
                "errors": self.stats['errors'],
                "uptime": time.time() - self.stats['start_time']
            },
            "alert_manager": self.alert_manager.get_alert_summary(),
            "health_monitor": {
                "monitoring_active": self.health_monitor.monitoring_active,
                "check_interval": self.health_monitor.check_interval
            },
            "metrics_collector": {
                "total_metrics": len(self.metrics_collector.metrics),
                "retention_period": self.metrics_collector.retention_period
            }
        }

    async def _get_current_anomalies(self) -> Dict[str, Any]:
        """Get current anomalies data."""
        if hasattr(self.metrics_collector, 'get_anomalies'):
            recent_anomalies = self.metrics_collector.get_anomalies(
                start_time=time.time() - 3600  # Last hour
            )
            
            return {
                "timestamp": time.time(),
                "recent_anomalies": [anomaly.to_dict() for anomaly in recent_anomalies],
                "summary": {
                    "total_recent": len(recent_anomalies),
                    "by_type": {},
                    "by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0}
                }
            }
        else:
            return {"timestamp": time.time(), "anomalies": [], "summary": {}}

    # Background task loops
    async def _feed_loop(self, feed_type: FeedType) -> None:
        """Main loop for a specific feed type."""
        interval = self.update_intervals.get(feed_type, 5.0)
        
        while True:
            try:
                # Get data based on feed type
                if feed_type == FeedType.METRICS:
                    data = await self._get_current_metrics()
                elif feed_type == FeedType.ALERTS:
                    data = await self._get_current_alerts()
                elif feed_type == FeedType.HEALTH:
                    data = await self._get_current_health()
                elif feed_type == FeedType.SYSTEM_STATUS:
                    data = await self._get_system_status()
                elif feed_type == FeedType.ANOMALIES:
                    data = await self._get_current_anomalies()
                else:
                    await asyncio.sleep(interval)
                    continue
                
                # Create and broadcast message
                message = DashboardMessage(
                    message_type=MessageType.DATA,
                    feed_type=feed_type,
                    data={"type": "update", "content": data}
                )
                
                await self._broadcast_message(message, feed_type)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in {feed_type.value} feed loop: {e}")
                await asyncio.sleep(interval)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to all clients."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if not self.connected_clients:
                    continue
                
                heartbeat_msg = DashboardMessage(
                    message_type=MessageType.HEARTBEAT,
                    data={
                        "server_time": time.time(),
                        "active_connections": len(self.connected_clients)
                    }
                )
                
                # Send to all clients
                tasks = [
                    self._send_message_to_client(client_id, heartbeat_msg)
                    for client_id in list(self.connected_clients.keys())
                ]
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")

    async def _cleanup_loop(self) -> None:
        """Clean up inactive clients and old messages."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = time.time()
                
                # Clean up inactive clients
                inactive_clients = []
                with self.client_lock:
                    for client_id, client in self.connected_clients.items():
                        if (current_time - client.last_heartbeat) > self.client_timeout:
                            inactive_clients.append(client_id)
                
                for client_id in inactive_clients:
                    await self._disconnect_client(client_id, "Client timeout")
                
                # Clean up old messages from queues (already handled by maxlen)
                # Log cleanup stats
                if inactive_clients:
                    self.logger.info(f"Cleaned up {len(inactive_clients)} inactive clients")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    def get_server_stats(self) -> Dict[str, Any]:
        """Get dashboard server statistics."""
        current_time = time.time()
        uptime = current_time - self.stats['start_time']
        
        client_stats = {}
        with self.client_lock:
            client_stats = {
                "total_clients": len(self.connected_clients),
                "authenticated_clients": len([c for c in self.connected_clients.values() if c.authenticated]),
                "subscription_counts": defaultdict(int)
            }
            
            for client in self.connected_clients.values():
                for subscription in client.subscriptions:
                    client_stats["subscription_counts"][subscription.value] += 1
        
        return {
            "uptime_seconds": uptime,
            "server_info": {
                "host": self.host,
                "port": self.port,
                "authentication_enabled": self.enable_authentication
            },
            "statistics": {
                **self.stats,
                "messages_per_second": self.stats['messages_sent'] / max(uptime, 1),
                "active_connections": len(self.connected_clients)
            },
            "clients": client_stats,
            "feed_queues": {
                feed_type.value: len(queue)
                for feed_type, queue in self.message_queues.items()
            }
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from unittest.mock import Mock
    
    async def demo_dashboard_feeds():
        print("BCI Dashboard Feeds Demo")
        print("=" * 40)
        
        # Create mock components
        alert_manager = Mock()
        alert_manager.get_active_alerts.return_value = []
        alert_manager.get_alert_summary.return_value = {"total_active": 0}
        
        health_monitor = Mock()
        health_monitor.get_comprehensive_health_summary.return_value = {"overall_health": {"status": "healthy"}}
        health_monitor.monitoring_active = True
        health_monitor.check_interval = 30.0
        
        metrics_collector = Mock()
        metrics_collector.get_all_summaries.return_value = {}
        metrics_collector.get_recent_metrics.return_value = []
        metrics_collector.counters = {}
        metrics_collector.gauges = {}
        metrics_collector.metrics = {}
        metrics_collector.retention_period = 3600
        
        # Create dashboard feed manager
        feed_manager = DashboardFeedManager(
            alert_manager=alert_manager,
            health_monitor=health_monitor,
            metrics_collector=metrics_collector,
            host="localhost",
            port=8765,
            enable_authentication=False  # Disable auth for demo
        )
        
        print(f"Starting dashboard server on {feed_manager.host}:{feed_manager.port}")
        
        try:
            # Start server
            await feed_manager.start_server()
            
            print("Dashboard server started successfully!")
            print("Connect to ws://localhost:8765 to test")
            print("Example client messages:")
            print('  Subscribe to metrics: {"message_type": "subscribe", "feed_type": "metrics"}')
            print('  Subscribe to alerts: {"message_type": "subscribe", "feed_type": "alerts"}')
            print('  Heartbeat: {"message_type": "heartbeat"}')
            print("\nServer will run for 30 seconds...")
            
            # Let server run for demo
            await asyncio.sleep(30)
            
            # Show stats
            stats = feed_manager.get_server_stats()
            print("\n--- Server Statistics ---")
            print(json.dumps(stats, indent=2, default=str))
            
        finally:
            # Stop server
            await feed_manager.stop_server()
            print("Dashboard server stopped")
    
    # Run demo
    asyncio.run(demo_dashboard_feeds())