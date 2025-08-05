"""
Connection pooling for efficient resource management in BCI operations.
"""

import asyncio
import time
import threading
import logging
from typing import Any, Dict, List, Optional, Callable, Generic, TypeVar, Union
from dataclasses import dataclass
from enum import Enum
import queue
import weakref
from contextlib import asynccontextmanager
import anthropic


T = TypeVar('T')


class PoolState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


@dataclass
class ConnectionMetrics:
    total_created: int = 0
    total_destroyed: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    avg_wait_time: float = 0.0
    max_wait_time: float = 0.0


@dataclass
class PooledConnection:
    connection: Any
    created_at: float
    last_used: float
    use_count: int = 0
    is_healthy: bool = True
    max_lifetime: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    @property
    def age(self) -> float:
        return time.time() - self.created_at
    
    @property
    def idle_time(self) -> float:
        return time.time() - self.last_used
    
    @property
    def is_expired(self) -> bool:
        if self.max_lifetime is None:
            return False
        return self.age > self.max_lifetime
    
    def mark_used(self) -> None:
        self.last_used = time.time()
        self.use_count += 1


class ConnectionPool(Generic[T]):
    """
    Generic connection pool with health monitoring and automatic recovery.
    """
    
    def __init__(self, 
                 connection_factory: Callable[[], T],
                 min_connections: int = 2,
                 max_connections: int = 10,
                 max_idle_time: float = 300.0,  # 5 minutes
                 max_lifetime: float = 3600.0,  # 1 hour
                 health_check: Optional[Callable[[T], bool]] = None,
                 connection_timeout: float = 30.0):
        
        self.connection_factory = connection_factory
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self.max_lifetime = max_lifetime
        self.health_check = health_check
        self.connection_timeout = connection_timeout
        
        self.logger = logging.getLogger(__name__)
        
        # Pool state
        self.state = PoolState.HEALTHY
        self.pool: queue.Queue[PooledConnection] = queue.Queue()
        self.active_connections: Dict[int, PooledConnection] = {}
        self.metrics = ConnectionMetrics()
        
        # Thread safety
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        # Background maintenance
        self._maintenance_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Initialize pool
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize the connection pool."""
        with self._lock:
            for _ in range(self.min_connections):
                try:
                    conn = self._create_connection()
                    if conn:
                        self.pool.put(conn)
                except Exception as e:
                    self.logger.error(f"Failed to initialize pool connection: {e}")
    
    def _create_connection(self) -> Optional[PooledConnection]:
        """Create a new pooled connection."""
        try:
            raw_connection = self.connection_factory()
            
            pooled_conn = PooledConnection(
                connection=raw_connection,
                created_at=time.time(),
                last_used=time.time(),
                max_lifetime=self.max_lifetime
            )
            
            self.metrics.total_created += 1
            self.metrics.active_connections += 1
            
            self.logger.debug("Created new pooled connection")
            return pooled_conn
            
        except Exception as e:
            self.logger.error(f"Failed to create connection: {e}")
            self.metrics.failed_connections += 1
            return None
    
    def _destroy_connection(self, pooled_conn: PooledConnection) -> None:
        """Destroy a pooled connection."""
        try:
            # Attempt graceful cleanup if connection has close method
            if hasattr(pooled_conn.connection, 'close'):
                pooled_conn.connection.close()
            
            self.metrics.total_destroyed += 1
            self.metrics.active_connections = max(0, self.metrics.active_connections - 1)
            
            self.logger.debug("Destroyed pooled connection")
            
        except Exception as e:
            self.logger.error(f"Error destroying connection: {e}")
    
    def _is_connection_healthy(self, pooled_conn: PooledConnection) -> bool:
        """Check if a connection is healthy."""
        try:
            # Check basic conditions
            if not pooled_conn.is_healthy or pooled_conn.is_expired:
                return False
            
            # Check idle time
            if pooled_conn.idle_time > self.max_idle_time:
                return False
            
            # Custom health check
            if self.health_check:
                return self.health_check(pooled_conn.connection)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
    
    async def get_connection(self, timeout: Optional[float] = None) -> PooledConnection:
        """Get a connection from the pool."""
        start_time = time.time()
        timeout = timeout or self.connection_timeout
        
        try:
            # Try to get from pool first
            while True:
                try:
                    pooled_conn = self.pool.get_nowait()
                    
                    if self._is_connection_healthy(pooled_conn):
                        pooled_conn.mark_used()
                        self.active_connections[id(pooled_conn)] = pooled_conn
                        self.metrics.pool_hits += 1
                        
                        wait_time = time.time() - start_time
                        self._update_wait_time_metrics(wait_time)
                        
                        return pooled_conn
                    else:
                        # Connection is unhealthy, destroy it
                        self._destroy_connection(pooled_conn)
                        
                except queue.Empty:
                    break
            
            # No healthy connections available, try to create new one
            if self.metrics.active_connections < self.max_connections:
                pooled_conn = self._create_connection()
                if pooled_conn:
                    pooled_conn.mark_used()
                    self.active_connections[id(pooled_conn)] = pooled_conn
                    self.metrics.pool_misses += 1
                    
                    wait_time = time.time() - start_time
                    self._update_wait_time_metrics(wait_time)
                    
                    return pooled_conn
            
            # Pool is full, wait for a connection to be returned
            with self._condition:
                if not self._condition.wait(timeout):
                    raise TimeoutError(f"Timeout waiting for connection after {timeout}s")
            
            # Retry after waiting
            return await self.get_connection(timeout - (time.time() - start_time))
            
        except Exception as e:
            self.logger.error(f"Failed to get connection: {e}")
            raise
    
    def return_connection(self, pooled_conn: PooledConnection, 
                         is_healthy: bool = True) -> None:
        """Return a connection to the pool."""
        try:
            with self._lock:
                # Remove from active connections
                conn_id = id(pooled_conn)
                if conn_id in self.active_connections:
                    del self.active_connections[conn_id]
                
                # Update health status
                pooled_conn.is_healthy = is_healthy
                
                if is_healthy and not pooled_conn.is_expired:
                    # Return to pool
                    self.pool.put(pooled_conn)
                    self.metrics.idle_connections += 1
                else:
                    # Destroy unhealthy or expired connections
                    self._destroy_connection(pooled_conn)
                
                # Notify waiting threads
                self._condition.notify()
                
        except Exception as e:
            self.logger.error(f"Error returning connection: {e}")
    
    def _update_wait_time_metrics(self, wait_time: float) -> None:
        """Update wait time metrics."""
        if self.metrics.pool_hits + self.metrics.pool_misses == 1:
            self.metrics.avg_wait_time = wait_time
        else:
            total_requests = self.metrics.pool_hits + self.metrics.pool_misses
            self.metrics.avg_wait_time = ((self.metrics.avg_wait_time * (total_requests - 1) + 
                                         wait_time) / total_requests)
        
        self.metrics.max_wait_time = max(self.metrics.max_wait_time, wait_time)
    
    async def cleanup_idle_connections(self) -> int:
        """Clean up idle and expired connections."""
        cleaned = 0
        
        with self._lock:
            # Check connections in pool
            temp_pool = queue.Queue()
            
            while not self.pool.empty():
                try:
                    pooled_conn = self.pool.get_nowait()
                    
                    if self._is_connection_healthy(pooled_conn):
                        temp_pool.put(pooled_conn)
                    else:
                        self._destroy_connection(pooled_conn)
                        cleaned += 1
                        
                except queue.Empty:
                    break
            
            # Replace pool with cleaned connections
            self.pool = temp_pool
            
            # Ensure minimum connections
            while self.pool.qsize() < self.min_connections:
                conn = self._create_connection()
                if conn:
                    self.pool.put(conn)
                else:
                    break
        
        if cleaned > 0:
            self.logger.info(f"Cleaned up {cleaned} idle connections")
        
        return cleaned
    
    async def start_maintenance(self, interval: float = 60.0) -> None:
        """Start background maintenance task."""
        if self._maintenance_task is not None:
            return
        
        async def maintenance_loop():
            while not self._shutdown_event.is_set():
                try:
                    await self.cleanup_idle_connections()
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Maintenance error: {e}")
                    await asyncio.sleep(interval)
        
        self._maintenance_task = asyncio.create_task(maintenance_loop())
        self.logger.info("Started connection pool maintenance")
    
    async def stop_maintenance(self) -> None:
        """Stop background maintenance task."""
        self._shutdown_event.set()
        
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None
        
        self.logger.info("Stopped connection pool maintenance")
    
    def close(self) -> None:
        """Close all connections and shut down pool."""
        with self._lock:
            self.state = PoolState.CLOSED
            
            # Destroy all pooled connections
            while not self.pool.empty():
                try:
                    pooled_conn = self.pool.get_nowait()
                    self._destroy_connection(pooled_conn)
                except queue.Empty:
                    break
            
            # Destroy active connections
            for pooled_conn in list(self.active_connections.values()):
                self._destroy_connection(pooled_conn)
            
            self.active_connections.clear()
        
        self.logger.info("Connection pool closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                "state": self.state.value,
                "pool_size": self.pool.qsize(),
                "active_connections": len(self.active_connections),
                "total_created": self.metrics.total_created,
                "total_destroyed": self.metrics.total_destroyed,
                "failed_connections": self.metrics.failed_connections,
                "pool_hits": self.metrics.pool_hits,
                "pool_misses": self.metrics.pool_misses,
                "hit_rate_pct": round((self.metrics.pool_hits / 
                                     max(1, self.metrics.pool_hits + self.metrics.pool_misses)) * 100, 2),
                "avg_wait_time_ms": round(self.metrics.avg_wait_time * 1000, 2),
                "max_wait_time_ms": round(self.metrics.max_wait_time * 1000, 2),
                "configuration": {
                    "min_connections": self.min_connections,
                    "max_connections": self.max_connections,
                    "max_idle_time": self.max_idle_time,
                    "max_lifetime": self.max_lifetime
                }
            }
    
    @asynccontextmanager
    async def get_connection_context(self, timeout: Optional[float] = None):
        """Context manager for getting and returning connections."""
        connection = None
        try:
            connection = await self.get_connection(timeout)
            yield connection.connection
        except Exception as e:
            if connection:
                self.return_connection(connection, is_healthy=False)
            raise
        else:
            if connection:
                self.return_connection(connection, is_healthy=True)


class ClaudeClientPool(ConnectionPool[anthropic.Anthropic]):
    """
    Specialized connection pool for Claude API clients.
    """
    
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        
        def create_claude_client() -> anthropic.Anthropic:
            return anthropic.Anthropic(api_key=api_key)
        
        def health_check(client: anthropic.Anthropic) -> bool:
            try:
                # Simple health check - just verify the client is usable
                return hasattr(client, 'messages') and client.api_key is not None
            except Exception:
                return False
        
        super().__init__(
            connection_factory=create_claude_client,
            health_check=health_check,
            min_connections=kwargs.get('min_connections', 1),
            max_connections=kwargs.get('max_connections', 5),
            max_idle_time=kwargs.get('max_idle_time', 600.0),  # 10 minutes
            max_lifetime=kwargs.get('max_lifetime', 7200.0),   # 2 hours
            connection_timeout=kwargs.get('connection_timeout', 10.0)
        )
    
    async def make_request(self, model: str, messages: list, **kwargs) -> Any:
        """Make a request using a pooled Claude client."""
        async with self.get_connection_context() as client:
            return await asyncio.to_thread(
                client.messages.create,
                model=model,
                messages=messages,
                **kwargs
            )


class DatabaseConnectionPool(ConnectionPool):
    """
    Database connection pool for clinical data storage.
    """
    
    def __init__(self, connection_string: str, driver: str = "sqlite", **kwargs):
        self.connection_string = connection_string
        self.driver = driver
        
        def create_db_connection():
            if driver == "sqlite":
                import sqlite3
                return sqlite3.connect(connection_string)
            elif driver == "postgresql":
                import psycopg2
                return psycopg2.connect(connection_string)
            else:
                raise ValueError(f"Unsupported database driver: {driver}")
        
        def health_check(conn) -> bool:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                return True
            except Exception:
                return False
        
        super().__init__(
            connection_factory=create_db_connection,
            health_check=health_check,
            **kwargs
        )


# Factory functions for common pool types
def create_claude_pool(api_key: str, **kwargs) -> ClaudeClientPool:
    """Create a Claude API client pool."""
    return ClaudeClientPool(api_key, **kwargs)


def create_database_pool(connection_string: str, driver: str = "sqlite", **kwargs) -> DatabaseConnectionPool:
    """Create a database connection pool."""
    return DatabaseConnectionPool(connection_string, driver, **kwargs)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_connection_pool():
        # Create a simple test pool
        def create_test_connection():
            return {"id": time.time(), "data": "test"}
        
        pool = ConnectionPool(
            connection_factory=create_test_connection,
            min_connections=2,
            max_connections=5
        )
        
        # Start maintenance
        await pool.start_maintenance(interval=5.0)
        
        try:
            # Test getting and returning connections
            conn1 = await pool.get_connection()
            print(f"Got connection 1: {conn1.connection}")
            
            conn2 = await pool.get_connection()
            print(f"Got connection 2: {conn2.connection}")
            
            # Return connections
            pool.return_connection(conn1)
            pool.return_connection(conn2)
            
            # Print stats
            stats = pool.get_stats()
            print("Pool statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
        finally:
            await pool.stop_maintenance()
            pool.close()
    
    asyncio.run(test_connection_pool())