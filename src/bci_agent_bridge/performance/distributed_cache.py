"""
Distributed caching system for BCI data across multiple nodes.
"""

import asyncio
import time
import json
import logging
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .caching import CacheManager, CachePolicy, CacheEntry


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_LOCAL = "l1_local"        # In-memory local cache
    L2_SHARED = "l2_shared"      # Shared memory/Redis cache  
    L3_DISTRIBUTED = "l3_distributed"  # Distributed cache across nodes
    L4_PERSISTENT = "l4_persistent"    # Disk-based persistent cache


class ConsistencyLevel(Enum):
    """Cache consistency levels."""
    EVENTUAL = "eventual"
    STRONG = "strong"
    CAUSAL = "causal"


@dataclass
class CacheNode:
    """Distributed cache node information."""
    node_id: str
    host: str
    port: int
    is_healthy: bool = True
    last_heartbeat: float = 0.0
    cache_size_bytes: int = 0
    hit_rate: float = 0.0


class DistributedNeuralCache:
    """
    Multi-tier distributed cache system optimized for neural data.
    
    Architecture:
    L1: Local in-memory cache (fastest, smallest)
    L2: Shared cache among local workers  
    L3: Distributed cache across nodes
    L4: Persistent cache for model storage
    """
    
    def __init__(self,
                 l1_size_mb: int = 100,
                 l2_size_mb: int = 500,
                 l3_size_mb: int = 2000,
                 consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
                 enable_compression: bool = True):
        
        self.consistency_level = consistency_level
        self.enable_compression = enable_compression
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache levels
        self.l1_cache = CacheManager(
            max_size_bytes=l1_size_mb * 1024 * 1024,
            policy=CachePolicy.ADAPTIVE,
            max_entries=1000
        )
        
        self.l2_cache = CacheManager(
            max_size_bytes=l2_size_mb * 1024 * 1024,
            policy=CachePolicy.NEURAL_OPTIMIZED,
            max_entries=10000
        )
        
        # Distributed cache nodes
        self.cache_nodes: Dict[str, CacheNode] = {}
        self.local_node_id = self._generate_node_id()
        
        # Cache coordination
        self._lock = threading.RLock()
        self._heartbeat_thread = None
        self._is_running = False
        
        # Performance optimization
        self.read_preferences = [CacheLevel.L1_LOCAL, CacheLevel.L2_SHARED, CacheLevel.L3_DISTRIBUTED]
        self.write_strategy = "write_through"  # write_through, write_back, write_around
        
        # Cache warming
        self.warming_patterns = {}
        self.access_patterns = {}
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="DistCache")
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        import socket
        hostname = socket.gethostname()
        timestamp = str(int(time.time() * 1000))
        return f"{hostname}_{timestamp}_{hash(hostname + timestamp) % 10000}"
    
    def start(self) -> None:
        """Start the distributed cache system."""
        with self._lock:
            if self._is_running:
                return
            
            self._is_running = True
            
            # Start heartbeat thread for node discovery
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="DistCache-Heartbeat",
                daemon=True
            )
            self._heartbeat_thread.start()
            
            self.logger.info(f"Distributed cache started with node ID: {self.local_node_id}")
    
    def stop(self) -> None:
        """Stop the distributed cache system."""
        with self._lock:
            self._is_running = False
            if self.executor:
                self.executor.shutdown(wait=True)
            self.logger.info("Distributed cache stopped")
    
    def put(self, key: str, value: Any, 
            ttl: Optional[float] = None,
            cache_level: Optional[CacheLevel] = None) -> bool:
        """Store value in distributed cache."""
        try:
            cache_key = self._normalize_key(key)
            
            # Determine cache levels to write to
            if cache_level:
                levels_to_write = [cache_level]
            else:
                levels_to_write = self._determine_write_levels(key, value)
            
            # Compress if enabled
            if self.enable_compression and isinstance(value, np.ndarray):
                value = self._compress_neural_data(value)
            
            success = True
            
            # Write to each level
            for level in levels_to_write:
                if not self._write_to_level(level, cache_key, value, ttl):
                    success = False
                    self.logger.warning(f"Failed to write to cache level {level.value}")
            
            # Update access patterns
            self._record_access(key, "write")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error putting value to cache: {e}")
            return False
    
    def get(self, key: str, 
            preferred_level: Optional[CacheLevel] = None) -> Optional[Any]:
        """Retrieve value from distributed cache."""
        try:
            cache_key = self._normalize_key(key)
            
            # Determine read order
            read_order = [preferred_level] if preferred_level else self.read_preferences
            
            for level in read_order:
                value = self._read_from_level(level, cache_key)
                if value is not None:
                    # Cache hit - promote to higher levels if needed
                    self._promote_to_higher_levels(level, cache_key, value)
                    
                    # Decompress if needed
                    if self.enable_compression and self._is_compressed(value):
                        value = self._decompress_neural_data(value)
                    
                    # Record hit
                    self._record_access(key, "hit", level)
                    return value
            
            # Cache miss
            self._record_access(key, "miss")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting value from cache: {e}")
            return None
    
    def _determine_write_levels(self, key: str, value: Any) -> List[CacheLevel]:
        """Determine which cache levels to write to based on data characteristics."""
        levels = [CacheLevel.L1_LOCAL]  # Always write to L1
        
        # For neural features and models, also write to L2
        if any(pattern in key for pattern in ['features', 'model', 'calibration']):
            levels.append(CacheLevel.L2_SHARED)
        
        # For shared models, write to distributed cache
        if 'model' in key and 'shared' in key:
            levels.append(CacheLevel.L3_DISTRIBUTED)
        
        # For trained models, write to persistent cache
        if 'trained_model' in key:
            levels.append(CacheLevel.L4_PERSISTENT)
        
        return levels
    
    def _write_to_level(self, level: CacheLevel, key: str, value: Any, ttl: Optional[float]) -> bool:
        """Write to specific cache level."""
        try:
            if level == CacheLevel.L1_LOCAL:
                return self.l1_cache.put(key, value, ttl)
            elif level == CacheLevel.L2_SHARED:
                return self.l2_cache.put(key, value, ttl)
            elif level == CacheLevel.L3_DISTRIBUTED:
                return self._write_to_distributed_cache(key, value, ttl)
            elif level == CacheLevel.L4_PERSISTENT:
                return self._write_to_persistent_cache(key, value, ttl)
            else:
                return False
        except Exception as e:
            self.logger.error(f"Error writing to level {level.value}: {e}")
            return False
    
    def _read_from_level(self, level: CacheLevel, key: str) -> Optional[Any]:
        """Read from specific cache level."""
        try:
            if level == CacheLevel.L1_LOCAL:
                return self.l1_cache.get(key)
            elif level == CacheLevel.L2_SHARED:
                return self.l2_cache.get(key)
            elif level == CacheLevel.L3_DISTRIBUTED:
                return self._read_from_distributed_cache(key)
            elif level == CacheLevel.L4_PERSISTENT:
                return self._read_from_persistent_cache(key)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error reading from level {level.value}: {e}")
            return None
    
    def _write_to_distributed_cache(self, key: str, value: Any, ttl: Optional[float]) -> bool:
        """Write to distributed cache across nodes."""
        # Determine target nodes using consistent hashing
        target_nodes = self._select_target_nodes(key, replica_count=2)
        
        success_count = 0
        for node in target_nodes:
            if self._write_to_node(node, key, value, ttl):
                success_count += 1
        
        # Require at least one successful write
        return success_count > 0
    
    def _read_from_distributed_cache(self, key: str) -> Optional[Any]:
        """Read from distributed cache."""
        target_nodes = self._select_target_nodes(key)
        
        for node in target_nodes:
            if node.is_healthy:
                value = self._read_from_node(node, key)
                if value is not None:
                    return value
        
        return None
    
    def _write_to_node(self, node: CacheNode, key: str, value: Any, ttl: Optional[float]) -> bool:
        """Write to specific distributed cache node."""
        try:
            # Serialize the data
            serialized_data = pickle.dumps({
                'key': key,
                'value': value,
                'ttl': ttl,
                'timestamp': time.time(),
                'node_id': self.local_node_id
            })
            
            # In a real implementation, this would make a network call
            # For now, we'll simulate successful write
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing to node {node.node_id}: {e}")
            return False
    
    def _read_from_node(self, node: CacheNode, key: str) -> Optional[Any]:
        """Read from specific distributed cache node."""
        try:
            # In a real implementation, this would make a network call
            # For now, we'll return None (cache miss)
            return None
            
        except Exception as e:
            self.logger.error(f"Error reading from node {node.node_id}: {e}")
            return None
    
    def _write_to_persistent_cache(self, key: str, value: Any, ttl: Optional[float]) -> bool:
        """Write to persistent disk cache."""
        try:
            import os
            cache_dir = "/tmp/bci_persistent_cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            filename = hashlib.md5(key.encode()).hexdigest() + ".pkl"
            filepath = os.path.join(cache_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'value': value,
                    'timestamp': time.time(),
                    'ttl': ttl
                }, f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing to persistent cache: {e}")
            return False
    
    def _read_from_persistent_cache(self, key: str) -> Optional[Any]:
        """Read from persistent disk cache."""
        try:
            import os
            cache_dir = "/tmp/bci_persistent_cache"
            filename = hashlib.md5(key.encode()).hexdigest() + ".pkl"
            filepath = os.path.join(cache_dir, filename)
            
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Check TTL
            if data.get('ttl') and time.time() - data['timestamp'] > data['ttl']:
                os.remove(filepath)
                return None
            
            return data['value']
            
        except Exception as e:
            self.logger.error(f"Error reading from persistent cache: {e}")
            return None
    
    def _select_target_nodes(self, key: str, replica_count: int = 1) -> List[CacheNode]:
        """Select target nodes for key using consistent hashing."""
        if not self.cache_nodes:
            return []
        
        # Simple implementation - in production, use proper consistent hashing
        key_hash = hash(key)
        sorted_nodes = sorted(self.cache_nodes.values(), key=lambda n: hash(n.node_id))
        
        target_nodes = []
        start_index = key_hash % len(sorted_nodes)
        
        for i in range(replica_count):
            node_index = (start_index + i) % len(sorted_nodes)
            node = sorted_nodes[node_index]
            if node.is_healthy:
                target_nodes.append(node)
        
        return target_nodes
    
    def _promote_to_higher_levels(self, hit_level: CacheLevel, key: str, value: Any) -> None:
        """Promote cache entry to higher (faster) levels."""
        level_priority = {
            CacheLevel.L4_PERSISTENT: 0,
            CacheLevel.L3_DISTRIBUTED: 1,
            CacheLevel.L2_SHARED: 2,
            CacheLevel.L1_LOCAL: 3
        }
        
        hit_priority = level_priority.get(hit_level, 0)
        
        # Promote to all higher priority levels
        for level, priority in level_priority.items():
            if priority > hit_priority:
                self._write_to_level(level, key, value, None)
    
    def _compress_neural_data(self, data: np.ndarray) -> bytes:
        """Compress neural data for efficient storage."""
        if not isinstance(data, np.ndarray):
            return pickle.dumps(data)
        
        # Use numpy's compressed format
        import io
        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=data)
        return buffer.getvalue()
    
    def _decompress_neural_data(self, compressed_data: bytes) -> np.ndarray:
        """Decompress neural data."""
        try:
            import io
            buffer = io.BytesIO(compressed_data)
            loaded = np.load(buffer)
            return loaded['data']
        except:
            # Fallback to pickle if not compressed numpy format
            return pickle.loads(compressed_data)
    
    def _is_compressed(self, data: Any) -> bool:
        """Check if data is in compressed format."""
        return isinstance(data, bytes) and len(data) > 0
    
    def _normalize_key(self, key: str) -> str:
        """Normalize cache key."""
        return key.lower().strip().replace(' ', '_')
    
    def _record_access(self, key: str, operation: str, level: Optional[CacheLevel] = None) -> None:
        """Record cache access for analytics."""
        if key not in self.access_patterns:
            self.access_patterns[key] = {
                'hits': 0,
                'misses': 0,
                'writes': 0,
                'last_access': 0,
                'access_count': 0
            }
        
        pattern = self.access_patterns[key]
        pattern['last_access'] = time.time()
        pattern['access_count'] += 1
        
        if operation == 'hit':
            pattern['hits'] += 1
        elif operation == 'miss':
            pattern['misses'] += 1
        elif operation == 'write':
            pattern['writes'] += 1
    
    def _heartbeat_loop(self) -> None:
        """Heartbeat loop for node discovery and health monitoring."""
        while self._is_running:
            try:
                self._send_heartbeat()
                self._check_node_health()
                time.sleep(30)  # Heartbeat every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(60)  # Back off on error
    
    def _send_heartbeat(self) -> None:
        """Send heartbeat to other nodes."""
        # In a real implementation, this would broadcast to other nodes
        pass
    
    def _check_node_health(self) -> None:
        """Check health of cache nodes."""
        current_time = time.time()
        unhealthy_nodes = []
        
        for node_id, node in self.cache_nodes.items():
            if current_time - node.last_heartbeat > 120:  # 2 minutes timeout
                node.is_healthy = False
                unhealthy_nodes.append(node_id)
        
        if unhealthy_nodes:
            self.logger.warning(f"Nodes marked as unhealthy: {unhealthy_nodes}")
    
    def add_cache_node(self, node: CacheNode) -> None:
        """Add a new cache node to the cluster."""
        with self._lock:
            self.cache_nodes[node.node_id] = node
            self.logger.info(f"Added cache node: {node.node_id} at {node.host}:{node.port}")
    
    def remove_cache_node(self, node_id: str) -> None:
        """Remove cache node from the cluster."""
        with self._lock:
            if node_id in self.cache_nodes:
                del self.cache_nodes[node_id]
                self.logger.info(f"Removed cache node: {node_id}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        
        total_hits = l1_stats['hits'] + l2_stats['hits']
        total_misses = l1_stats['misses'] + l2_stats['misses']
        overall_hit_rate = (total_hits / max(total_hits + total_misses, 1)) * 100
        
        return {
            "overall_hit_rate": round(overall_hit_rate, 2),
            "l1_cache": l1_stats,
            "l2_cache": l2_stats,
            "distributed_nodes": len([n for n in self.cache_nodes.values() if n.is_healthy]),
            "total_nodes": len(self.cache_nodes),
            "access_patterns_count": len(self.access_patterns),
            "compression_enabled": self.enable_compression,
            "consistency_level": self.consistency_level.value
        }
    
    def warm_cache(self, patterns: Dict[str, Any]) -> None:
        """Warm cache with frequently accessed data patterns."""
        self.warming_patterns.update(patterns)
        
        # Implement cache warming logic
        for pattern, data in patterns.items():
            self.put(f"warmed_{pattern}", data, ttl=3600)  # 1 hour TTL
        
        self.logger.info(f"Cache warmed with {len(patterns)} patterns")