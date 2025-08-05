"""
Advanced caching system for BCI data with intelligent cache policies.
"""

import time
import hashlib
import pickle
import threading
import asyncio
from typing import Any, Dict, Optional, Callable, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import OrderedDict
import weakref
import numpy as np


class CachePolicy(Enum):
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # Adaptive based on access patterns
    NEURAL_OPTIMIZED = "neural"    # Optimized for neural data patterns


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at
    
    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheStats:
    """Statistics for cache performance monitoring."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size_bytes = 0
        self.entry_count = 0
        self.avg_access_time = 0.0
        self.hit_rate = 0.0
        
        self._lock = threading.Lock()
    
    def record_hit(self, access_time: float = 0.0) -> None:
        with self._lock:
            self.hits += 1
            self._update_hit_rate()
            if access_time > 0:
                self._update_avg_access_time(access_time)
    
    def record_miss(self) -> None:
        with self._lock:
            self.misses += 1
            self._update_hit_rate()
    
    def record_eviction(self) -> None:
        with self._lock:
            self.evictions += 1
    
    def _update_hit_rate(self) -> None:
        total = self.hits + self.misses
        self.hit_rate = (self.hits / total) if total > 0 else 0.0
    
    def _update_avg_access_time(self, new_time: float) -> None:
        if self.hits == 1:
            self.avg_access_time = new_time
        else:
            self.avg_access_time = (self.avg_access_time * (self.hits - 1) + new_time) / self.hits
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": round(self.hit_rate * 100, 2),
                "size_bytes": self.size_bytes,
                "entry_count": self.entry_count,
                "avg_access_time_ms": round(self.avg_access_time * 1000, 2)
            }


class CacheManager:
    """
    High-performance cache manager with multiple eviction policies.
    """
    
    def __init__(self, max_size_bytes: int = 100 * 1024 * 1024,  # 100MB
                 max_entries: int = 1000, policy: CachePolicy = CachePolicy.LRU,
                 default_ttl: Optional[float] = None):
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.policy = policy
        self.default_ttl = default_ttl
        
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()  # For LRU
        self.frequency_count: Dict[str, int] = {}  # For LFU
        self.stats = CacheStats()
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Adaptive policy learning
        self.access_patterns: Dict[str, List[float]] = {}
        self.pattern_weights: Dict[str, float] = {}
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of cached value."""
        try:
            if isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (str, bytes)):
                return len(value)
            else:
                # Fallback to pickle size estimation
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 1024  # Default estimate
    
    def _generate_key(self, key_parts: Union[str, Tuple]) -> str:
        """Generate cache key from parts."""
        if isinstance(key_parts, str):
            return key_parts
        
        # Create hash for complex key parts
        key_str = str(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _should_evict(self) -> bool:
        """Check if eviction is needed."""
        return (len(self.cache) >= self.max_entries or 
                self.stats.size_bytes >= self.max_size_bytes)
    
    def _evict_entry(self) -> Optional[str]:
        """Evict entry based on policy."""
        if not self.cache:
            return None
        
        evict_key = None
        
        if self.policy == CachePolicy.LRU:
            # Evict least recently used
            evict_key = next(iter(self.access_order))
        
        elif self.policy == CachePolicy.LFU:
            # Evict least frequently used
            min_freq = min(self.frequency_count.values())
            for key, freq in self.frequency_count.items():
                if freq == min_freq and key in self.cache:
                    evict_key = key
                    break
        
        elif self.policy == CachePolicy.TTL:
            # Evict expired entries first, then oldest
            current_time = time.time()
            expired_keys = [k for k, v in self.cache.items() if v.is_expired]
            if expired_keys:
                evict_key = expired_keys[0]
            else:
                # Evict oldest if no expired entries
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
                evict_key = oldest_key
        
        elif self.policy == CachePolicy.ADAPTIVE:
            # Use learning algorithm to decide
            evict_key = self._adaptive_eviction()
        
        elif self.policy == CachePolicy.NEURAL_OPTIMIZED:
            # Specialized for neural data patterns
            evict_key = self._neural_optimized_eviction()
        
        if evict_key and evict_key in self.cache:
            self._remove_entry(evict_key)
            self.stats.record_eviction()
            return evict_key
        
        return None
    
    def _adaptive_eviction(self) -> Optional[str]:
        """Adaptive eviction based on learned access patterns."""
        if not self.cache:
            return None
        
        # Score entries based on multiple factors
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Recency score (higher = more recent)
            recency_score = 1.0 / (current_time - entry.last_accessed + 1)
            
            # Frequency score
            frequency_score = entry.access_count / max(entry.age_seconds, 1)
            
            # Size penalty (larger entries get lower scores)
            size_penalty = 1.0 / (entry.size_bytes / 1024 + 1)  # KB normalization
            
            # Pattern weight (learned importance)
            pattern_weight = self.pattern_weights.get(key, 1.0)
            
            # Combined score (lower = more likely to evict)
            scores[key] = (recency_score * 0.4 + frequency_score * 0.3 + 
                          size_penalty * 0.2 + pattern_weight * 0.1)
        
        # Return key with lowest score
        return min(scores, key=scores.get)
    
    def _neural_optimized_eviction(self) -> Optional[str]:
        """Eviction optimized for neural data access patterns."""
        if not self.cache:
            return None
        
        current_time = time.time()
        neural_scores = {}
        
        for key, entry in self.cache.items():
            # Neural data typically accessed in temporal windows
            # Recent data is more valuable
            temporal_score = np.exp(-(current_time - entry.last_accessed) / 300)  # 5-minute decay
            
            # Calibration data and models are very valuable
            is_model_data = 'model' in key.lower() or 'calibration' in key.lower()
            model_bonus = 2.0 if is_model_data else 1.0
            
            # Raw neural data can be re-computed if needed
            is_raw_data = 'raw' in key.lower() or 'neural_data' in key.lower()
            raw_penalty = 0.5 if is_raw_data else 1.0
            
            # Feature data is intermediate value
            is_features = 'features' in key.lower() or 'extracted' in key.lower()
            feature_bonus = 1.5 if is_features else 1.0
            
            neural_scores[key] = temporal_score * model_bonus * raw_penalty * feature_bonus
        
        # Return key with lowest neural score
        return min(neural_scores, key=neural_scores.get)
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from all tracking structures."""
        if key in self.cache:
            entry = self.cache[key]
            self.stats.size_bytes -= entry.size_bytes
            self.stats.entry_count -= 1
            del self.cache[key]
        
        if key in self.access_order:
            del self.access_order[key]
        
        if key in self.frequency_count:
            del self.frequency_count[key]
    
    def put(self, key: Union[str, Tuple], value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache."""
        start_time = time.time()
        
        with self._lock:
            cache_key = self._generate_key(key)
            
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Check if single item is too large
            if size_bytes > self.max_size_bytes:
                self.logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return False
            
            # Evict entries if needed
            while self._should_evict():
                evicted = self._evict_entry()
                if not evicted:
                    break
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=start_time,
                last_accessed=start_time,
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl
            )
            
            # Store entry
            if cache_key in self.cache:
                # Update existing entry
                old_entry = self.cache[cache_key]
                self.stats.size_bytes -= old_entry.size_bytes
            else:
                self.stats.entry_count += 1
            
            self.cache[cache_key] = entry
            self.stats.size_bytes += size_bytes
            
            # Update tracking structures
            self.access_order[cache_key] = True
            self.frequency_count[cache_key] = self.frequency_count.get(cache_key, 0) + 1
            
            # Learn access patterns for adaptive policy
            if self.policy == CachePolicy.ADAPTIVE:
                if cache_key not in self.access_patterns:
                    self.access_patterns[cache_key] = []
                self.access_patterns[cache_key].append(start_time)
                # Keep only recent access times
                self.access_patterns[cache_key] = self.access_patterns[cache_key][-100:]
            
            return True
    
    def get(self, key: Union[str, Tuple]) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.time()
        
        with self._lock:
            cache_key = self._generate_key(key)
            
            if cache_key not in self.cache:
                self.stats.record_miss()
                return None
            
            entry = self.cache[cache_key]
            
            # Check if expired
            if entry.is_expired:
                self._remove_entry(cache_key)
                self.stats.record_miss()
                return None
            
            # Update access tracking
            entry.touch()
            
            # Move to end for LRU
            if cache_key in self.access_order:
                del self.access_order[cache_key]
            self.access_order[cache_key] = True
            
            # Update frequency
            self.frequency_count[cache_key] = self.frequency_count.get(cache_key, 0) + 1
            
            access_time = time.time() - start_time
            self.stats.record_hit(access_time)
            
            return entry.value
    
    def delete(self, key: Union[str, Tuple]) -> bool:
        """Delete entry from cache."""
        with self._lock:
            cache_key = self._generate_key(key)
            if cache_key in self.cache:
                self._remove_entry(cache_key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.frequency_count.clear()
            self.access_patterns.clear()
            self.pattern_weights.clear()
            self.stats = CacheStats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats = self.stats.get_stats()
            stats.update({
                "policy": self.policy.value,
                "max_size_bytes": self.max_size_bytes,
                "max_entries": self.max_entries,
                "memory_utilization_pct": round((self.stats.size_bytes / self.max_size_bytes) * 100, 2),
                "entry_utilization_pct": round((self.stats.entry_count / self.max_entries) * 100, 2)
            })
            return stats


class NeuralDataCache(CacheManager):
    """
    Specialized cache for neural data with domain-specific optimizations.
    """
    
    def __init__(self, max_size_bytes: int = 500 * 1024 * 1024):  # 500MB for neural data
        super().__init__(
            max_size_bytes=max_size_bytes,
            max_entries=10000,
            policy=CachePolicy.NEURAL_OPTIMIZED,
            default_ttl=3600.0  # 1 hour default TTL
        )
        
        # Neural-specific configurations
        self.feature_cache_ttl = 1800.0  # 30 minutes for features
        self.model_cache_ttl = 7200.0    # 2 hours for models
        self.raw_data_ttl = 300.0        # 5 minutes for raw data
    
    def cache_neural_features(self, subject_id: str, paradigm: str, 
                            features: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """Cache extracted neural features."""
        key = f"features_{subject_id}_{paradigm}_{int(time.time())}"
        return self.put(key, {
            'features': features,
            'metadata': metadata or {},
            'paradigm': paradigm,
            'subject_id': subject_id
        }, ttl=self.feature_cache_ttl)
    
    def cache_decoder_model(self, model_id: str, model_data: Any, 
                          paradigm: str, subject_id: str = None) -> bool:
        """Cache trained decoder model."""
        key = f"model_{model_id}_{paradigm}_{subject_id or 'general'}"
        return self.put(key, model_data, ttl=self.model_cache_ttl)
    
    def cache_raw_neural_data(self, session_id: str, data: np.ndarray, 
                            sampling_rate: int, channels: List[str]) -> bool:
        """Cache raw neural data (shorter TTL)."""
        key = f"raw_{session_id}_{int(time.time())}"
        return self.put(key, {
            'data': data,
            'sampling_rate': sampling_rate,
            'channels': channels,
            'session_id': session_id
        }, ttl=self.raw_data_ttl)
    
    def get_neural_features(self, subject_id: str, paradigm: str, 
                          max_age_seconds: float = 1800) -> Optional[Dict[str, Any]]:
        """Get cached neural features within age limit."""
        current_time = time.time()
        
        # Search for matching features
        with self._lock:
            for key, entry in self.cache.items():
                if (key.startswith(f"features_{subject_id}_{paradigm}") and
                    current_time - entry.created_at <= max_age_seconds):
                    return self.get(key)
        
        return None
    
    def get_decoder_model(self, model_id: str, paradigm: str, 
                        subject_id: str = None) -> Optional[Any]:
        """Get cached decoder model."""
        key = f"model_{model_id}_{paradigm}_{subject_id or 'general'}"
        return self.get(key)
    
    def cleanup_expired_neural_data(self) -> int:
        """Clean up expired neural data and return count of removed entries."""
        removed_count = 0
        current_time = time.time()
        
        with self._lock:
            keys_to_remove = []
            for key, entry in self.cache.items():
                # More aggressive cleanup for raw data
                if (key.startswith('raw_') and 
                    current_time - entry.created_at > self.raw_data_ttl):
                    keys_to_remove.append(key)
                elif entry.is_expired:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
                removed_count += 1
        
        return removed_count
    
    def get_neural_cache_summary(self) -> Dict[str, Any]:
        """Get summary of neural data cache contents."""
        with self._lock:
            summary = {
                'total_entries': len(self.cache),
                'raw_data_entries': 0,
                'feature_entries': 0,
                'model_entries': 0,
                'other_entries': 0,
                'oldest_entry_age': 0,
                'newest_entry_age': 0
            }
            
            if self.cache:
                current_time = time.time()
                ages = []
                
                for key, entry in self.cache.items():
                    age = current_time - entry.created_at
                    ages.append(age)
                    
                    if key.startswith('raw_'):
                        summary['raw_data_entries'] += 1
                    elif key.startswith('features_'):
                        summary['feature_entries'] += 1
                    elif key.startswith('model_'):
                        summary['model_entries'] += 1
                    else:
                        summary['other_entries'] += 1
                
                summary['oldest_entry_age'] = max(ages)
                summary['newest_entry_age'] = min(ages)
            
            return summary


# Factory function for creating domain-specific caches
def create_cache_manager(cache_type: str, **kwargs) -> CacheManager:
    """Factory function to create appropriate cache manager."""
    
    if cache_type == "neural":
        return NeuralDataCache(**kwargs)
    elif cache_type == "general":
        return CacheManager(policy=CachePolicy.LRU, **kwargs)
    elif cache_type == "adaptive":
        return CacheManager(policy=CachePolicy.ADAPTIVE, **kwargs)
    elif cache_type == "ttl":
        return CacheManager(policy=CachePolicy.TTL, **kwargs)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test neural data cache
    neural_cache = NeuralDataCache()
    
    # Cache some test data
    test_features = np.random.randn(64, 250)  # 64 channels, 250 samples
    neural_cache.cache_neural_features("subject_001", "P300", test_features)
    
    # Retrieve data
    cached_features = neural_cache.get_neural_features("subject_001", "P300")
    print(f"Retrieved features shape: {cached_features['features'].shape if cached_features else 'None'}")
    
    # Print cache stats
    print("Cache statistics:")
    stats = neural_cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")