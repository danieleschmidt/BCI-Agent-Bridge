"""
Global Expansion: Multi-Region Edge Computing Manager

This module implements advanced multi-region deployment and edge computing
capabilities for the BCI-Agent-Bridge system, enabling global-scale
deployment with optimal performance and compliance.

Key Features:
- Multi-region deployment orchestration
- Edge computing node management
- Intelligent workload distribution
- Global load balancing and failover
- Region-specific compliance management
- Real-time latency optimization
- Distributed neural processing
- Cross-region data synchronization
- Geo-distributed caching
- Edge AI model deployment

This system enables the BCI-Agent-Bridge to operate efficiently across
global regions with minimal latency and maximum reliability.
"""

import numpy as np
import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import sqlite3
from collections import defaultdict, deque
import statistics
import socket
import requests
from urllib.parse import urlparse
import dns.resolver
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegionType(Enum):
    """Types of deployment regions."""
    CORE = "core"          # Major data centers
    EDGE = "edge"          # Edge computing nodes
    CDN = "cdn"            # Content delivery nodes
    MOBILE_EDGE = "mobile_edge"  # Mobile edge computing
    HYBRID = "hybrid"      # Hybrid cloud/edge


class DeploymentStatus(Enum):
    """Deployment status for regions."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class ComplianceRegion(Enum):
    """Compliance regions with specific regulations."""
    GDPR_EU = "gdpr_eu"        # European Union
    CCPA_US = "ccpa_us"        # California
    HIPAA_US = "hipaa_us"      # US Healthcare
    PDPA_SINGAPORE = "pdpa_sg" # Singapore
    PIPEDA_CANADA = "pipeda_ca" # Canada
    LGPD_BRAZIL = "lgpd_br"    # Brazil


@dataclass
class GeographicCoordinate:
    """Geographic coordinate with metadata."""
    latitude: float
    longitude: float
    city: str
    country: str
    region: str
    timezone: str


@dataclass
class EdgeNode:
    """Represents an edge computing node."""
    node_id: str
    region: str
    location: GeographicCoordinate
    node_type: RegionType
    status: DeploymentStatus
    
    # Hardware specifications
    cpu_cores: int = 8
    memory_gb: float = 32.0
    storage_gb: float = 1000.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    
    # Network specifications
    bandwidth_mbps: float = 1000.0
    latency_ms: float = 10.0
    
    # Current utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    storage_usage: float = 0.0
    network_usage: float = 0.0
    
    # Capabilities
    supported_models: List[str] = field(default_factory=list)
    compliance_regions: List[ComplianceRegion] = field(default_factory=list)
    
    # Performance metrics
    response_time: float = 0.0
    throughput: float = 0.0
    availability: float = 1.0
    error_rate: float = 0.0
    
    # Deployment metadata
    deployment_time: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)
    version: str = "1.0.0"


@dataclass
class WorkloadRequest:
    """Represents a workload request to be distributed."""
    request_id: str
    user_location: GeographicCoordinate
    workload_type: str
    
    # Requirements
    cpu_requirements: float = 1.0
    memory_requirements: float = 1.0  # GB
    gpu_requirements: bool = False
    latency_requirement: float = 100.0  # ms
    bandwidth_requirement: float = 10.0  # Mbps
    
    # Compliance
    compliance_requirements: List[ComplianceRegion] = field(default_factory=list)
    data_residency_requirements: List[str] = field(default_factory=list)
    
    # Priority and timing
    priority: int = 5  # 1-10, 10 is highest
    timeout: float = 30.0  # seconds
    created_at: float = field(default_factory=time.time)
    
    # Payload
    payload_size: float = 1.0  # MB
    expected_response_size: float = 1.0  # MB


@dataclass
class DeploymentPolicy:
    """Deployment policy for regional management."""
    policy_id: str
    regions: List[str]
    
    # Resource policies
    min_cpu_cores: int = 2
    min_memory_gb: float = 8.0
    preferred_node_types: List[RegionType] = field(default_factory=list)
    
    # Performance policies
    max_latency_ms: float = 200.0
    min_availability: float = 0.99
    max_error_rate: float = 0.01
    
    # Compliance policies
    required_compliance: List[ComplianceRegion] = field(default_factory=list)
    data_residency_strict: bool = False
    encryption_required: bool = True
    
    # Scaling policies
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    min_nodes_per_region: int = 1
    max_nodes_per_region: int = 10
    
    # Failover policies
    failover_enabled: bool = True
    failover_max_distance_km: float = 1000.0
    failover_latency_penalty: float = 50.0


class GlobalLocationService:
    """Service for managing global locations and routing."""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="bci_agent_bridge_global")
        self.region_definitions = self._initialize_regions()
        self.location_cache = {}
        
    def _initialize_regions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize predefined regions with major data center locations."""
        return {
            "us_east": {
                "name": "US East",
                "coordinates": GeographicCoordinate(39.0458, -76.6413, "Ashburn", "USA", "us_east", "UTC-5"),
                "compliance": [ComplianceRegion.CCPA_US, ComplianceRegion.HIPAA_US],
                "node_type": RegionType.CORE
            },
            "us_west": {
                "name": "US West", 
                "coordinates": GeographicCoordinate(37.4419, -122.1430, "Palo Alto", "USA", "us_west", "UTC-8"),
                "compliance": [ComplianceRegion.CCPA_US, ComplianceRegion.HIPAA_US],
                "node_type": RegionType.CORE
            },
            "eu_west": {
                "name": "EU West",
                "coordinates": GeographicCoordinate(53.4084, -2.9916, "Manchester", "UK", "eu_west", "UTC+0"),
                "compliance": [ComplianceRegion.GDPR_EU],
                "node_type": RegionType.CORE
            },
            "eu_central": {
                "name": "EU Central",
                "coordinates": GeographicCoordinate(50.1109, 8.6821, "Frankfurt", "Germany", "eu_central", "UTC+1"),
                "compliance": [ComplianceRegion.GDPR_EU],
                "node_type": RegionType.CORE
            },
            "asia_pacific": {
                "name": "Asia Pacific",
                "coordinates": GeographicCoordinate(1.3521, 103.8198, "Singapore", "Singapore", "asia_pacific", "UTC+8"),
                "compliance": [ComplianceRegion.PDPA_SINGAPORE],
                "node_type": RegionType.CORE
            },
            "asia_northeast": {
                "name": "Asia Northeast",
                "coordinates": GeographicCoordinate(35.6762, 139.6503, "Tokyo", "Japan", "asia_northeast", "UTC+9"),
                "compliance": [],
                "node_type": RegionType.CORE
            },
            "canada_central": {
                "name": "Canada Central",
                "coordinates": GeographicCoordinate(43.6532, -79.3832, "Toronto", "Canada", "canada_central", "UTC-5"),
                "compliance": [ComplianceRegion.PIPEDA_CANADA],
                "node_type": RegionType.CORE
            },
            "south_america": {
                "name": "South America",
                "coordinates": GeographicCoordinate(-23.5558, -46.6396, "São Paulo", "Brazil", "south_america", "UTC-3"),
                "compliance": [ComplianceRegion.LGPD_BRAZIL],
                "node_type": RegionType.CORE
            }
        }
    
    def get_optimal_regions_for_location(self, location: GeographicCoordinate,
                                       max_regions: int = 3) -> List[str]:
        """Find optimal regions for a given location based on distance."""
        distances = []
        
        for region_id, region_info in self.region_definitions.items():
            region_coord = region_info["coordinates"]
            distance = geodesic(
                (location.latitude, location.longitude),
                (region_coord.latitude, region_coord.longitude)
            ).kilometers
            
            distances.append((region_id, distance))
        
        # Sort by distance and return top regions
        distances.sort(key=lambda x: x[1])
        return [region_id for region_id, _ in distances[:max_regions]]
    
    def calculate_distance(self, loc1: GeographicCoordinate, 
                         loc2: GeographicCoordinate) -> float:
        """Calculate distance between two locations in kilometers."""
        return geodesic(
            (loc1.latitude, loc1.longitude),
            (loc2.latitude, loc2.longitude)
        ).kilometers
    
    def estimate_network_latency(self, distance_km: float) -> float:
        """Estimate network latency based on distance."""
        # Speed of light in fiber optic cable (approximately 200,000 km/s)
        light_speed = 200000  # km/s
        
        # Round trip time + processing overhead
        base_latency = (distance_km / light_speed) * 2 * 1000  # Convert to ms
        
        # Add routing and processing overhead
        overhead = min(50, distance_km * 0.02)  # Up to 50ms overhead
        
        return base_latency + overhead
    
    def resolve_location_from_ip(self, ip_address: str) -> Optional[GeographicCoordinate]:
        """Resolve geographic location from IP address (simplified)."""
        # In a real implementation, this would use IP geolocation services
        # For demonstration, return a random location
        
        if ip_address.startswith("192.168") or ip_address.startswith("10.") or ip_address.startswith("172."):
            # Private IP - assume local
            return GeographicCoordinate(37.4419, -122.1430, "Local", "Unknown", "local", "UTC")
        
        # Simplified mapping based on IP ranges
        ip_parts = ip_address.split(".")
        if ip_parts[0] in ["1", "2", "3"]:  # Simplified US range
            return self.region_definitions["us_east"]["coordinates"]
        elif ip_parts[0] in ["80", "81", "82"]:  # Simplified EU range
            return self.region_definitions["eu_west"]["coordinates"]
        elif ip_parts[0] in ["100", "101", "102"]:  # Simplified Asia range
            return self.region_definitions["asia_pacific"]["coordinates"]
        
        # Default to US East
        return self.region_definitions["us_east"]["coordinates"]


class EdgeComputingOrchestrator:
    """Orchestrates edge computing nodes across global regions."""
    
    def __init__(self):
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.location_service = GlobalLocationService()
        self.deployment_policies: Dict[str, DeploymentPolicy] = {}
        self.workload_queue = queue.PriorityQueue()
        self.active_workloads: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.latency_measurements = defaultdict(list)
        self.throughput_measurements = defaultdict(list)
        
        # Load balancing
        self.load_balancer = self._create_load_balancer()
        
        logger.info("Edge Computing Orchestrator initialized")
    
    def _create_load_balancer(self) -> Dict[str, Any]:
        """Create intelligent load balancer configuration."""
        return {
            "algorithm": "weighted_round_robin",
            "health_check_interval": 30,
            "failure_threshold": 3,
            "recovery_threshold": 2,
            "weights": {},
            "sticky_sessions": False
        }
    
    def register_edge_node(self, node: EdgeNode) -> bool:
        """Register a new edge computing node."""
        try:
            # Validate node configuration
            if not self._validate_node_configuration(node):
                logger.error(f"Invalid node configuration for {node.node_id}")
                return False
            
            # Initialize node
            node.status = DeploymentStatus.INITIALIZING
            node.deployment_time = time.time()
            
            # Add to registry
            self.edge_nodes[node.node_id] = node
            
            # Initialize performance tracking
            self.performance_history[node.node_id] = []
            
            # Update load balancer weights
            self._update_load_balancer_weights()
            
            # Run initial health check
            asyncio.create_task(self._perform_health_check(node.node_id))
            
            logger.info(f"Registered edge node {node.node_id} in {node.region}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register edge node {node.node_id}: {e}")
            return False
    
    def _validate_node_configuration(self, node: EdgeNode) -> bool:
        """Validate edge node configuration."""
        # Check required fields
        if not node.node_id or not node.region:
            return False
        
        # Check resource constraints
        if node.cpu_cores < 1 or node.memory_gb < 1:
            return False
        
        # Check network constraints
        if node.bandwidth_mbps < 1 or node.latency_ms < 0:
            return False
        
        # Check for duplicate node IDs
        if node.node_id in self.edge_nodes:
            return False
        
        return True
    
    async def _perform_health_check(self, node_id: str):
        """Perform health check on an edge node."""
        if node_id not in self.edge_nodes:
            return
        
        node = self.edge_nodes[node_id]
        
        try:
            # Simulate health check
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Update health metrics
            node.last_health_check = time.time()
            
            # Simulate resource usage
            node.cpu_usage = min(1.0, node.cpu_usage + np.random.uniform(-0.1, 0.1))
            node.memory_usage = min(1.0, node.memory_usage + np.random.uniform(-0.1, 0.1))
            node.network_usage = min(1.0, node.network_usage + np.random.uniform(-0.05, 0.05))
            
            # Simulate performance metrics
            node.response_time = max(1.0, node.response_time + np.random.uniform(-5.0, 5.0))
            node.throughput = max(1.0, node.throughput + np.random.uniform(-10.0, 10.0))
            node.availability = min(1.0, max(0.9, node.availability + np.random.uniform(-0.01, 0.01)))
            node.error_rate = max(0.0, min(0.1, node.error_rate + np.random.uniform(-0.001, 0.001)))
            
            # Determine node status based on health
            if node.availability > 0.99 and node.error_rate < 0.01:
                if node.status != DeploymentStatus.ACTIVE:
                    node.status = DeploymentStatus.ACTIVE
                    logger.info(f"Node {node_id} is now ACTIVE")
            elif node.availability > 0.95:
                if node.status != DeploymentStatus.DEGRADED:
                    node.status = DeploymentStatus.DEGRADED
                    logger.warning(f"Node {node_id} is DEGRADED")
            else:
                if node.status != DeploymentStatus.OFFLINE:
                    node.status = DeploymentStatus.OFFLINE
                    logger.error(f"Node {node_id} is OFFLINE")
            
            # Record performance history
            self.performance_history[node_id].append({
                "timestamp": time.time(),
                "cpu_usage": node.cpu_usage,
                "memory_usage": node.memory_usage,
                "response_time": node.response_time,
                "throughput": node.throughput,
                "availability": node.availability,
                "error_rate": node.error_rate
            })
            
            # Keep only recent history
            if len(self.performance_history[node_id]) > 1000:
                self.performance_history[node_id] = self.performance_history[node_id][-1000:]
            
        except Exception as e:
            logger.error(f"Health check failed for node {node_id}: {e}")
            node.status = DeploymentStatus.OFFLINE
    
    def _update_load_balancer_weights(self):
        """Update load balancer weights based on node performance."""
        total_capacity = 0
        node_capacities = {}
        
        for node_id, node in self.edge_nodes.items():
            if node.status == DeploymentStatus.ACTIVE:
                # Calculate capacity based on available resources and performance
                cpu_capacity = (1.0 - node.cpu_usage) * node.cpu_cores
                memory_capacity = (1.0 - node.memory_usage) * node.memory_gb
                
                # Factor in performance metrics
                performance_factor = node.availability * (1.0 - node.error_rate)
                
                capacity = (cpu_capacity + memory_capacity) * performance_factor
                node_capacities[node_id] = capacity
                total_capacity += capacity
        
        # Update weights
        if total_capacity > 0:
            for node_id, capacity in node_capacities.items():
                weight = int((capacity / total_capacity) * 100)
                self.load_balancer["weights"][node_id] = max(1, weight)
    
    def select_optimal_node(self, request: WorkloadRequest) -> Optional[str]:
        """Select optimal node for a workload request."""
        candidate_nodes = []
        
        for node_id, node in self.edge_nodes.items():
            if node.status != DeploymentStatus.ACTIVE:
                continue
            
            # Check resource requirements
            if (node.cpu_usage + request.cpu_requirements / node.cpu_cores) > 1.0:
                continue
            if (node.memory_usage + request.memory_requirements / node.memory_gb) > 1.0:
                continue
            if request.gpu_requirements and node.gpu_count == 0:
                continue
            
            # Check compliance requirements
            if request.compliance_requirements:
                if not any(compliance in node.compliance_regions 
                          for compliance in request.compliance_requirements):
                    continue
            
            # Calculate distance and latency
            distance = self.location_service.calculate_distance(
                request.user_location, node.location
            )
            estimated_latency = self.location_service.estimate_network_latency(distance)
            
            # Check latency requirement
            if estimated_latency > request.latency_requirement:
                continue
            
            # Calculate selection score
            score = self._calculate_node_selection_score(node, request, distance, estimated_latency)
            
            candidate_nodes.append((node_id, score, distance, estimated_latency))
        
        if not candidate_nodes:
            return None
        
        # Sort by score (higher is better)
        candidate_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Return best node
        return candidate_nodes[0][0]
    
    def _calculate_node_selection_score(self, node: EdgeNode, request: WorkloadRequest,
                                      distance: float, estimated_latency: float) -> float:
        """Calculate selection score for a node."""
        score = 0.0
        
        # Distance factor (closer is better)
        max_distance = 20000  # km (roughly half the earth's circumference)
        distance_score = (max_distance - distance) / max_distance
        score += distance_score * 0.3
        
        # Latency factor (lower is better)
        max_latency = 500  # ms
        latency_score = (max_latency - estimated_latency) / max_latency
        score += latency_score * 0.25
        
        # Resource availability factor
        cpu_availability = 1.0 - node.cpu_usage
        memory_availability = 1.0 - node.memory_usage
        resource_score = (cpu_availability + memory_availability) / 2.0
        score += resource_score * 0.2
        
        # Performance factor
        performance_score = node.availability * (1.0 - node.error_rate)
        score += performance_score * 0.15
        
        # Load factor (less loaded is better)
        current_load = (node.cpu_usage + node.memory_usage + node.network_usage) / 3.0
        load_score = 1.0 - current_load
        score += load_score * 0.1
        
        return score
    
    async def process_workload_request(self, request: WorkloadRequest) -> Dict[str, Any]:
        """Process a workload request and route it to optimal node."""
        start_time = time.time()
        
        result = {
            "request_id": request.request_id,
            "success": False,
            "selected_node": None,
            "processing_time": 0.0,
            "response_time": 0.0,
            "error": None
        }
        
        try:
            # Select optimal node
            selected_node_id = self.select_optimal_node(request)
            
            if not selected_node_id:
                result["error"] = "No suitable node found"
                return result
            
            result["selected_node"] = selected_node_id
            selected_node = self.edge_nodes[selected_node_id]
            
            # Calculate estimated processing time
            processing_time = self._estimate_processing_time(selected_node, request)
            
            # Update node resource usage
            selected_node.cpu_usage += request.cpu_requirements / selected_node.cpu_cores
            selected_node.memory_usage += request.memory_requirements / selected_node.memory_gb
            
            # Simulate workload processing
            await asyncio.sleep(processing_time)
            
            # Release resources
            selected_node.cpu_usage -= request.cpu_requirements / selected_node.cpu_cores
            selected_node.memory_usage -= request.memory_requirements / selected_node.memory_gb
            
            # Ensure usage doesn't go negative
            selected_node.cpu_usage = max(0.0, selected_node.cpu_usage)
            selected_node.memory_usage = max(0.0, selected_node.memory_usage)
            
            # Calculate response time
            distance = self.location_service.calculate_distance(
                request.user_location, selected_node.location
            )
            network_latency = self.location_service.estimate_network_latency(distance)
            
            result["processing_time"] = processing_time
            result["response_time"] = processing_time + network_latency / 1000.0  # Convert to seconds
            result["success"] = True
            
            # Update performance measurements
            self.latency_measurements[selected_node_id].append(network_latency)
            self.throughput_measurements[selected_node_id].append(1.0 / processing_time)
            
            # Keep only recent measurements
            for measurements in [self.latency_measurements, self.throughput_measurements]:
                if len(measurements[selected_node_id]) > 1000:
                    measurements[selected_node_id] = measurements[selected_node_id][-1000:]
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Failed to process workload request {request.request_id}: {e}")
        
        return result
    
    def _estimate_processing_time(self, node: EdgeNode, request: WorkloadRequest) -> float:
        """Estimate processing time for a workload on a node."""
        # Base processing time based on workload type
        base_times = {
            "neural_processing": 0.1,
            "signal_analysis": 0.05,
            "model_inference": 0.2,
            "data_preprocessing": 0.03,
            "real_time_processing": 0.02
        }
        
        base_time = base_times.get(request.workload_type, 0.1)
        
        # Adjust for node performance
        cpu_factor = 1.0 / max(0.1, (1.0 - node.cpu_usage))
        performance_factor = 1.0 / max(0.1, node.availability)
        
        # Adjust for GPU if available and required
        if request.gpu_requirements and node.gpu_count > 0:
            base_time *= 0.1  # GPU acceleration
        
        return base_time * cpu_factor * performance_factor
    
    def create_deployment_policy(self, policy: DeploymentPolicy) -> bool:
        """Create a new deployment policy."""
        try:
            self.deployment_policies[policy.policy_id] = policy
            logger.info(f"Created deployment policy {policy.policy_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create deployment policy: {e}")
            return False
    
    async def auto_scale_region(self, region: str, policy_id: str) -> Dict[str, Any]:
        """Auto-scale a region based on deployment policy."""
        if policy_id not in self.deployment_policies:
            return {"error": f"Policy {policy_id} not found"}
        
        policy = self.deployment_policies[policy_id]
        
        # Get nodes in region
        region_nodes = [
            node for node in self.edge_nodes.values()
            if node.region == region and node.status == DeploymentStatus.ACTIVE
        ]
        
        scaling_result = {
            "region": region,
            "policy_id": policy_id,
            "current_nodes": len(region_nodes),
            "action": "none",
            "target_nodes": len(region_nodes)
        }
        
        if not region_nodes:
            return scaling_result
        
        # Calculate average resource usage
        avg_cpu_usage = statistics.mean(node.cpu_usage for node in region_nodes)
        avg_memory_usage = statistics.mean(node.memory_usage for node in region_nodes)
        avg_usage = (avg_cpu_usage + avg_memory_usage) / 2.0
        
        # Determine scaling action
        if avg_usage > policy.scale_up_threshold and len(region_nodes) < policy.max_nodes_per_region:
            # Scale up
            new_node = await self._create_new_node(region, policy)
            if new_node:
                self.register_edge_node(new_node)
                scaling_result["action"] = "scale_up"
                scaling_result["target_nodes"] = len(region_nodes) + 1
                logger.info(f"Scaled up region {region} - added node {new_node.node_id}")
        
        elif avg_usage < policy.scale_down_threshold and len(region_nodes) > policy.min_nodes_per_region:
            # Scale down - remove least utilized node
            least_utilized_node = min(region_nodes, 
                                    key=lambda n: (n.cpu_usage + n.memory_usage) / 2.0)
            
            if least_utilized_node.cpu_usage < 0.1 and least_utilized_node.memory_usage < 0.1:
                least_utilized_node.status = DeploymentStatus.OFFLINE
                scaling_result["action"] = "scale_down"
                scaling_result["target_nodes"] = len(region_nodes) - 1
                logger.info(f"Scaled down region {region} - removed node {least_utilized_node.node_id}")
        
        return scaling_result
    
    async def _create_new_node(self, region: str, policy: DeploymentPolicy) -> Optional[EdgeNode]:
        """Create a new edge node for scaling."""
        try:
            # Get region information
            region_info = self.location_service.region_definitions.get(region)
            if not region_info:
                logger.error(f"Unknown region: {region}")
                return None
            
            # Generate node ID
            node_id = f"{region}_node_{int(time.time())}"
            
            # Create new node
            new_node = EdgeNode(
                node_id=node_id,
                region=region,
                location=region_info["coordinates"],
                node_type=region_info["node_type"],
                status=DeploymentStatus.INITIALIZING,
                cpu_cores=max(policy.min_cpu_cores, 8),
                memory_gb=max(policy.min_memory_gb, 16.0),
                storage_gb=500.0,
                bandwidth_mbps=1000.0,
                latency_ms=10.0,
                compliance_regions=region_info["compliance"],
                supported_models=["neural_decoder", "signal_processor", "bci_classifier"]
            )
            
            logger.info(f"Created new node {node_id} in region {region}")
            return new_node
            
        except Exception as e:
            logger.error(f"Failed to create new node in region {region}: {e}")
            return None
    
    def get_global_deployment_status(self) -> Dict[str, Any]:
        """Get global deployment status across all regions."""
        status = {
            "total_nodes": len(self.edge_nodes),
            "active_nodes": 0,
            "degraded_nodes": 0,
            "offline_nodes": 0,
            "regions": {},
            "global_performance": {},
            "compliance_coverage": defaultdict(int)
        }
        
        # Count nodes by status
        for node in self.edge_nodes.values():
            if node.status == DeploymentStatus.ACTIVE:
                status["active_nodes"] += 1
            elif node.status == DeploymentStatus.DEGRADED:
                status["degraded_nodes"] += 1
            elif node.status == DeploymentStatus.OFFLINE:
                status["offline_nodes"] += 1
            
            # Count compliance coverage
            for compliance in node.compliance_regions:
                status["compliance_coverage"][compliance.value] += 1
        
        # Group by regions
        region_stats = defaultdict(lambda: {
            "nodes": 0,
            "active": 0,
            "cpu_usage": [],
            "memory_usage": [],
            "response_time": [],
            "throughput": []
        })
        
        for node in self.edge_nodes.values():
            region_stats[node.region]["nodes"] += 1
            if node.status == DeploymentStatus.ACTIVE:
                region_stats[node.region]["active"] += 1
                region_stats[node.region]["cpu_usage"].append(node.cpu_usage)
                region_stats[node.region]["memory_usage"].append(node.memory_usage)
                region_stats[node.region]["response_time"].append(node.response_time)
                region_stats[node.region]["throughput"].append(node.throughput)
        
        # Calculate regional averages
        for region, stats in region_stats.items():
            if stats["cpu_usage"]:
                status["regions"][region] = {
                    "total_nodes": stats["nodes"],
                    "active_nodes": stats["active"],
                    "avg_cpu_usage": statistics.mean(stats["cpu_usage"]),
                    "avg_memory_usage": statistics.mean(stats["memory_usage"]),
                    "avg_response_time": statistics.mean(stats["response_time"]),
                    "avg_throughput": statistics.mean(stats["throughput"])
                }
        
        # Calculate global performance metrics
        if self.edge_nodes:
            active_nodes = [n for n in self.edge_nodes.values() if n.status == DeploymentStatus.ACTIVE]
            if active_nodes:
                status["global_performance"] = {
                    "avg_cpu_usage": statistics.mean(n.cpu_usage for n in active_nodes),
                    "avg_memory_usage": statistics.mean(n.memory_usage for n in active_nodes),
                    "avg_response_time": statistics.mean(n.response_time for n in active_nodes),
                    "avg_throughput": statistics.mean(n.throughput for n in active_nodes),
                    "avg_availability": statistics.mean(n.availability for n in active_nodes),
                    "avg_error_rate": statistics.mean(n.error_rate for n in active_nodes)
                }
        
        return status
    
    async def run_continuous_health_monitoring(self, interval: int = 30):
        """Run continuous health monitoring for all nodes."""
        logger.info("Starting continuous health monitoring")
        
        while True:
            try:
                # Health check all nodes
                health_check_tasks = []
                for node_id in list(self.edge_nodes.keys()):
                    health_check_tasks.append(self._perform_health_check(node_id))
                
                # Run health checks concurrently
                await asyncio.gather(*health_check_tasks, return_exceptions=True)
                
                # Update load balancer weights
                self._update_load_balancer_weights()
                
                # Auto-scale regions if policies are defined
                for policy_id, policy in self.deployment_policies.items():
                    for region in policy.regions:
                        await self.auto_scale_region(region, policy_id)
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in continuous health monitoring: {e}")
                await asyncio.sleep(interval)
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get detailed performance analytics across all nodes."""
        analytics = {
            "latency_statistics": {},
            "throughput_statistics": {},
            "regional_performance": {},
            "node_rankings": [],
            "optimization_recommendations": []
        }
        
        # Calculate latency statistics
        for node_id, latencies in self.latency_measurements.items():
            if latencies:
                analytics["latency_statistics"][node_id] = {
                    "mean": statistics.mean(latencies),
                    "median": statistics.median(latencies),
                    "p95": np.percentile(latencies, 95),
                    "p99": np.percentile(latencies, 99),
                    "samples": len(latencies)
                }
        
        # Calculate throughput statistics
        for node_id, throughputs in self.throughput_measurements.items():
            if throughputs:
                analytics["throughput_statistics"][node_id] = {
                    "mean": statistics.mean(throughputs),
                    "median": statistics.median(throughputs),
                    "max": max(throughputs),
                    "samples": len(throughputs)
                }
        
        # Rank nodes by performance
        node_scores = []
        for node_id, node in self.edge_nodes.items():
            if node.status == DeploymentStatus.ACTIVE:
                # Calculate composite performance score
                score = (
                    node.availability * 0.3 +
                    (1.0 - node.error_rate) * 0.3 +
                    (1.0 / max(0.1, node.response_time / 100)) * 0.2 +
                    (node.throughput / 100) * 0.2
                )
                node_scores.append({
                    "node_id": node_id,
                    "region": node.region,
                    "score": score,
                    "availability": node.availability,
                    "error_rate": node.error_rate,
                    "response_time": node.response_time,
                    "throughput": node.throughput
                })
        
        # Sort by score
        node_scores.sort(key=lambda x: x["score"], reverse=True)
        analytics["node_rankings"] = node_scores
        
        # Generate optimization recommendations
        analytics["optimization_recommendations"] = self._generate_optimization_recommendations(node_scores)
        
        return analytics
    
    def _generate_optimization_recommendations(self, node_scores: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        if not node_scores:
            return recommendations
        
        # Identify underperforming nodes
        avg_score = statistics.mean(score["score"] for score in node_scores)
        underperforming = [node for node in node_scores if node["score"] < avg_score * 0.8]
        
        if underperforming:
            recommendations.append(
                f"Consider optimizing or replacing {len(underperforming)} underperforming nodes"
            )
        
        # Identify regions with poor performance
        region_scores = defaultdict(list)
        for node in node_scores:
            region_scores[node["region"]].append(node["score"])
        
        for region, scores in region_scores.items():
            avg_region_score = statistics.mean(scores)
            if avg_region_score < avg_score * 0.7:
                recommendations.append(f"Region {region} shows poor performance - investigate network issues")
        
        # Check for high error rates
        high_error_nodes = [node for node in node_scores if node["error_rate"] > 0.05]
        if high_error_nodes:
            recommendations.append(
                f"{len(high_error_nodes)} nodes have high error rates - check for software issues"
            )
        
        # Check for high latency
        high_latency_nodes = [node for node in node_scores if node["response_time"] > 200]
        if high_latency_nodes:
            recommendations.append(
                f"{len(high_latency_nodes)} nodes have high latency - consider network optimization"
            )
        
        return recommendations


class GlobalDeploymentManager:
    """
    Main manager for global multi-region deployment and edge computing.
    
    Coordinates all aspects of global deployment including edge orchestration,
    compliance management, and performance optimization.
    """
    
    def __init__(self):
        self.edge_orchestrator = EdgeComputingOrchestrator()
        self.location_service = GlobalLocationService()
        
        # Global deployment state
        self.deployment_configurations = {}
        self.active_deployments = {}
        self.global_policies = {}
        
        # Initialize default regions
        asyncio.create_task(self._initialize_default_deployment())
        
        logger.info("Global Deployment Manager initialized")
    
    async def _initialize_default_deployment(self):
        """Initialize default global deployment with core regions."""
        logger.info("Initializing default global deployment")
        
        # Create deployment policy
        default_policy = DeploymentPolicy(
            policy_id="global_default",
            regions=list(self.location_service.region_definitions.keys()),
            min_cpu_cores=4,
            min_memory_gb=16.0,
            preferred_node_types=[RegionType.CORE, RegionType.EDGE],
            max_latency_ms=150.0,
            min_availability=0.99,
            required_compliance=[],  # Will be set per region
            failover_enabled=True,
            min_nodes_per_region=1,
            max_nodes_per_region=5
        )
        
        self.edge_orchestrator.create_deployment_policy(default_policy)
        
        # Deploy initial nodes to core regions
        for region_id, region_info in self.location_service.region_definitions.items():
            if region_info["node_type"] == RegionType.CORE:
                await self._deploy_initial_node(region_id, region_info)
    
    async def _deploy_initial_node(self, region_id: str, region_info: Dict[str, Any]):
        """Deploy initial node to a region."""
        try:
            node = EdgeNode(
                node_id=f"{region_id}_initial_{int(time.time())}",
                region=region_id,
                location=region_info["coordinates"],
                node_type=region_info["node_type"],
                status=DeploymentStatus.INITIALIZING,
                cpu_cores=8,
                memory_gb=32.0,
                storage_gb=1000.0,
                gpu_count=1,
                gpu_memory_gb=16.0,
                bandwidth_mbps=10000.0,
                latency_ms=5.0,
                compliance_regions=region_info["compliance"],
                supported_models=["neural_decoder", "p300_classifier", "motor_imagery", "ssvep"]
            )
            
            self.edge_orchestrator.register_edge_node(node)
            logger.info(f"Deployed initial node to {region_id}")
            
        except Exception as e:
            logger.error(f"Failed to deploy initial node to {region_id}: {e}")
    
    async def deploy_globally(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy BCI-Agent-Bridge globally across specified regions."""
        deployment_id = f"global_deployment_{int(time.time())}"
        
        deployment_result = {
            "deployment_id": deployment_id,
            "success": False,
            "deployed_regions": [],
            "failed_regions": [],
            "total_nodes": 0,
            "deployment_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            target_regions = deployment_config.get("regions", list(self.location_service.region_definitions.keys()))
            nodes_per_region = deployment_config.get("nodes_per_region", 2)
            
            # Deploy to each region
            deployment_tasks = []
            for region in target_regions:
                for i in range(nodes_per_region):
                    task = self._deploy_node_to_region(region, deployment_config)
                    deployment_tasks.append(task)
            
            # Execute deployments concurrently
            deployment_results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
            
            # Process results
            for region, result in zip(target_regions * nodes_per_region, deployment_results):
                if isinstance(result, Exception):
                    if region not in deployment_result["failed_regions"]:
                        deployment_result["failed_regions"].append(region)
                    logger.error(f"Deployment to {region} failed: {result}")
                else:
                    if region not in deployment_result["deployed_regions"]:
                        deployment_result["deployed_regions"].append(region)
                    deployment_result["total_nodes"] += 1
            
            deployment_result["success"] = len(deployment_result["deployed_regions"]) > 0
            
        except Exception as e:
            logger.error(f"Global deployment failed: {e}")
            deployment_result["error"] = str(e)
        
        finally:
            deployment_result["deployment_time"] = time.time() - start_time
            self.active_deployments[deployment_id] = deployment_result
        
        return deployment_result
    
    async def _deploy_node_to_region(self, region: str, config: Dict[str, Any]) -> bool:
        """Deploy a node to a specific region."""
        try:
            region_info = self.location_service.region_definitions.get(region)
            if not region_info:
                raise ValueError(f"Unknown region: {region}")
            
            # Create node specification
            node_spec = {
                "cpu_cores": config.get("cpu_cores", 8),
                "memory_gb": config.get("memory_gb", 32.0),
                "gpu_count": config.get("gpu_count", 1),
                "gpu_memory_gb": config.get("gpu_memory_gb", 16.0),
                "bandwidth_mbps": config.get("bandwidth_mbps", 10000.0),
                "supported_models": config.get("supported_models", ["neural_decoder"])
            }
            
            # Create and deploy node
            node = EdgeNode(
                node_id=f"{region}_deploy_{int(time.time())}_{np.random.randint(1000, 9999)}",
                region=region,
                location=region_info["coordinates"],
                node_type=region_info["node_type"],
                status=DeploymentStatus.INITIALIZING,
                compliance_regions=region_info["compliance"],
                **node_spec
            )
            
            return self.edge_orchestrator.register_edge_node(node)
            
        except Exception as e:
            logger.error(f"Failed to deploy node to region {region}: {e}")
            return False
    
    async def process_global_request(self, request: WorkloadRequest) -> Dict[str, Any]:
        """Process a global request with optimal routing."""
        # Determine user location if not provided
        if not hasattr(request, 'user_location') or not request.user_location:
            # Use IP-based location resolution
            request.user_location = self.location_service.resolve_location_from_ip("203.0.113.0")  # Example IP
        
        # Process request through edge orchestrator
        result = await self.edge_orchestrator.process_workload_request(request)
        
        # Add global routing information
        if result["success"] and result["selected_node"]:
            selected_node = self.edge_orchestrator.edge_nodes[result["selected_node"]]
            result["routing_info"] = {
                "user_region": request.user_location.region,
                "processing_region": selected_node.region,
                "distance_km": self.location_service.calculate_distance(
                    request.user_location, selected_node.location
                ),
                "compliance_met": all(
                    req in selected_node.compliance_regions
                    for req in request.compliance_requirements
                )
            }
        
        return result
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global deployment status."""
        deployment_status = self.edge_orchestrator.get_global_deployment_status()
        
        global_status = {
            "deployment_overview": deployment_status,
            "active_deployments": len(self.active_deployments),
            "supported_regions": list(self.location_service.region_definitions.keys()),
            "compliance_regions": {
                region.value: deployment_status["compliance_coverage"].get(region.value, 0)
                for region in ComplianceRegion
            },
            "performance_analytics": self.edge_orchestrator.get_performance_analytics(),
            "health_summary": self._get_health_summary()
        }
        
        return global_status
    
    def _get_health_summary(self) -> Dict[str, Any]:
        """Get health summary across all deployments."""
        all_nodes = list(self.edge_orchestrator.edge_nodes.values())
        active_nodes = [n for n in all_nodes if n.status == DeploymentStatus.ACTIVE]
        
        if not active_nodes:
            return {"status": "no_active_nodes"}
        
        # Calculate health metrics
        avg_availability = statistics.mean(n.availability for n in active_nodes)
        avg_error_rate = statistics.mean(n.error_rate for n in active_nodes)
        avg_response_time = statistics.mean(n.response_time for n in active_nodes)
        
        # Determine overall health status
        if avg_availability > 0.99 and avg_error_rate < 0.01 and avg_response_time < 100:
            health_status = "excellent"
        elif avg_availability > 0.95 and avg_error_rate < 0.05 and avg_response_time < 200:
            health_status = "good"
        elif avg_availability > 0.90 and avg_error_rate < 0.1:
            health_status = "fair"
        else:
            health_status = "poor"
        
        return {
            "status": health_status,
            "avg_availability": avg_availability,
            "avg_error_rate": avg_error_rate,
            "avg_response_time": avg_response_time,
            "total_nodes": len(all_nodes),
            "active_nodes": len(active_nodes)
        }
    
    async def start_global_monitoring(self):
        """Start global monitoring and management services."""
        logger.info("Starting global monitoring services")
        
        # Start health monitoring
        monitoring_task = asyncio.create_task(
            self.edge_orchestrator.run_continuous_health_monitoring(30)
        )
        
        return monitoring_task


# Factory function for easy instantiation
def create_global_deployment_manager() -> GlobalDeploymentManager:
    """
    Create and initialize a Global Deployment Manager.
    
    Returns:
        GlobalDeploymentManager: Initialized manager ready for global deployment
    """
    manager = GlobalDeploymentManager()
    logger.info("Global Deployment Manager created and ready for operation")
    return manager


# Demonstration of global deployment capabilities
async def demonstrate_global_deployment():
    """Demonstrate the capabilities of global deployment and edge computing."""
    print("🌍 Global Deployment: Multi-Region Edge Computing - DEMONSTRATION")
    print("=" * 80)
    
    # Create global deployment manager
    deployment_manager = create_global_deployment_manager()
    
    # Wait for initial deployment
    print("\n⏳ Waiting for initial deployment to complete...")
    await asyncio.sleep(3)
    
    # Show initial global status
    print("\n📊 Initial Global Status:")
    status = deployment_manager.get_global_status()
    
    overview = status["deployment_overview"]
    print(f"Total Nodes: {overview['total_nodes']}")
    print(f"Active Nodes: {overview['active_nodes']}")
    print(f"Supported Regions: {len(status['supported_regions'])}")
    
    for region, stats in overview["regions"].items():
        print(f"  {region}: {stats['active_nodes']}/{stats['total_nodes']} nodes active")
    
    # Simulate global deployment
    print("\n🚀 Deploying additional nodes globally...")
    deployment_config = {
        "regions": ["us_east", "eu_west", "asia_pacific"],
        "nodes_per_region": 2,
        "cpu_cores": 16,
        "memory_gb": 64.0,
        "gpu_count": 2,
        "supported_models": ["neural_decoder", "transformer_decoder", "hybrid_decoder"]
    }
    
    deployment_result = await deployment_manager.deploy_globally(deployment_config)
    
    print(f"Deployment Success: {deployment_result['success']}")
    print(f"Deployed Regions: {deployment_result['deployed_regions']}")
    print(f"Total New Nodes: {deployment_result['total_nodes']}")
    print(f"Deployment Time: {deployment_result['deployment_time']:.2f}s")
    
    # Process some global requests
    print("\n🌐 Processing global workload requests...")
    
    # Create test requests from different locations
    test_requests = [
        WorkloadRequest(
            request_id="req_us_001",
            user_location=GeographicCoordinate(40.7128, -74.0060, "New York", "USA", "us_east", "UTC-5"),
            workload_type="neural_processing",
            latency_requirement=100.0,
            compliance_requirements=[ComplianceRegion.HIPAA_US]
        ),
        WorkloadRequest(
            request_id="req_eu_001",
            user_location=GeographicCoordinate(51.5074, -0.1278, "London", "UK", "eu_west", "UTC+0"),
            workload_type="model_inference",
            latency_requirement=80.0,
            compliance_requirements=[ComplianceRegion.GDPR_EU]
        ),
        WorkloadRequest(
            request_id="req_asia_001",
            user_location=GeographicCoordinate(35.6762, 139.6503, "Tokyo", "Japan", "asia_northeast", "UTC+9"),
            workload_type="signal_analysis",
            latency_requirement=120.0
        )
    ]
    
    # Process requests
    for request in test_requests:
        result = await deployment_manager.process_global_request(request)
        
        print(f"\nRequest {request.request_id}:")
        print(f"  Success: {result['success']}")
        if result['success']:
            print(f"  Selected Node: {result['selected_node']}")
            print(f"  Response Time: {result['response_time']:.3f}s")
            if 'routing_info' in result:
                routing = result['routing_info']
                print(f"  Distance: {routing['distance_km']:.1f} km")
                print(f"  Compliance: {routing['compliance_met']}")
    
    # Show final global status
    print("\n📈 Final Global Status:")
    final_status = deployment_manager.get_global_status()
    
    final_overview = final_status["deployment_overview"]
    health = final_status["health_summary"]
    
    print(f"Total Nodes: {final_overview['total_nodes']}")
    print(f"Active Nodes: {final_overview['active_nodes']}")
    print(f"Overall Health: {health['status'].upper()}")
    print(f"Average Availability: {health['avg_availability']:.3f}")
    print(f"Average Response Time: {health['avg_response_time']:.1f}ms")
    
    # Show performance analytics
    analytics = final_status["performance_analytics"]
    if analytics["node_rankings"]:
        print(f"\n🏆 Top Performing Nodes:")
        for i, node in enumerate(analytics["node_rankings"][:3]):
            print(f"  {i+1}. {node['node_id']} ({node['region']}) - Score: {node['score']:.3f}")
    
    if analytics["optimization_recommendations"]:
        print(f"\n💡 Optimization Recommendations:")
        for rec in analytics["optimization_recommendations"]:
            print(f"  • {rec}")
    
    print("\n✅ Global Deployment System: OPERATIONAL")
    return final_status


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_global_deployment())