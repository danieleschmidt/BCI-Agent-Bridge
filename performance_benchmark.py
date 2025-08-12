#!/usr/bin/env python3
"""
Performance benchmark testing for BCI-Agent-Bridge
"""

import time
import numpy as np
import asyncio
import json
import psutil
import gc

def run_performance_benchmarks():
    print('⚡ Running performance benchmarks...')
    
    benchmark_results = {
        'timestamp': time.time(),
        'benchmarks': {},
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2)
        }
    }
    
    # Benchmark 1: Neural data processing latency
    print('📊 Benchmarking neural data processing latency...')
    try:
        from bci_agent_bridge import BCIBridge
        from bci_agent_bridge.core.bridge import NeuralData
        
        bridge = BCIBridge(paradigm='P300')
        
        # Test various data sizes
        latencies = []
        data_sizes = [250, 500, 1000, 2000]  # samples
        
        for size in data_sizes:
            test_data = np.random.randn(8, size)
            neural_data = NeuralData(
                data=test_data,
                timestamp=time.time(),
                channels=[f'CH{i}' for i in range(1, 9)],
                sampling_rate=250
            )
            
            # Measure processing time
            start_time = time.perf_counter()
            intention = bridge.decode_intention(neural_data)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            print(f'  {size} samples: {latency_ms:.2f}ms')
        
        benchmark_results['benchmarks']['neural_processing'] = {
            'latencies_ms': latencies,
            'avg_latency_ms': np.mean(latencies),
            'max_latency_ms': np.max(latencies),
            'min_latency_ms': np.min(latencies)
        }
        
        print(f'✅ Neural processing: avg={np.mean(latencies):.2f}ms, max={np.max(latencies):.2f}ms')
        
    except Exception as e:
        print(f'❌ Neural processing benchmark failed: {e}')
        benchmark_results['benchmarks']['neural_processing'] = {'error': str(e)}
    
    # Benchmark 2: Real-time streaming throughput
    print('📊 Benchmarking real-time streaming throughput...')
    try:
        async def streaming_benchmark():
            bridge = BCIBridge(paradigm='P300')
            
            samples_collected = 0
            start_time = time.perf_counter()
            target_duration = 5.0  # seconds
            
            async for neural_data in bridge.stream():
                samples_collected += 1
                
                elapsed = time.perf_counter() - start_time
                if elapsed >= target_duration:
                    bridge.stop_streaming()
                    break
            
            throughput = samples_collected / elapsed
            return throughput, samples_collected, elapsed
        
        throughput, samples, duration = asyncio.run(streaming_benchmark())
        
        benchmark_results['benchmarks']['streaming'] = {
            'throughput_samples_per_sec': throughput,
            'total_samples': samples,
            'duration_sec': duration
        }
        
        print(f'✅ Streaming throughput: {throughput:.1f} samples/sec ({samples} samples in {duration:.2f}s)')
        
    except Exception as e:
        print(f'❌ Streaming benchmark failed: {e}')
        benchmark_results['benchmarks']['streaming'] = {'error': str(e)}
    
    # Benchmark 3: Memory usage under load
    print('📊 Benchmarking memory usage under load...')
    try:
        # Get baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        bridge = BCIBridge(paradigm='P300', buffer_size=5000)
        
        # Generate load
        memory_samples = [baseline_memory]
        for i in range(100):
            test_data = np.random.randn(8, 250)
            neural_data = NeuralData(
                data=test_data,
                timestamp=time.time(),
                channels=[f'CH{i}' for i in range(1, 9)],
                sampling_rate=250
            )
            
            bridge._add_to_buffer_safe(neural_data)
            
            if i % 20 == 0:  # Sample memory every 20 iterations
                current_memory = psutil.Process().memory_info().rss / (1024**2)
                memory_samples.append(current_memory)
        
        final_memory = psutil.Process().memory_info().rss / (1024**2)
        memory_increase = final_memory - baseline_memory
        
        benchmark_results['benchmarks']['memory_usage'] = {
            'baseline_mb': baseline_memory,
            'final_mb': final_memory,
            'increase_mb': memory_increase,
            'max_mb': max(memory_samples),
            'samples': memory_samples
        }
        
        print(f'✅ Memory usage: baseline={baseline_memory:.1f}MB, increase={memory_increase:.1f}MB')
        
    except Exception as e:
        print(f'❌ Memory benchmark failed: {e}')
        benchmark_results['benchmarks']['memory_usage'] = {'error': str(e)}
    
    # Benchmark 4: Concurrent processing
    print('📊 Benchmarking concurrent processing...')
    try:
        async def concurrent_processing_test():
            bridges = [BCIBridge(paradigm='P300') for _ in range(4)]
            
            async def process_neural_data(bridge, iterations=50):
                processing_times = []
                for i in range(iterations):
                    test_data = np.random.randn(8, 250)
                    neural_data = NeuralData(
                        data=test_data,
                        timestamp=time.time(),
                        channels=[f'CH{i}' for i in range(1, 9)],
                        sampling_rate=250
                    )
                    
                    start_time = time.perf_counter()
                    intention = bridge.decode_intention(neural_data)
                    end_time = time.perf_counter()
                    
                    processing_times.append((end_time - start_time) * 1000)
                
                return processing_times
            
            # Run concurrent processing
            start_time = time.perf_counter()
            results = await asyncio.gather(*[
                process_neural_data(bridge) for bridge in bridges
            ])
            end_time = time.perf_counter()
            
            total_duration = end_time - start_time
            total_operations = sum(len(result) for result in results)
            operations_per_sec = total_operations / total_duration
            
            all_times = [time for result in results for time in result]
            avg_latency = np.mean(all_times)
            
            return operations_per_sec, avg_latency, total_operations, total_duration
        
        ops_per_sec, avg_latency, total_ops, duration = asyncio.run(concurrent_processing_test())
        
        benchmark_results['benchmarks']['concurrent_processing'] = {
            'operations_per_sec': ops_per_sec,
            'avg_latency_ms': avg_latency,
            'total_operations': total_ops,
            'duration_sec': duration
        }
        
        print(f'✅ Concurrent processing: {ops_per_sec:.1f} ops/sec, avg latency={avg_latency:.2f}ms')
        
    except Exception as e:
        print(f'❌ Concurrent processing benchmark failed: {e}')
        benchmark_results['benchmarks']['concurrent_processing'] = {'error': str(e)}
    
    # Benchmark 5: Privacy protection overhead
    print('📊 Benchmarking privacy protection overhead...')
    try:
        from bci_agent_bridge.privacy.differential_privacy import DifferentialPrivacy
        
        privacy_engine = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        # Test without privacy
        test_data = np.random.randn(1000, 100)
        
        start_time = time.perf_counter()
        for i in range(100):
            _ = test_data.copy()  # Simulate processing without privacy
        no_privacy_time = time.perf_counter() - start_time
        
        # Test with privacy
        start_time = time.perf_counter()
        for i in range(100):
            _ = privacy_engine.add_noise(test_data.flatten(), sensitivity=1.0)
        with_privacy_time = time.perf_counter() - start_time
        
        privacy_overhead = ((with_privacy_time - no_privacy_time) / no_privacy_time) * 100
        
        benchmark_results['benchmarks']['privacy_overhead'] = {
            'no_privacy_sec': no_privacy_time,
            'with_privacy_sec': with_privacy_time,
            'overhead_percent': privacy_overhead
        }
        
        print(f'✅ Privacy overhead: {privacy_overhead:.1f}% ({with_privacy_time:.3f}s vs {no_privacy_time:.3f}s)')
        
    except Exception as e:
        print(f'❌ Privacy benchmark failed: {e}')
        benchmark_results['benchmarks']['privacy_overhead'] = {'error': str(e)}
    
    # Performance assessment
    print('')
    print('⚡ PERFORMANCE BENCHMARK RESULTS:')
    print('=' * 60)
    
    # Analyze results
    performance_score = 0
    max_score = 100
    
    # Neural processing latency scoring (target: <10ms)
    if 'neural_processing' in benchmark_results['benchmarks']:
        avg_latency = benchmark_results['benchmarks']['neural_processing'].get('avg_latency_ms', float('inf'))
        if avg_latency < 5:
            performance_score += 25
            print('✅ Neural processing latency: EXCELLENT (<5ms)')
        elif avg_latency < 10:
            performance_score += 20
            print('✅ Neural processing latency: GOOD (<10ms)')
        elif avg_latency < 20:
            performance_score += 15
            print('⚠️  Neural processing latency: ACCEPTABLE (<20ms)')
        else:
            performance_score += 5
            print('❌ Neural processing latency: SLOW (>20ms)')
    
    # Streaming throughput scoring (target: >200 samples/sec)
    if 'streaming' in benchmark_results['benchmarks']:
        throughput = benchmark_results['benchmarks']['streaming'].get('throughput_samples_per_sec', 0)
        if throughput > 200:
            performance_score += 25
            print('✅ Streaming throughput: EXCELLENT (>200 samples/sec)')
        elif throughput > 100:
            performance_score += 20
            print('✅ Streaming throughput: GOOD (>100 samples/sec)')
        elif throughput > 50:
            performance_score += 15
            print('⚠️  Streaming throughput: ACCEPTABLE (>50 samples/sec)')
        else:
            performance_score += 5
            print('❌ Streaming throughput: LOW (<50 samples/sec)')
    
    # Memory usage scoring (target: <100MB increase)
    if 'memory_usage' in benchmark_results['benchmarks']:
        memory_increase = benchmark_results['benchmarks']['memory_usage'].get('increase_mb', float('inf'))
        if memory_increase < 50:
            performance_score += 25
            print('✅ Memory usage: EXCELLENT (<50MB increase)')
        elif memory_increase < 100:
            performance_score += 20
            print('✅ Memory usage: GOOD (<100MB increase)')
        elif memory_increase < 200:
            performance_score += 15
            print('⚠️  Memory usage: ACCEPTABLE (<200MB increase)')
        else:
            performance_score += 5
            print('❌ Memory usage: HIGH (>200MB increase)')
    
    # Privacy overhead scoring (target: <20%)
    if 'privacy_overhead' in benchmark_results['benchmarks']:
        overhead = benchmark_results['benchmarks']['privacy_overhead'].get('overhead_percent', float('inf'))
        if overhead < 10:
            performance_score += 25
            print('✅ Privacy overhead: EXCELLENT (<10%)')
        elif overhead < 20:
            performance_score += 20
            print('✅ Privacy overhead: GOOD (<20%)')
        elif overhead < 40:
            performance_score += 15
            print('⚠️  Privacy overhead: ACCEPTABLE (<40%)')
        else:
            performance_score += 5
            print('❌ Privacy overhead: HIGH (>40%)')
    
    benchmark_results['performance_score'] = performance_score
    benchmark_results['max_score'] = max_score
    
    print('')
    print(f'📊 OVERALL PERFORMANCE SCORE: {performance_score}/{max_score}')
    
    if performance_score >= 85:
        print('🚀 PERFORMANCE STATUS: EXCELLENT')
    elif performance_score >= 70:
        print('⚡ PERFORMANCE STATUS: GOOD')
    elif performance_score >= 55:
        print('⚠️  PERFORMANCE STATUS: ACCEPTABLE')
    else:
        print('❌ PERFORMANCE STATUS: NEEDS OPTIMIZATION')
    
    # Save benchmark report
    with open('performance_benchmark.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    print('')
    print('📄 Performance benchmark report saved to performance_benchmark.json')
    
    return benchmark_results

if __name__ == '__main__':
    run_performance_benchmarks()