"""Check available Ray resources"""
import ray
import psutil

print("System Resources:")
print(f"  CPU cores: {psutil.cpu_count()}")
print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

try:
    import torch
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print("  GPU: Not available")
except:
    print("  GPU: Could not detect")

print("\nTesting Ray resource allocation...")
print("\nWith num_cpus=0.4, num_gpus=0.09:")
print(f"  Max actors by CPU (2 CPUs): {2 / 0.4:.1f} actors")
print(f"  Max actors by GPU (1 GPU): {1 / 0.09:.1f} actors")
print(f"  → Limiting factor: CPU allows only 5 actors")
print(f"  → Problem: Full experiment needs 10 clients!")

print("\nRecommended fix for A100:")
print("  num_cpus=0.2, num_gpus=0.09")
print(f"  Max actors by CPU: {2 / 0.2:.1f} actors")
print(f"  Max actors by GPU: {1 / 0.09:.1f} actors")
print(f"  → Can handle 10 clients ✓")
