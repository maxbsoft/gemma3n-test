#!/usr/bin/env python3
"""
GPU Memory Cleanup Script
"""
import requests
import sys

def cleanup_gpu_memory(api_base="http://localhost:8000"):
    """Force cleanup GPU memory via API"""
    try:
        response = requests.post(f"{api_base}/v1/gpu/cleanup", timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            if 'free_memory_gb' in result:
                print(f"💾 Free memory: {result['free_memory_gb']} GB")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to server: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def get_gpu_status(api_base="http://localhost:8000"):
    """Get current GPU status"""
    try:
        response = requests.get(f"{api_base}/v1/status", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('gpu_info'):
                gpu_info = result['gpu_info']
                print(f"🖥️  GPU: {gpu_info.get('name', 'Unknown')}")
                print(f"💾 Memory allocated: {gpu_info.get('memory_allocated', 'Unknown')}")
                print(f"💾 Memory total: {gpu_info.get('memory_total', 'Unknown')}")
                return True
            else:
                print("❌ No GPU information available")
                return False
        else:
            print(f"❌ Error getting status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to get GPU status: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Memory Management")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--cleanup", action="store_true", help="Force cleanup GPU memory")
    parser.add_argument("--status", action="store_true", help="Show GPU status")
    
    args = parser.parse_args()
    
    if not args.cleanup and not args.status:
        # Default: show status and cleanup
        print("🔍 Checking GPU status...")
        get_gpu_status(args.api)
        print("\n🧹 Cleaning up GPU memory...")
        cleanup_gpu_memory(args.api)
    else:
        if args.status:
            print("🔍 GPU Status:")
            get_gpu_status(args.api)
        
        if args.cleanup:
            print("🧹 Cleaning up GPU memory...")
            cleanup_gpu_memory(args.api) 