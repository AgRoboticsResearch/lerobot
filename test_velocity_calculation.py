#!/usr/bin/env python3
"""
Test Velocity Calculation

This script tests the velocity calculation logic to ensure it works correctly.
"""

import time
import numpy as np
from collections import deque


def test_velocity_calculation():
    """Test velocity calculation logic."""
    print("🧪 Testing Velocity Calculation")
    print("=" * 40)
    
    # Initialize test variables (similar to teleoperator)
    pose_history = deque(maxlen=10)
    velocity_history = deque(maxlen=5)
    last_update_time = None
    
    # Test 1: Stationary camera (should show 0 velocity)
    print("\n📷 Test 1: Stationary Camera")
    print("-" * 20)
    
    # Simulate stationary camera with small noise
    base_pos = np.array([1.0, 2.0, 3.0])
    
    for i in range(5):
        current_time = time.time()
        
        # Add small noise to simulate ORB-SLAM variations
        noise = np.random.normal(0, 0.001, 3)  # 1mm noise
        current_pos = base_pos + noise
        
        # Create pose matrix
        pose = np.eye(4)
        pose[:3, 3] = current_pos
        pose_history.append(pose)
        
        # Calculate velocity
        if len(pose_history) >= 2 and last_update_time is not None:
            dt = current_time - last_update_time
            if dt > 0.001:
                prev_pos = pose_history[-2][:3, 3]
                velocity = np.linalg.norm(current_pos - prev_pos) / dt
                if velocity > 0.01:  # 1cm/s threshold
                    velocity_history.append(velocity)
                else:
                    velocity_history.append(0.0)
            else:
                velocity_history.append(0.0)
        else:
            velocity_history.append(0.0)
        
        last_update_time = current_time
        
        # Display results
        avg_velocity = np.mean(velocity_history) if velocity_history else 0.0
        print(f"Frame {i+1}: Position {current_pos}, Velocity: {avg_velocity:.6f} m/s")
        
        time.sleep(0.1)  # Simulate 10Hz
    
    # Test 2: Moving camera (should show non-zero velocity)
    print("\n📷 Test 2: Moving Camera")
    print("-" * 20)
    
    # Clear history
    pose_history.clear()
    velocity_history.clear()
    last_update_time = None
    
    # Simulate moving camera
    for i in range(5):
        current_time = time.time()
        
        # Move camera in x direction
        current_pos = np.array([1.0 + i * 0.01, 2.0, 3.0])  # 1cm per frame
        
        # Create pose matrix
        pose = np.eye(4)
        pose[:3, 3] = current_pos
        pose_history.append(pose)
        
        # Calculate velocity
        if len(pose_history) >= 2 and last_update_time is not None:
            dt = current_time - last_update_time
            if dt > 0.001:
                prev_pos = pose_history[-2][:3, 3]
                velocity = np.linalg.norm(current_pos - prev_pos) / dt
                if velocity > 0.01:  # 1cm/s threshold
                    velocity_history.append(velocity)
                else:
                    velocity_history.append(0.0)
            else:
                velocity_history.append(0.0)
        else:
            velocity_history.append(0.0)
        
        last_update_time = current_time
        
        # Display results
        avg_velocity = np.mean(velocity_history) if velocity_history else 0.0
        print(f"Frame {i+1}: Position {current_pos}, Velocity: {avg_velocity:.6f} m/s")
        
        time.sleep(0.1)  # Simulate 10Hz
    
    print("\n✅ Velocity calculation test completed!")


if __name__ == "__main__":
    test_velocity_calculation() 