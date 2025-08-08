#!/usr/bin/env python3
"""
Visualize ORB-SLAM Trajectory

This script loads and visualizes the saved ORB-SLAM trajectory data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os


def load_trajectory(filename):
    """Load trajectory data from .npz file."""
    data = np.load(filename)
    points = data['points']
    times = data['times']
    return points, times


def visualize_trajectory_3d(points, times, title="ORB-SLAM Camera Trajectory"):
    """Create 3D visualization of the trajectory."""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax1.scatter(points[0, 0], points[0, 1], points[0, 2], c='g', s=100, label='Start')
    ax1.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c='r', s=100, label='End')
    
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_zlabel('Z (meters)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)
    
    # 2D projections
    ax2 = fig.add_subplot(222)
    ax2.plot(points[:, 0], points[:, 1], 'b-', linewidth=2)
    ax2.scatter(points[0, 0], points[0, 1], c='g', s=100, label='Start')
    ax2.scatter(points[-1, 0], points[-1, 1], c='r', s=100, label='End')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title('Top View (X-Y)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    ax3 = fig.add_subplot(223)
    ax3.plot(points[:, 0], points[:, 2], 'b-', linewidth=2)
    ax3.scatter(points[0, 0], points[0, 2], c='g', s=100, label='Start')
    ax3.scatter(points[-1, 0], points[-1, 2], c='r', s=100, label='End')
    ax3.set_xlabel('X (meters)')
    ax3.set_ylabel('Z (meters)')
    ax3.set_title('Side View (X-Z)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    ax4 = fig.add_subplot(224)
    ax4.plot(points[:, 1], points[:, 2], 'b-', linewidth=2)
    ax4.scatter(points[0, 1], points[0, 2], c='g', s=100, label='Start')
    ax4.scatter(points[-1, 1], points[-1, 2], c='r', s=100, label='End')
    ax4.set_xlabel('Y (meters)')
    ax4.set_ylabel('Z (meters)')
    ax4.set_title('Side View (Y-Z)')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')
    
    plt.tight_layout()
    return fig


def analyze_trajectory(points, times):
    """Analyze trajectory statistics."""
    # Calculate distances
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_distance = np.sum(distances)
    max_distance_from_start = np.max(np.linalg.norm(points - points[0], axis=1))
    
    # Calculate velocities
    velocities = distances / np.diff(times)
    avg_velocity = np.mean(velocities)
    max_velocity = np.max(velocities)
    
    # Calculate trajectory statistics
    duration = times[-1] - times[0]
    avg_speed = total_distance / duration
    
    print("📊 Trajectory Analysis:")
    print(f"   Total points: {len(points)}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Total distance: {total_distance:.3f} meters")
    print(f"   Max distance from start: {max_distance_from_start:.3f} meters")
    print(f"   Average speed: {avg_speed:.3f} m/s")
    print(f"   Average velocity: {avg_velocity:.3f} m/s")
    print(f"   Max velocity: {max_velocity:.3f} m/s")
    
    # Position ranges
    x_range = np.ptp(points[:, 0])
    y_range = np.ptp(points[:, 1])
    z_range = np.ptp(points[:, 2])
    print(f"   X range: {x_range:.3f} meters")
    print(f"   Y range: {y_range:.3f} meters")
    print(f"   Z range: {z_range:.3f} meters")
    
    return {
        'total_distance': total_distance,
        'max_distance': max_distance_from_start,
        'duration': duration,
        'avg_speed': avg_speed,
        'avg_velocity': avg_velocity,
        'max_velocity': max_velocity
    }


def plot_velocity_profile(points, times):
    """Plot velocity profile over time."""
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    velocities = distances / np.diff(times)
    time_midpoints = (times[:-1] + times[1:]) / 2
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Velocity over time
    ax1.plot(time_midpoints, velocities, 'b-', linewidth=2)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Velocity Profile')
    ax1.grid(True)
    
    # Cumulative distance
    cumulative_distance = np.cumsum(distances)
    ax2.plot(times[1:], cumulative_distance, 'r-', linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Cumulative Distance (meters)')
    ax2.set_title('Cumulative Distance')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


def main():
    """Main function."""
    print("🎯 ORB-SLAM Trajectory Visualization")
    print("=" * 50)
    
    # Find trajectory files
    trajectory_files = glob.glob("orb_slam_trajectory_*.npz")
    
    if not trajectory_files:
        print("❌ No trajectory files found!")
        print("Run the visualization test first to generate trajectory data.")
        return
    
    # Use the most recent file
    latest_file = max(trajectory_files, key=os.path.getctime)
    print(f"📁 Loading trajectory from: {latest_file}")
    
    try:
        # Load trajectory data
        points, times = load_trajectory(latest_file)
        print(f"✅ Loaded {len(points)} trajectory points")
        
        # Analyze trajectory
        stats = analyze_trajectory(points, times)
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        
        # 3D trajectory plot
        fig1 = visualize_trajectory_3d(points, times)
        fig1.savefig('trajectory_3d.png', dpi=300, bbox_inches='tight')
        print("💾 Saved 3D trajectory plot as 'trajectory_3d.png'")
        
        # Velocity profile
        fig2 = plot_velocity_profile(points, times)
        fig2.savefig('velocity_profile.png', dpi=300, bbox_inches='tight')
        print("💾 Saved velocity profile as 'velocity_profile.png'")
        
        # Show plots
        plt.show()
        
        print("\n✅ Trajectory visualization completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 