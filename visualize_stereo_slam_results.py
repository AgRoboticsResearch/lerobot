#!/usr/bin/env python3
"""
Comprehensive Stereo SLAM Visualization and Analysis
Shows detailed results from the stereo ORB-SLAM implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import pandas as pd
from pathlib import Path

def parse_trajectory_file(filepath):
    """Parse the stereo trajectory file."""
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                # Format: timestamp frame_count tx ty tz qx qy qz qw
                parts = line.strip().split()
                if len(parts) >= 9:
                    timestamp = float(parts[0])
                    frame_count = int(parts[1])
                    tx, ty, tz = float(parts[2]), float(parts[3]), float(parts[4])
                    qx, qy, qz, qw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                    poses.append([timestamp, frame_count, tx, ty, tz, qx, qy, qz, qw])
    
    return np.array(poses)

def create_comprehensive_visualization(trajectory_file, output_dir="stereo_slam_analysis"):
    """Create comprehensive stereo SLAM visualization."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Parse trajectory data
    poses = parse_trajectory_file(trajectory_file)
    
    if len(poses) == 0:
        print("No trajectory data found!")
        return
    
    # Extract data
    timestamps = poses[:, 0]
    frame_counts = poses[:, 1].astype(int)
    translations = poses[:, 2:5]
    quaternions = poses[:, 5:9]
    
    # Calculate relative timestamps
    relative_times = timestamps - timestamps[0]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 3D Trajectory Plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(translations[:, 0], translations[:, 1], translations[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax1.scatter(translations[0, 0], translations[0, 1], translations[0, 2], c='green', s=100, label='Start')
    ax1.scatter(translations[-1, 0], translations[-1, 1], translations[-1, 2], c='red', s=100, label='End')
    
    # Add coordinate axes
    ax1.quiver(0, 0, 0, 1, 0, 0, color='red', arrow_length_ratio=0.1, label='X')
    ax1.quiver(0, 0, 0, 0, 1, 0, color='green', arrow_length_ratio=0.1, label='Y')
    ax1.quiver(0, 0, 0, 0, 0, 1, color='blue', arrow_length_ratio=0.1, label='Z')
    
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_zlabel('Z (meters)')
    ax1.set_title('Stereo SLAM 3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Position vs Time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(relative_times, translations[:, 0], 'r-', label='X', linewidth=2)
    ax2.plot(relative_times, translations[:, 1], 'g-', label='Y', linewidth=2)
    ax2.plot(relative_times, translations[:, 2], 'b-', label='Z', linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Position (meters)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # 3. XY Projection (Top View)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(translations[:, 0], translations[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax3.scatter(translations[0, 0], translations[0, 1], c='green', s=100, label='Start')
    ax3.scatter(translations[-1, 0], translations[-1, 1], c='red', s=100, label='End')
    ax3.set_xlabel('X (meters)')
    ax3.set_ylabel('Y (meters)')
    ax3.set_title('XY Projection (Top View)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # 4. Velocity Analysis
    ax4 = fig.add_subplot(2, 3, 4)
    # Calculate velocities
    velocities = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    time_diffs = np.diff(relative_times)
    instantaneous_velocities = velocities / time_diffs
    
    ax4.plot(relative_times[1:], instantaneous_velocities, 'purple', linewidth=2)
    ax4.axhline(y=np.mean(instantaneous_velocities), color='red', linestyle='--', 
                label=f'Avg: {np.mean(instantaneous_velocities):.3f} m/s')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Instantaneous Velocity')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Distance from Origin
    ax5 = fig.add_subplot(2, 3, 5)
    distances_from_origin = np.linalg.norm(translations, axis=1)
    ax5.plot(relative_times, distances_from_origin, 'orange', linewidth=2)
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Distance from Origin (meters)')
    ax5.set_title('Distance from Origin')
    ax5.grid(True)
    
    # 6. Translation Components
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(relative_times, translations[:, 0], 'r-', label='X', alpha=0.7)
    ax6.plot(relative_times, translations[:, 1], 'g-', label='Y', alpha=0.7)
    ax6.plot(relative_times, translations[:, 2], 'b-', label='Z', alpha=0.7)
    ax6.fill_between(relative_times, translations[:, 0], alpha=0.2, color='red')
    ax6.fill_between(relative_times, translations[:, 1], alpha=0.2, color='green')
    ax6.fill_between(relative_times, translations[:, 2], alpha=0.2, color='blue')
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Translation (meters)')
    ax6.set_title('Translation Components')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_stereo_slam_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create statistics summary
    total_distance = np.sum(velocities)
    avg_velocity = total_distance / relative_times[-1]
    drift_distance = np.linalg.norm(translations[-1] - translations[0])
    velocity_variance = np.var(instantaneous_velocities)
    
    print("\n" + "="*60)
    print("🎯 STEREO SLAM COMPREHENSIVE ANALYSIS")
    print("="*60)
    print(f"📊 Total Frames Processed: {len(poses)}")
    print(f"⏱️  Duration: {relative_times[-1]:.1f} seconds")
    print(f"📏 Total Distance Traveled: {total_distance:.3f} meters")
    print(f"🚀 Average Velocity: {avg_velocity:.3f} m/s")
    print(f"🎯 Drift Distance: {drift_distance:.3f} meters")
    print(f"📈 Velocity Variance: {velocity_variance:.6f}")
    print(f"📍 Translation Ranges:")
    print(f"   X: [{translations[:, 0].min():.3f}, {translations[:, 0].max():.3f}] meters")
    print(f"   Y: [{translations[:, 1].min():.3f}, {translations[:, 1].max():.3f}] meters")
    print(f"   Z: [{translations[:, 2].min():.3f}, {translations[:, 2].max():.3f}] meters")
    print(f"🎨 Visualization saved to: {output_dir}/comprehensive_stereo_slam_analysis.png")
    print("="*60)

def create_trajectory_comparison():
    """Compare stereo vs monocular trajectories if available."""
    stereo_file = "test_stereo_trajectory_extended.txt"
    mono_file = "test_lerobot_live_trajectory.txt"
    
    if not Path(stereo_file).exists() or not Path(mono_file).exists():
        print("Trajectory files not found for comparison")
        return
    
    # Parse both trajectories
    stereo_poses = parse_trajectory_file(stereo_file)
    mono_poses = parse_trajectory_file(mono_file)
    
    if len(stereo_poses) == 0 or len(mono_poses) == 0:
        print("No trajectory data found for comparison")
        return
    
    # Extract translations
    stereo_translations = stereo_poses[:, 2:5]
    mono_translations = mono_poses[:, 2:5]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 3D comparison
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(stereo_translations[:, 0], stereo_translations[:, 1], stereo_translations[:, 2], 
             'b-', linewidth=2, label='Stereo SLAM')
    ax1.plot(mono_translations[:, 0], mono_translations[:, 1], mono_translations[:, 2], 
             'r--', linewidth=2, label='Monocular SLAM')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_zlabel('Z (meters)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # XY comparison
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(stereo_translations[:, 0], stereo_translations[:, 1], 'b-', linewidth=2, label='Stereo')
    ax2.plot(mono_translations[:, 0], mono_translations[:, 1], 'r--', linewidth=2, label='Monocular')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title('XY Projection Comparison')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # Distance comparison
    ax3 = fig.add_subplot(2, 2, 3)
    stereo_distances = np.linalg.norm(stereo_translations, axis=1)
    mono_distances = np.linalg.norm(mono_translations, axis=1)
    
    ax3.plot(stereo_distances, 'b-', linewidth=2, label='Stereo')
    ax3.plot(mono_distances, 'r--', linewidth=2, label='Monocular')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Distance from Origin (meters)')
    ax3.set_title('Distance from Origin Comparison')
    ax3.legend()
    ax3.grid(True)
    
    # Statistics comparison
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate statistics
    stereo_total = np.sum(np.linalg.norm(np.diff(stereo_translations, axis=0), axis=1))
    mono_total = np.sum(np.linalg.norm(np.diff(mono_translations, axis=0), axis=1))
    stereo_drift = np.linalg.norm(stereo_translations[-1] - stereo_translations[0])
    mono_drift = np.linalg.norm(mono_translations[-1] - mono_translations[0])
    
    stats_text = f"""Comparison Statistics:

Stereo SLAM:
• Total Distance: {stereo_total:.3f}m
• Drift: {stereo_drift:.3f}m
• Frames: {len(stereo_poses)}

Monocular SLAM:
• Total Distance: {mono_total:.3f}m
• Drift: {mono_drift:.3f}m
• Frames: {len(mono_poses)}

Improvement:
• Distance Ratio: {stereo_total/mono_total:.2f}x
• Drift Ratio: {stereo_drift/mono_drift:.2f}x
"""
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12, 
             verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig("stereo_vs_mono_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("📊 STEREO vs MONOCULAR COMPARISON")
    print("="*60)
    print(f"🎯 Stereo Total Distance: {stereo_total:.3f}m")
    print(f"👁️  Monocular Total Distance: {mono_total:.3f}m")
    print(f"📈 Distance Ratio: {stereo_total/mono_total:.2f}x")
    print(f"🎯 Stereo Drift: {stereo_drift:.3f}m")
    print(f"👁️  Monocular Drift: {mono_drift:.3f}m")
    print(f"📉 Drift Ratio: {stereo_drift/mono_drift:.2f}x")
    print("="*60)

if __name__ == "__main__":
    print("🎯 Stereo SLAM Visualization and Analysis")
    print("="*50)
    
    # Create comprehensive visualization
    create_comprehensive_visualization("test_stereo_trajectory_extended.txt")
    
    # Create comparison if monocular data exists
    create_trajectory_comparison()
    
    print("\n✅ Visualization complete!")
    print("📁 Check the generated PNG files for detailed analysis") 