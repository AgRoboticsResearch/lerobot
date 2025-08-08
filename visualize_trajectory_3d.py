#!/usr/bin/env python3
"""
3D Trajectory Visualization for ORB-SLAM

This script visualizes the ORB-SLAM trajectories in 3D space
to better understand camera movement and pose estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path


def parse_trajectory_file(filepath):
    """Parse ORB-SLAM trajectory file."""
    timestamps = []
    frame_counts = []
    translations = []
    quaternions = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) >= 9:
                timestamp = float(parts[0])
                frame_count = int(parts[1])
                tx, ty, tz = float(parts[2]), float(parts[3]), float(parts[4])
                qx, qy, qz, qw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                
                timestamps.append(timestamp)
                frame_counts.append(frame_count)
                translations.append([tx, ty, tz])
                quaternions.append([qx, qy, qz, qw])
    
    return np.array(timestamps), np.array(frame_counts), np.array(translations), np.array(quaternions)


def plot_3d_trajectory(translations, quaternions, title, color='blue', marker='o'):
    """Plot 3D trajectory with camera orientations."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory points
    x, y, z = translations[:, 0], translations[:, 1], translations[:, 2]
    ax.scatter(x, y, z, c=color, marker=marker, s=50, alpha=0.7, label='Camera positions')
    
    # Plot trajectory line
    ax.plot(x, y, z, color=color, alpha=0.5, linewidth=2)
    
    # Plot camera orientations (coordinate frames)
    for i in range(0, len(translations), max(1, len(translations)//10)):  # Plot every 10th frame
        pos = translations[i]
        quat = quaternions[i]
        
        # Convert quaternion to rotation matrix
        R = quaternion_to_rotation_matrix(quat)
        
        # Scale factor for coordinate frame visualization
        scale = 0.01
        
        # Plot coordinate frame axes
        origin = pos
        x_axis = origin + scale * R[:, 0]
        y_axis = origin + scale * R[:, 1]
        z_axis = origin + scale * R[:, 2]
        
        ax.quiver(origin[0], origin[1], origin[2], 
                 x_axis[0] - origin[0], x_axis[1] - origin[1], x_axis[2] - origin[2],
                 color='red', alpha=0.7, length=scale, arrow_length_ratio=0.3)
        ax.quiver(origin[0], origin[1], origin[2], 
                 y_axis[0] - origin[0], y_axis[1] - origin[1], y_axis[2] - origin[2],
                 color='green', alpha=0.7, length=scale, arrow_length_ratio=0.3)
        ax.quiver(origin[0], origin[1], origin[2], 
                 z_axis[0] - origin[0], z_axis[1] - origin[1], z_axis[2] - origin[2],
                 color='blue', alpha=0.7, length=scale, arrow_length_ratio=0.3)
    
    # Set labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return fig, ax


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    qx, qy, qz, qw = q
    
    # Normalize quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*qx**2 - 2*qz**2, 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*qx**2 - 2*qy**2]
    ])
    
    return R


def plot_trajectory_comparison(trajectory_files, labels, colors):
    """Plot multiple trajectories for comparison."""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for filepath, label, color in zip(trajectory_files, labels, colors):
        if Path(filepath).exists():
            timestamps, frame_counts, translations, quaternions = parse_trajectory_file(filepath)
            
            x, y, z = translations[:, 0], translations[:, 1], translations[:, 2]
            
            # Plot trajectory
            ax.plot(x, y, z, color=color, linewidth=3, alpha=0.8, label=label)
            ax.scatter(x, y, z, c=color, marker='o', s=30, alpha=0.6)
            
            # Mark start and end points
            ax.scatter(x[0], y[0], z[0], c=color, marker='s', s=100, alpha=0.8, label=f'{label} Start')
            ax.scatter(x[-1], y[-1], z[-1], c=color, marker='^', s=100, alpha=0.8, label=f'{label} End')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('ORB-SLAM Trajectory Comparison')
    ax.legend()
    
    return fig, ax


def plot_trajectory_analysis(translations, quaternions, title):
    """Plot detailed trajectory analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Trajectory Analysis: {title}')
    
    # Position over time
    timestamps = np.arange(len(translations))
    axes[0, 0].plot(timestamps, translations[:, 0], 'r-', label='X', linewidth=2)
    axes[0, 0].plot(timestamps, translations[:, 1], 'g-', label='Y', linewidth=2)
    axes[0, 0].plot(timestamps, translations[:, 2], 'b-', label='Z', linewidth=2)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Position (meters)')
    axes[0, 0].set_title('Position vs Frame')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Velocity (derivative of position)
    velocity = np.diff(translations, axis=0)
    axes[0, 1].plot(timestamps[1:], velocity[:, 0], 'r-', label='Vx', linewidth=2)
    axes[0, 1].plot(timestamps[1:], velocity[:, 1], 'g-', label='Vy', linewidth=2)
    axes[0, 1].plot(timestamps[1:], velocity[:, 2], 'b-', label='Vz', linewidth=2)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Velocity (m/frame)')
    axes[0, 1].set_title('Velocity vs Frame')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 2D trajectory projections
    axes[1, 0].plot(translations[:, 0], translations[:, 1], 'b-', linewidth=2)
    axes[1, 0].scatter(translations[:, 0], translations[:, 1], c=range(len(translations)), cmap='viridis', s=30)
    axes[1, 0].set_xlabel('X (meters)')
    axes[1, 0].set_ylabel('Y (meters)')
    axes[1, 0].set_title('XY Projection')
    axes[1, 0].grid(True)
    axes[1, 0].axis('equal')
    
    # Distance from origin
    distances = np.linalg.norm(translations, axis=1)
    axes[1, 1].plot(timestamps, distances, 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Distance from Origin (meters)')
    axes[1, 1].set_title('Distance from Origin')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize ORB-SLAM trajectories in 3D')
    parser.add_argument('--trajectory-files', nargs='+', 
                       default=['test_trajectory_extended.txt', 'test_stereo_trajectory_extended.txt'],
                       help='Trajectory files to visualize')
    parser.add_argument('--labels', nargs='+', 
                       default=['Extended Depth ORB-SLAM', 'Extended Stereo RGB ORB-SLAM'],
                       help='Labels for each trajectory')
    parser.add_argument('--output-dir', default='trajectory_plots',
                       help='Output directory for plots')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved plots')
    parser.add_argument('--duration', type=int, default=60,
                       help='Expected test duration in seconds for analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Colors for different trajectories
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    print("🎯 Extended ORB-SLAM 3D Trajectory Visualization")
    print("=" * 60)
    
    # Plot individual trajectories
    for i, filepath in enumerate(args.trajectory_files):
        if Path(filepath).exists():
            print(f"📊 Processing: {filepath}")
            
            timestamps, frame_counts, translations, quaternions = parse_trajectory_file(filepath)
            
            print(f"   Frames: {len(translations)}")
            print(f"   Duration: {timestamps[-1] - timestamps[0]:.1f}s")
            print(f"   Frame rate: {len(translations) / (timestamps[-1] - timestamps[0]):.1f} Hz")
            print(f"   Translation range: X[{translations[:, 0].min():.3f}, {translations[:, 0].max():.3f}]")
            print(f"   Translation range: Y[{translations[:, 1].min():.3f}, {translations[:, 1].max():.3f}]")
            print(f"   Translation range: Z[{translations[:, 2].min():.3f}, {translations[:, 2].max():.3f}]")
            
            # Calculate additional metrics
            total_distance = np.sum(np.linalg.norm(np.diff(translations, axis=0), axis=1))
            avg_velocity = total_distance / (timestamps[-1] - timestamps[0])
            drift_distance = np.linalg.norm(translations[-1] - translations[0])
            
            print(f"   Total distance: {total_distance:.3f}m")
            print(f"   Average velocity: {avg_velocity:.3f}m/s")
            print(f"   Drift distance: {drift_distance:.3f}m")
            
            # 3D trajectory plot
            title = f"3D Trajectory: {args.labels[i] if i < len(args.labels) else filepath}"
            fig, ax = plot_3d_trajectory(translations, quaternions, title, colors[i % len(colors)])
            
            output_file = output_dir / f"trajectory_3d_{i+1}.png"
            fig.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
            print(f"   Saved: {output_file}")
            
            # Trajectory analysis
            analysis_fig = plot_trajectory_analysis(translations, quaternions, title)
            analysis_file = output_dir / f"trajectory_analysis_{i+1}.png"
            analysis_fig.savefig(analysis_file, dpi=args.dpi, bbox_inches='tight')
            print(f"   Saved: {analysis_file}")
            
            plt.close(fig)
            plt.close(analysis_fig)
        else:
            print(f"⚠️  File not found: {filepath}")
    
    # Plot comparison if multiple files exist
    existing_files = [f for f in args.trajectory_files if Path(f).exists()]
    if len(existing_files) > 1:
        print(f"\n📈 Creating trajectory comparison...")
        
        existing_labels = [args.labels[i] for i, f in enumerate(args.trajectory_files) if Path(f).exists()]
        comparison_fig, comparison_ax = plot_trajectory_comparison(existing_files, existing_labels, colors)
        
        comparison_file = output_dir / "trajectory_comparison.png"
        comparison_fig.savefig(comparison_file, dpi=args.dpi, bbox_inches='tight')
        print(f"   Saved: {comparison_file}")
        
        plt.close(comparison_fig)
    
    print(f"\n✅ Extended visualization complete! Check the '{output_dir}' directory for plots.")
    print(f"📁 Generated files:")
    for file in output_dir.glob("*.png"):
        print(f"   - {file}")
    
    # Summary of key metrics
    print(f"\n📊 Key Performance Metrics:")
    for i, filepath in enumerate(existing_files):
        timestamps, frame_counts, translations, quaternions = parse_trajectory_file(filepath)
        total_distance = np.sum(np.linalg.norm(np.diff(translations, axis=0), axis=1))
        avg_velocity = total_distance / (timestamps[-1] - timestamps[0])
        drift_distance = np.linalg.norm(translations[-1] - translations[0])
        
        label = args.labels[i] if i < len(args.labels) else Path(filepath).stem
        print(f"   {label}:")
        print(f"     - Distance: {total_distance:.3f}m")
        print(f"     - Velocity: {avg_velocity:.3f}m/s")
        print(f"     - Drift: {drift_distance:.3f}m")
        print(f"     - Frames: {len(translations)}")


if __name__ == "__main__":
    main() 