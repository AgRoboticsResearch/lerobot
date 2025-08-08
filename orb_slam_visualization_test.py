#!/usr/bin/env python3
"""
ORB-SLAM Visualization Test

This script tests ORB-SLAM trajectory visualization to verify it's working correctly.
"""

import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.utils.orb_slam_integration import create_orb_slam_processor, OrbSlamConfig
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OrbSlamVisualizationTest:
    """Test ORB-SLAM trajectory visualization."""
    
    def __init__(self):
        """Initialize the test."""
        # Camera configuration
        self.camera_config = RealSenseCameraConfig(
            serial_number_or_name="031522070877",
            fps=30,
            width=640,
            height=480,
            use_depth=True
        )
        
        # ORB-SLAM configuration
        self.orb_slam_config = OrbSlamConfig(
            max_features=2000,
            output_frequency=30.0,
            enable_visualization=True,
            pose_history_size=100
        )
        
        # Initialize components
        self.camera = None
        self.orb_slam_processor = None
        
        # Trajectory storage
        self.trajectory_points = []
        self.trajectory_times = []
        self.start_time = None
        
        # Visualization
        self.fig = None
        self.ax = None
        self.line = None
        self.point = None
        self.is_running = False
        
        print("✅ ORB-SLAM Visualization Test initialized")
    
    def connect(self):
        """Connect to camera and initialize ORB-SLAM."""
        print("🔌 Connecting to camera...")
        
        try:
            # Connect to camera
            print("📷 Connecting to RealSense camera...")
            self.camera = RealSenseCamera(self.camera_config)
            self.camera.connect()
            print("✅ Camera connected")
            
            # Initialize ORB-SLAM
            print("🎯 Initializing ORB-SLAM...")
            self.orb_slam_processor = create_orb_slam_processor(self.orb_slam_config)
            print("✅ ORB-SLAM initialized")
            
            # Initialize visualization
            self._setup_visualization()
            
            print("✅ All systems connected successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_visualization(self):
        """Setup 3D trajectory visualization."""
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up the plot
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')
        self.ax.set_title('ORB-SLAM Camera Trajectory (Real-time)')
        
        # Set reasonable limits
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-1, 1])
        
        # Add grid
        self.ax.grid(True)
        
        # Initialize trajectory line
        self.line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
        self.point, = self.ax.plot([], [], [], 'ro', markersize=8, label='Current Position')
        
        # Add legend
        self.ax.legend()
        
        plt.tight_layout()
        plt.show(block=False)
    
    def start_visualization(self):
        """Start the trajectory visualization."""
        if not self.is_running:
            self.is_running = True
            self.start_time = time.time()
            
            print("🚀 Starting ORB-SLAM trajectory visualization...")
            print("📷 Move your RealSense camera to see the trajectory!")
            print("🛑 Press Ctrl+C to stop")
            
            try:
                while self.is_running:
                    # Get camera frames
                    color_frame = self.camera.async_read()
                    if color_frame is None:
                        continue
                    
                    # Process with ORB-SLAM
                    frames = {"left": color_frame, "right": color_frame}
                    camera_pose = self.orb_slam_processor.process_camera_frames(frames)
                    
                    if camera_pose is not None:
                        # Extract position
                        position = camera_pose[:3, 3]
                        current_time = time.time() - self.start_time
                        
                        # Store trajectory
                        self.trajectory_points.append(position)
                        self.trajectory_times.append(current_time)
                        
                        # Update visualization
                        self._update_visualization()
                        
                        # Print status
                        print(f"⏱️ Time: {current_time:.2f}s | 📍 Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] | 📊 Points: {len(self.trajectory_points)}")
                    
                    time.sleep(0.033)  # ~30Hz
                    
            except KeyboardInterrupt:
                print("\n🛑 Visualization stopped by user")
            except Exception as e:
                print(f"❌ Error during visualization: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.stop()
    
    def _update_visualization(self):
        """Update the 3D trajectory plot."""
        if len(self.trajectory_points) < 2:
            return
        
        # Convert to numpy arrays
        points = np.array(self.trajectory_points)
        
        # Update trajectory line
        self.line.set_data(points[:, 0], points[:, 1])
        self.line.set_3d_properties(points[:, 2])
        
        # Update current position point
        current_pos = points[-1]
        self.point.set_data([current_pos[0]], [current_pos[1]])
        self.point.set_3d_properties([current_pos[2]])
        
        # Auto-adjust limits if needed
        if len(points) > 10:
            x_range = np.ptp(points[:, 0])
            y_range = np.ptp(points[:, 1])
            z_range = np.ptp(points[:, 2])
            
            if x_range > 0.5 or y_range > 0.5 or z_range > 0.5:
                margin = 0.2
                self.ax.set_xlim([np.min(points[:, 0]) - margin, np.max(points[:, 0]) + margin])
                self.ax.set_ylim([np.min(points[:, 1]) - margin, np.max(points[:, 1]) + margin])
                self.ax.set_zlim([np.min(points[:, 2]) - margin, np.max(points[:, 2]) + margin])
        
        # Update the plot
        plt.draw()
        plt.pause(0.001)
    
    def save_trajectory(self):
        """Save the trajectory data."""
        if len(self.trajectory_points) > 0:
            filename = f"orb_slam_trajectory_{int(time.time())}.npz"
            np.savez(filename, 
                    points=np.array(self.trajectory_points),
                    times=np.array(self.trajectory_times))
            print(f"💾 Trajectory saved to {filename}")
            
            # Calculate trajectory statistics
            points = np.array(self.trajectory_points)
            total_distance = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            max_distance = np.max(np.linalg.norm(points - points[0], axis=1))
            
            print(f"📊 Trajectory Statistics:")
            print(f"   Total points: {len(points)}")
            print(f"   Total distance: {total_distance:.3f} meters")
            print(f"   Max distance from start: {max_distance:.3f} meters")
            print(f"   Duration: {self.trajectory_times[-1]:.2f} seconds")
    
    def stop(self):
        """Stop visualization."""
        self.is_running = False
        print("🛑 Stopping visualization...")
    
    def disconnect(self):
        """Disconnect from all systems."""
        try:
            if self.camera:
                self.camera.disconnect()
                print("🔌 Camera disconnected")
            
            # Save trajectory
            self.save_trajectory()
            
            # Close plot
            if self.fig:
                plt.close(self.fig)
            
            print("✅ All systems disconnected")
            
        except Exception as e:
            print(f"❌ Error during disconnection: {e}")


def main():
    """Main function."""
    print("🎯 ORB-SLAM Trajectory Visualization Test")
    print("=" * 50)
    
    # Create test
    test = OrbSlamVisualizationTest()
    
    try:
        # Connect to systems
        if not test.connect():
            print("❌ Failed to connect to systems")
            return
        
        # Start visualization
        test.start_visualization()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        test.disconnect()


if __name__ == "__main__":
    main() 