
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

def main():
    parser = argparse.ArgumentParser(description="Test single joint tracking on SO-101")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--joint", default="elbow_flex", help="Joint to test")
    parser.add_argument("--amplitude", type=float, default=20.0, help="Amplitude of movement in degrees")
    parser.add_argument("--period", type=float, default=2.0, help="Period of sine wave in seconds")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds")
    args = parser.parse_args()

    print(f"Testing joint: {args.joint} on port {args.port}")

    # Initialize Robot
    config = SO101FollowerConfig(
        port=args.port,
        id="so101_test_joint",
        use_degrees=True,
        cameras={}
    )
    robot = SO101Follower(config)
    robot.connect()
    print("Robot connected.")

    try:
        # Get initial position
        obs = robot.get_observation()
        initial_pos = obs[f"{args.joint}.pos"]
        print(f"Initial {args.joint} position: {initial_pos:.2f}")

        # Move to center of sine wave (initial pos)
        # We assume the current position is a safe start or user positioned it.
        # Actually, let's use the current position as the center to avoid jumps.
        center_pos = initial_pos

        t_start = time.time()
        times = []
        targets = []
        actuals = []
        
        print("Starting sine wave tracking...")
        while True:
            t_curr = time.time() - t_start
            if t_curr > args.duration:
                break

            # Compute sine wave target
            # target = center + A * sin(2 * pi * t / T)
            target = center_pos + args.amplitude * np.sin(2 * np.pi * t_curr / args.period)
            
            # Send action
            # We must send commands for all motors, or at least the ones we care about.
            # get_observation gives us current state of others, we can just hold them or send 0 velocity?
            # Safest is to read current and only update the target joint.
            # But reading every loop might be slow? robot.send_action only updates what's in the dict?
            # let's try sending only the target joint.
            
            action = {f"{args.joint}.pos": target}
            robot.send_action(action)
            
            # Wait a bit to govern loop rate (approx 50Hz)
            time.sleep(0.02)
            
            # Read actual
            obs_new = robot.get_observation()
            actual = obs_new[f"{args.joint}.pos"]
            
            times.append(t_curr)
            targets.append(target)
            actuals.append(actual)

    except KeyboardInterrupt:
        print("Test interrupted.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")

        if times:
            # Calculate errors
            targets_arr = np.array(targets)
            actuals_arr = np.array(actuals)
            errors = targets_arr - actuals_arr
            mae = np.mean(np.abs(errors))
            max_err = np.max(np.abs(errors))
            
            print(f"Mean Absolute Error: {mae:.2f} deg")
            print(f"Max Absolute Error: {max_err:.2f} deg")

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(times, targets, label='Target', linestyle='--')
            plt.plot(times, actuals, label='Actual')
            plt.title(f"Tracking Test: {args.joint}\nMAE: {mae:.2f}, Max: {max_err:.2f}")
            plt.xlabel("Time (s)")
            plt.ylabel("Position (deg)")
            plt.legend()
            plt.grid(True)
            filename = f"test_joint_{args.joint}.png"
            plt.savefig(filename)
            print(f"Plot saved to {filename}")

if __name__ == "__main__":
    main()
