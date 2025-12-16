
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

def main():
    parser = argparse.ArgumentParser(description="Test single joint step response on SO-101")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--joint", default="elbow_flex", help="Joint to test")
    parser.add_argument("--step_size", type=float, default=20.0, help="Step size in degrees")
    parser.add_argument("--duration", type=float, default=3.0, help="Recording duration in seconds")
    args = parser.parse_args()

    print(f"Testing step response: {args.joint} on port {args.port}, Step: {args.step_size} deg")

    # Initialize Robot
    config = SO101FollowerConfig(
        port=args.port,
        id="so101_step_test",
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

        # Target position
        target_pos = initial_pos + args.step_size
        print(f"Target position: {target_pos:.2f}")

        # Wait a moment
        time.sleep(1.0)
        
        times = []
        targets = []
        actuals = []
        
        print("Sending step command and recording...")
        t_start = time.time()
        
        # Send step command ONCE
        action = {f"{args.joint}.pos": target_pos}
        robot.send_action(action)
        
        while True:
            t_curr = time.time() - t_start
            if t_curr > args.duration:
                break
            
            # Read actual
            obs_new = robot.get_observation()
            actual = obs_new[f"{args.joint}.pos"]
            
            times.append(t_curr)
            targets.append(target_pos)
            actuals.append(actual)
            
            # High frequency recording (approx 100Hz)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Test interrupted.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")

        if times:
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(times, targets, label='Target (Step)', linestyle='--')
            plt.plot(times, actuals, label='Actual Response')
            plt.title(f"Step Response: {args.joint}\nStep: {args.step_size} deg")
            plt.xlabel("Time (s)")
            plt.ylabel("Position (deg)")
            plt.legend()
            plt.grid(True)
            filename = f"step_response_{args.joint}.png"
            plt.savefig(filename)
            print(f"Plot saved to {filename}")

if __name__ == "__main__":
    main()
