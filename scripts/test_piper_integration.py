import sys
import os
import time
import numpy as np

# Add lerobot_robot_piper to path so we can import Piper
sys.path.append(os.path.join(os.path.dirname(__file__), "../lerobot_robot_piper"))

from lerobot_robot_piper.piper import Piper
from lerobot_robot_piper.config_piper import PiperConfig

def main():
    print("Initializing Piper...")
    # Assume CAN0 is the interface, matching default
    config = PiperConfig(can_interface='can0', include_gripper=True)
    robot = Piper(config)
    
    print("Connecting to robot...")
    try:
        robot.connect()
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Connected!")
    
    # Read initial state
    obs = robot.get_observation()
    print("Initial Observation:")
    for k, v in obs.items():
        if "pos" in k:
            print(f"  {k}: {v:.2f}")

    # Safety check: avoid moving if limits are unknown or weird
    # (Assuming SDK handled limits, but good to be cautious)
    
    # Target: Rotation of joint_6 (wrist roll) by +10 degrees
    # joint_6 is index 5
    # Let's get the current joint positions in degrees (via aliases or direct names)
    current_joints = []
    for i in range(1, 7):
        key = f"joint_{i}.pos"
        val = obs.get(key, 0.0)
        current_joints.append(val)
        
    print(f"Current Joints: {current_joints}")
    
    target_joints = list(current_joints)
    # Move joint 6 by 10 degrees, capping at limit (180 approx)
    target_joints[5] = min(180.0, target_joints[5] + 10.0)
    
    print(f"Target Joints:  {target_joints}")
    print("Sending command for 2 seconds...")

    # Action dictionary format expected by Piper.send_action
    # It expects { "joint_name.pos": value_deg } if use_degrees=True (default)
    action = {}
    for i, val in enumerate(target_joints):
        action[f"joint_{i+1}.pos"] = val
    
    if "gripper.pos" in obs:
        action["gripper.pos"] = obs["gripper.pos"]

    start_time = time.time()
    duration = 2.0
    hz = 50
    dt = 1.0 / hz
    
    try:
        while time.time() - start_time < duration:
            robot.send_action(action)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        print("Disconnecting...")
        robot.disconnect()
        print("Done.")

if __name__ == "__main__":
    main()
