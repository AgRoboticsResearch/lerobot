import time
import argparse
import numpy as np
import copy

from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone-os", type=str, default="ios", choices=["ios", "android"], help="Phone OS (ios or android)")
    args = parser.parse_args()

    phone_os_enum = PhoneOS.IOS if args.phone_os == "ios" else PhoneOS.ANDROID
    teleop_config = PhoneConfig(phone_os=phone_os_enum)
    teleop_device = Phone(teleop_config)
    
    # Initialize the processor
    mapper = MapPhoneActionToRobotAction(platform=phone_os_enum)

    print("Connecting to phone...")
    teleop_device.connect()
    print("Connected.")
    
    # Wait for calibration
    if not teleop_device.is_calibrated:
        print("Please press and hold B1 (or equivalent) to enable and set origin...")
    
    last_dump = 0.0
    try:
        while True:
            # Get raw phone action
            action_in = teleop_device.get_action()
            
            if not action_in:
                print("No Data")
                time.sleep(0.1)
                continue

            # Check raw inputs for button state BEFORE processing (as processor pops them)
            raw_inputs = action_in.get("phone.raw_inputs", {})
            b1 = raw_inputs.get("b1", 0)
            
            # Debug: Print all raw inputs if B1 seems missing but user claims pressing
            if b1 == 0:
                now = time.time()
                if now - last_dump > 1.0:
                    print(f"raw_inputs keys={list(raw_inputs.keys())} values={raw_inputs}")
                    last_dump = now
            
            pos_in = action_in.get("phone.pos")
            # Create a deep copy because the processor modifies the dict in-place
            action_out = mapper.action(copy.deepcopy(action_in))
            
            is_enabled = action_out.get("enabled", False)
            
            # Extract mapped targets
            tx = action_out.get("target_x", 0.0)
            ty = action_out.get("target_y", 0.0)
            tz = action_out.get("target_z", 0.0)
            twx = action_out.get("target_wx", 0.0)
            twy = action_out.get("target_wy", 0.0)
            twz = action_out.get("target_wz", 0.0)
            
            if pos_in is not None:
                # Compare Input -> Output
                # Format: In[X,Y,Z] -> Out[X,Y,Z]
                print(f"{'ENABLED ' if is_enabled else 'DISABLED'} (B1={b1})")
                print(f"  Pos: [{pos_in[0]:.3f}, {pos_in[1]:.3f}, {pos_in[2]:.3f}] -> Target: [{tx:.3f}, {ty:.3f}, {tz:.3f}]")
                print(f"  Rot: [..vec..]            -> TargetRot: [{twx:.3f}, {twy:.3f}, {twz:.3f}]")
                print("-" * 40)
            else:
                print(f"No Pose Data (B1={b1})")
                
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        teleop_device.disconnect()

if __name__ == "__main__":
    main()
