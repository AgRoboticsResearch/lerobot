import time
import argparse
import numpy as np

from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.teleop_phone import Phone

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone-os", type=str, default="ios", choices=["ios", "android"], help="Phone OS (ios or android)")
    args = parser.parse_args()

    phone_os_enum = PhoneOS.IOS if args.phone_os == "ios" else PhoneOS.ANDROID
    teleop_config = PhoneConfig(phone_os=phone_os_enum)
    teleop_device = Phone(teleop_config)

    print("Connecting to phone...")
    teleop_device.connect()
    print("Connected.")
    
    # Wait for calibration
    if not teleop_device.is_calibrated:
        print("Please press and hold B1 (or equivalent) to enable and calibrate...")
        # Connection method usually handles calibration wait for iOS/Android logic 
        # but let's loop until we see valid data
    
    try:
        while True:
            action = teleop_device.get_action()
            
            # Action is empty if no pose or not calibrated (though we passed calibration check above)
            if not action:
                print("No Data (Check ARKit/Network)")
                time.sleep(0.1)
                continue

            is_enabled = action.get("phone.enabled", False)
            raw = action.get("phone.raw_inputs", {})
            b1 = raw.get("b1", 0)
            

            # Print position if available, regardless of enabled state
            pos = action.get("phone.pos")
            rot = action.get("phone.rot")

            if pos is not None:
                status_str = f"{'ENABLED ' if is_enabled else 'DISABLED'} (B1={b1}) | Pos(m): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
                if rot:
                    rvec = rot.as_rotvec()
                    status_str += f" | RotVec(rad): [{rvec[0]:.3f}, {rvec[1]:.3f}, {rvec[2]:.3f}]"
                print(status_str)
            else:
                print(f"No Pose Data (B1={b1})")
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        teleop_device.disconnect()

if __name__ == "__main__":
    main()
