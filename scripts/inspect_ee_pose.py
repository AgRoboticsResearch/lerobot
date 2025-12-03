import pinocchio
import numpy as np

def main():
    urdf_path = "/home/hls/codes/lerobot/piper_description/urdf/piper_description.urdf"
    model = pinocchio.buildModelFromUrdf(urdf_path)
    data = model.createData()
    
    # Identify end-effector frame
    target_frame_name = "gripper_base"
    if not model.existFrame(target_frame_name):
        print(f"Frame '{target_frame_name}' not found. Available frames:")
        for f in model.frames:
            print(f"  {f.name}")
        return

    FRAME_ID = model.getFrameId(target_frame_name)
    
    # Home configuration (zeros)
    q0 = pinocchio.neutral(model)
    
    # Compute FK
    pinocchio.framesForwardKinematics(model, data, q0)
    pose = data.oMf[FRAME_ID]
    
    print(f"End-Effector ('{target_frame_name}') Pose at Home Configuration (q=0):")
    print(f"Position (x, y, z): {pose.translation.T}")
    print(f"Rotation Matrix:\n{pose.rotation}")
    
    rpy = pinocchio.rpy.matrixToRpy(pose.rotation)
    print(f"Rotation (Roll, Pitch, Yaw) [rad]: {rpy.T}")
    print(f"Rotation (Roll, Pitch, Yaw) [deg]: {np.degrees(rpy).T}")

if __name__ == "__main__":
    main()
