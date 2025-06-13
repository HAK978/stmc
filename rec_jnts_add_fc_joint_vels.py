import os
import numpy as np
import torch

def foot_detect_exact(positions: torch.Tensor, fid_l: list, fid_r: list, thres: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact foot contact detection copied from motion_representation notebook.
    """
    velfactor = torch.tensor([thres, thres], device=positions.device)

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).float()

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).float()

    return feet_l, feet_r

def recover_joints_and_extend_rifke_to_275(rifke_features: torch.Tensor, feet_thres: float = 0.002):
    """
    Recover joints, compute local velocities and foot contacts, and return extended RIFKE features (275D).
    """
    joints_local_flatten = rifke_features[:, -69:]
    joints_local = joints_local_flatten.reshape(rifke_features.shape[0], 23, 3)
    root = torch.zeros((joints_local.shape[0], 1, 3), device=joints_local.device)
    joints_full = torch.cat((root, joints_local), dim=1)  # (frames, 24, 3)

    # Local joint velocities (excluding pelvis)
    joints_body = joints_full[:, 1:23]  # 22 joints
    local_velocities = joints_body[1:] - joints_body[:-1]  # (frames-1, 22, 3)
    local_vel_flattened = local_velocities.reshape(local_velocities.shape[0], -1)  # (frames-1, 66)

    # Foot contact detection
    fid_l = [7, 10]  # left_ankle, left_foot
    fid_r = [8, 11]  # right_ankle, right_foot
    feet_l, feet_r = foot_detect_exact(joints_full, fid_l, fid_r, feet_thres)
    feet_contact = torch.cat([feet_l, feet_r], dim=1)  # (frames-1, 4)

    rifke_cut = rifke_features[:-1]  # align length
    extended_features = torch.cat([rifke_cut, local_vel_flattened, feet_contact], dim=-1)  # (frames-1, 275)

    return joints_full, extended_features

def process_all_rifke_files(rifke_root_dir, recovered_joints_root, extended_features_root):
    rifke_root_dir = os.path.normpath(rifke_root_dir)
    recovered_joints_root = os.path.normpath(recovered_joints_root)
    extended_features_root = os.path.normpath(extended_features_root)

    for root, dirs, files in os.walk(rifke_root_dir):
        for file in files:
            if file.endswith(".npy"):
                rifke_path = os.path.join(root, file)
                relative_path = os.path.relpath(rifke_path, rifke_root_dir)

                save_joints_path = os.path.join(recovered_joints_root, relative_path)
                save_features_path = os.path.join(extended_features_root, relative_path)

                try:
                    rifke_features = torch.from_numpy(np.load(rifke_path)).float()
                    joints, extended_275 = recover_joints_and_extend_rifke_to_275(rifke_features)

                    os.makedirs(os.path.dirname(save_joints_path), exist_ok=True)
                    os.makedirs(os.path.dirname(save_features_path), exist_ok=True)

                    np.save(save_joints_path, joints.cpu().numpy())
                    np.save(save_features_path, extended_275.cpu().numpy())

                    print(f"✅ Processed: {relative_path}")
                except Exception as e:
                    print(f"❌ Failed: {relative_path} due to {e}")

# ------------------ PATHS (EDIT HERE) ------------------

rifke_base = r"C:\Users\harsh\GitHub\stmc\datasets\motions\AMASS_20.0_fps_nh_smplrifke"
recovered_joints_base = r"C:\Users\harsh\GitHub\stmc\datasets\motions\recovered_joints"
extended_features_base = r"C:\Users\harsh\GitHub\stmc\datasets\motions\extended_features_275"

process_all_rifke_files(rifke_base, recovered_joints_base, extended_features_base)
