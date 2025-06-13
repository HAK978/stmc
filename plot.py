# import numpy as np
# import cv2
# import os
# from tqdm import tqdm
# import argparse

# # SMPL-24 Joint Connections with anatomical coloring
# SMPL_BONE_GROUPS = {
#     'pelvis': [(0,1), (0,2), (0,3)],
#     'legs': [(1,4), (4,7), (7,10), (2,5), (5,8), (8,11)],
#     'spine': [(3,6), (6,9), (9,12)],
#     'arms': [(12,13), (13,16), (16,18), (18,20), (20,22),
#              (12,14), (14,17), (17,19), (19,21), (21,23)],
#     'head': [(12,15)]
# }
# BONE_COLORS = {
#     'pelvis': (255, 255, 0),  # Yellow
#     'legs': (255, 165, 0),    # Orange
#     'spine': (0, 255, 255),   # Cyan
#     'arms': (255, 0, 255),    # Magenta
#     'head': (0, 255, 0)       # Green
# }

# def load_motion_data(file_path):
#     """Load .npy file and validate shape"""
#     data = np.load(file_path)
#     if data.shape[1] not in [205, 24*3]:
#         raise ValueError(f"Expected (frames, 205) or (frames, 72), got {data.shape}")
#     return data

# def convert_rifke_to_joints(rifke_data):
#     """Convert SMPL-RIFKE (205-D) to 24 joint positions"""
#     poses = rifke_data[:, :72].reshape(-1, 24, 3)
#     poses[:, 22] = poses[:, 20]  # Left hand
#     poses[:, 23] = poses[:, 21]  # Right hand
#     return poses

# def normalize_joints(joints):
#     """Center and scale joints for visualization"""
#     joints = joints.copy()
#     # Center at pelvis
#     joints -= joints[:, 0:1, :]
#     # Scale to fit frame
#     y_range = np.max(joints[:, :, 1]) - np.min(joints[:, :, 1])
#     scale = 500 / y_range if y_range > 0 else 1.0
#     return joints * scale

# def render_pro_skeleton(pose_data, output_file=None, fps=60):
#     """Professional-quality skeleton rendering"""
#     try:
#         # Convert and normalize
#         if pose_data.shape[1] == 205:
#             joints = convert_rifke_to_joints(pose_data)
#         else:
#             joints = pose_data
#         joints = normalize_joints(joints)

#         # Video setup
#         if output_file:
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_file, fourcc, fps, (1280, 720))

#         for frame in tqdm(range(len(joints)), desc="Rendering"):
#             img = np.zeros((720, 1280, 3), dtype=np.uint8)
#             frame_joints = joints[frame]

#             # Convert to screen coords
#             x = (frame_joints[:, 0] + 640).astype(int)
#             y = (360 - frame_joints[:, 1]).astype(int)  # Flip Y

#             # Draw bones
#             for group, connections in SMPL_BONE_GROUPS.items():
#                 for i, j in connections:
#                     if (0 <= x[i] < 1280 and 0 <= y[i] < 720 and
#                         0 <= x[j] < 1280 and 0 <= y[j] < 720):
#                         cv2.line(img, (x[i], y[i]), (x[j], y[j]), 
#                                 BONE_COLORS[group], 2)

#             # Draw joints
#             for xi, yi in zip(x, y):
#                 if 0 <= xi < 1280 and 0 <= yi < 720:
#                     cv2.circle(img, (xi, yi), 4, (255, 255, 255), -1)

#             if output_file:
#                 out.write(img)
#             else:
#                 cv2.imshow('Skeleton', img)
#                 if cv2.waitKey(1) == 27: break

#         if output_file:
#             out.release()
#             print(f"Saved {os.path.getsize(output_file)/1024:.1f} KB video")

#     except Exception as e:
#         print(f"Error: {str(e)}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("file", help="Path to .npy motion file")
#     parser.add_argument("--output", help="Output video file")
#     args = parser.parse_args()

#     try:
#         data = load_motion_data(args.file)
#         print(f"Loaded motion data: {data.shape}")
#         render_pro_skeleton(data, args.output)
#     except Exception as e:
#         print(f"Failed: {str(e)}")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# ===== USER CONFIGURATION =====
file_path = r"C:\Users\harsh\GitHub\stmc\datasets\motions\AMASS_20.0_fps_nh_smpljoints_neutral_nobetas\ACCAD\Male2MartialArtsExtended_c3d\Form 1_poses.npy"  # Raw string
output_file = "martial_arts_form.mp4"     
view_angle = "front"  # "side", "top", or "free"     
playback_speed = 0.5          
# =============================

def load_motion_data(path):
    return np.load(path)  # Assumes data is already in (frames, joints, 3) format

def animate_motion(pose_data):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # View setup
    if view_angle == "side":
        ax.view_init(elev=10, azim=90)
    elif view_angle == "top":
        ax.view_init(elev=89, azim=-90)
    elif view_angle == "front":
        ax.view_init(elev=10, azim=0)
    
    connections = [
        (0,1),(0,2),(0,3),(1,4),(4,7),(7,10),
        (2,5),(5,8),(8,11),(3,6),(6,9),(9,12),
        (12,13),(13,16),(16,18),(18,20),
        (12,14),(14,17),(17,19),(19,21),
        (12,15)
    ]
    
    
    def update(frame):
        ax.clear()
        frame_data = pose_data[frame]

        # Centering and ground alignment
        min_z = np.min(pose_data[:, :, 2])
        frame_data = frame_data.copy()
        frame_data[:, 2] -= min_z  # Move ground to z=0

        # Dynamic camera angle for 'free' mode
        if view_angle == "free":
            ax.view_init(elev=20, azim=(frame * 2) % 360)
        elif view_angle == "side":
            ax.view_init(elev=10, azim=90)
        elif view_angle == "top":
            ax.view_init(elev=89, azim=-90)

        # Set axis limits
        max_range = np.max(np.abs(pose_data - pose_data[:,0:1,:])) * 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range * 2])

        # Draw joints
        ax.scatter(frame_data[:,0], frame_data[:,1], frame_data[:,2], c='red', s=40)

        # Draw bones
        for (i, j) in connections:
            ax.plot([frame_data[i, 0], frame_data[j, 0]],
                    [frame_data[i, 1], frame_data[j, 1]],
                    [frame_data[i, 2], frame_data[j, 2]],
                    'b-', linewidth=2)

        ax.set_title(f"Frame {frame+1}/{len(pose_data)}")
        ax.axis('off')  # Optional: turn off axis lines


    ani = FuncAnimation(fig, update, frames=len(pose_data), 
                       interval=50/playback_speed)  # Slower playback
    
    if output_file:
        ani.save(output_file, writer='ffmpeg', fps=30*playback_speed)
        print(f"Saved to {output_file}")
    else:
        plt.show()

# Run
try:
    motion_data = load_motion_data(file_path)
    print(f"Data shape: {motion_data.shape}")  # Verify (frames, joints, 3)
    animate_motion(motion_data)
except Exception as e:
    print(f"Error: {str(e)}")