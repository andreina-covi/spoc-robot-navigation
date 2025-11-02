import numpy as np
import cv2

# ---------- Intrinsics helpers ----------
def intrinsics_from_fov(W: int, H: int, fov_v_deg: float):
    """Compute (fx, fy, cx, cy) from vertical FOV and image size."""
    f_y = (H / 2.0) / np.tan(np.deg2rad(fov_v_deg) / 2.0)
    f_x = f_y * (W / H)  # square pixels
    c_x, c_y = W / 2.0, H / 2.0
    K = np.array([[f_x, 0.0, c_x],
                  [0.0, f_y, c_y],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K

# ---------- Rotations (Unity/AI2-THOR convention) ----------
def rot_matrix_camera_to_world(yaw_deg: float, pitch_deg: float, roll_deg: float):
    """
    Build camera->world rotation given yaw (around +Y), pitch (around +X), roll (around +Z).
    Composition matches common Unity convention: R_cw = R_yaw * R_pitch * R_roll.
    """
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])

    Ry = np.array([[ np.cos(y), 0, np.sin(y)],
                   [ 0,        1, 0       ],
                   [-np.sin(y), 0, np.cos(y)]], dtype=np.float64)

    Rx = np.array([[1, 0,         0        ],
                   [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p),  np.cos(p)]], dtype=np.float64)

    Rz = np.array([[ np.cos(r), -np.sin(r), 0],
                   [ np.sin(r),  np.cos(r), 0],
                   [ 0,          0,         1]], dtype=np.float64)

    R_cw = Ry @ Rx @ Rz
    return R_cw

def extrinsics_world_to_camera(cam_pos_xyz, yaw_deg, pitch_deg, roll_deg):
    """
    Return (R_wc, tvec) so that X_cam = R_wc * X_world + tvec.
    cam_pos_xyz is (Cx, Cy, Cz) in world.
    """
    C = np.asarray(cam_pos_xyz, dtype=np.float64).reshape(3, 1)
    R_cw = rot_matrix_camera_to_world(yaw_deg, pitch_deg, roll_deg)
    R_wc = R_cw.T
    tvec = -R_wc @ C
    return R_wc, tvec

def project_point(point3D, cam_pos, cam_rot_deg, fov_v_deg, img_w, img_h):
    # --- Step 1. Camera intrinsics
    f_y = (img_h / 2) / np.tan(np.deg2rad(fov_v_deg) / 2)
    f_x = f_y * (img_w / img_h)
    c_x, c_y = img_w / 2, img_h / 2

    # --- Step 2. Rotation (yaw=y, pitch=x, roll=z) in degrees
    pitch, yaw, roll = np.deg2rad(cam_rot_deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]])
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [ 0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll),  np.cos(roll), 0],
                   [0, 0, 1]])
    # Unity / AI2-THOR order ≈ Ry * Rx * Rz
    R = Ry @ Rx @ Rz

    # --- Step 3. Transform world → camera coordinates
    Pw = np.array(point3D).reshape(3,1)
    C  = np.array(cam_pos).reshape(3,1)
    Pc = R.T @ (Pw - C)   # camera coordinates
    print("Pc", Pc)
    Xc, Yc, Zc = Pc.flatten()

    if Zc <= 0:
        return None  # behind camera

    # --- Step 4. Project to pixel
    u = f_x * (Xc / Zc) + c_x
    v = f_y * (Yc / Zc) + c_y

    return (float(u), float(v))


# # ---------- Projection ----------
# def project_world_points_to_image(points_xyz, cam_pos_xyz, yaw_deg, pitch_deg, roll_deg,
#                                   W, H, fov_v_deg, dist_coeffs=None):
#     """
#     points_xyz: (N,3) array of world points
#     Returns:
#       uv: (N,2) float32 pixel coords
#       in_front: (N,) bool  (Z_cam > 0)
#       in_image: (N,) bool  (0<=u<W, 0<=v<H and in_front)
#     """
#     points_xyz = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
#     K = intrinsics_from_fov(W, H, fov_v_deg)
#     R_wc, tvec = extrinsics_world_to_camera(cam_pos_xyz, yaw_deg, pitch_deg, roll_deg)

#     # OpenCV wants rvec (Rodrigues) and tvec as (3,1)
#     rvec, _ = cv2.Rodrigues(R_wc)   # R_wc -> rvec
#     tvec = tvec.reshape(3, 1)
#     if dist_coeffs is None:
#         dist_coeffs = np.zeros((5, 1), dtype=np.float64)  # assume no distortion (simulators)

#     # Project
#     imgpts, _ = cv2.projectPoints(points_xyz, rvec, tvec, K, dist_coeffs)
#     uv = imgpts.reshape(-1, 2).astype(np.float32)

#     # Also compute Z_cam to check "in front"
#     # X_cam = R_wc * X_world + t
#     Xw = points_xyz.T  # (3,N)
#     Xc = (R_wc @ Xw) + tvec  # (3,N)
#     Zc = Xc[2, :].flatten()
#     in_front = Zc > 0

#     u, v = uv[:, 0], uv[:, 1]
#     in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
#     in_image = in_front & in_bounds

#     return uv, in_front, in_image

# ---------- Example ----------
if __name__ == "__main__":
    # Image size + vertical FOV (replace with your real values)
    W, H = 396, 224
    FOV_V = 59

    # Camera pose from your metadata (example numbers)
    cam_pos = (5.85, 0.9, 2.55)         # (x,y,z) in meters
    yaw, pitch, roll = 0, 210, 0 # degrees

    # # Some world points to project (agent/object centers)
    # points = np.array([
    #     [5.85, 0.9, 2.55]  # object A
    #     # [1.0, 1.2, 2.0],  # straight ahead
    #     # [0.0, 0.5, 2.2],  # object B
    # ], dtype=np.float64)

    # uv, in_front, in_image = project_world_points_to_image(
    #     points, cam_pos, yaw, pitch, roll, W, H, FOV_V
    # )

    # for i, p in enumerate(points):
    #     print(f"Point {p} -> uv={uv[i]}, in_front={in_front[i]}, in_image={in_image[i]}")

    # cameraMatrix = intrinsics_from_fov(W, H, FOV_V)
    # distCoeffs = np.zeros((3, 1))
    # imgpts, _ = cv2.projectPoints((5.85, 0.9, 2.55), rvec, tvec, cameraMatrix)
    # print(cameraMatrix)
    # u, v = project_point((0, 0, 0), cam_pos, (pitch, yaw, roll), FOV_V, W, H)
    # print(u, v)
    