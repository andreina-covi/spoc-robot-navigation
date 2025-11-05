import numpy as np
import cv2
import seaborn as sns

def project_3d_to_2d(world_coords, K_matrix, R_matrix, t_vector):
    """
    Projects 3D world coordinates to 2D image coordinates.

    Args:
        world_coords (np.array): A 3xN array of 3D world coordinates (x, y, z).
        K_matrix (np.array): The 3x3 camera intrinsic matrix.
        R_matrix (np.array): The 3x3 camera rotation matrix (extrinsic).
        t_vector (np.array): The 3x1 camera translation vector (extrinsic).

    Returns:
        np.array: A 2xN array of 2D image coordinates (u, v).
    """
    print(world_coords.shape, K_matrix.shape, R_matrix.shape, t_vector.shape)
    # Convert world coordinates to homogeneous coordinates
    world_coords_homog = np.vstack((world_coords, np.ones((1, world_coords.shape[1]))))

    # Construct the extrinsic matrix (Rotation and Translation)
    extrinsic_matrix = np.hstack((R_matrix, t_vector))

    # Project 3D world coordinates to camera coordinates
    camera_coords_homog = np.dot(extrinsic_matrix, world_coords_homog)

    # Project camera coordinates to image plane using intrinsic matrix
    image_coords_homog = np.dot(K_matrix, camera_coords_homog)

    # Normalize homogeneous coordinates to get 2D pixel coordinates
    u = image_coords_homog[0, :] / image_coords_homog[2, :]
    v = image_coords_homog[1, :] / image_coords_homog[2, :]

    return np.vstack((u, v))


# --- Intrinsics from vertical FOV ---
def intrinsics_from_fov(W, H, fov_h, fov_v):
    fy = (H/2.0) / np.tan(np.deg2rad(fov_v)/2.0)
    fx = (W/2.0) / np.tan(np.deg2rad(fov_h)/2.0)
    fx = fy * (W/H)
    cx, cy = W/2.0, H/2.0
    return fx, fy, cx, cy

# --- World -> Agent-local (Unity/THOR axes: x right, y up, z forward) ---
def world_to_local(agent_pos, agent_rot_deg, object_pos):
    # agent_rot_deg = (pitch_x, yaw_y, roll_z) in degrees
    pitch, yaw, roll = np.deg2rad(agent_rot_deg)
    Rx = np.array([[1,0,0],
                   [0,np.cos(pitch),-np.sin(pitch)],
                   [0,np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[ np.cos(yaw),0,np.sin(yaw)],
                   [0,1,0],
                   [-np.sin(yaw),0,np.cos(yaw)]])
    Rz = np.array([[np.cos(roll),-np.sin(roll),0],
                   [np.sin(roll), np.cos(roll),0],
                   [0,0,1]])
    R_cw = Ry @ Rx @ Rz         # camera->world
    R_wc = R_cw.T               # world->camera
    Pw = np.asarray(object_pos).reshape(3,1)
    C  = np.asarray(agent_pos).reshape(3,1)
    Pl = (R_wc @ (Pw - C)).flatten()  # local (Xl,Yl,Zl)
    # return R_wc, Pw, C, Pl
    return Pl  # (x_l, y_l, z_l)

# --- Local -> pixel (u,v) using pinhole ---
def projection_with_local_vector(local_xyz, W, H, fov_h, fov_v):
    xl, yl, zl = local_xyz
    if zl <= 0:
        return None  # behind camera
    fx, fy, cx, cy = intrinsics_from_fov(W, H, fov_h, fov_v)
    u = fx * (xl / zl) #+ cx
    v = fy * (yl / zl) #+ cy
    # if 0 <= u < W and 0 <= v < H:
    #     return (float(u), float(v))
    return float(u), float(v)

def projection_with_angles(fx, fy, alpha, betha, cx, cy):
    u = fx * np.tan(np.deg2rad(alpha)) + cx
    v = fy * np.tan(np.deg2rad(betha)) + cy
    return float(u), float(v)

def calculate_angle(coord1, coord2):
    angle = np.arctan2(coord1, coord2)
    return np.degrees(angle)
# # --- Convenience: world -> pixel directly ---
# def world_to_pixel(agent_pos, agent_rot_deg, point_world, W, H, fov_v_deg):
#     local = world_to_local(agent_pos, agent_rot_deg, point_world)
#     return local_to_pixel(local, W, H, fov_v_deg)

# --- Draw helper ---
def draw_on_frame(frame_bgr, uvs, W, H, palette):
    if uvs is None:
        return frame_bgr
    # u, v = int(round(uv[0])), int(round(uv[1]))
    out = frame_bgr.copy()
    cv2.circle(out, (0, 0), 6, (255, 0, 0), -1)
    cv2.circle(out, (W//2, H//2), 6, (0, 0, 255), -1)
    for i, (u, v) in enumerate(uvs):
        u, v = int(round(u)), int(round(v))
        (r, g, b) = palette[i%10]
        (r, g, b) = (int(255*r), int(255*g), int(255*b))
        cv2.circle(out, (u, v), 6, (r, g, b), -1)
        cv2.putText(out, str(i), (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # if label:
    #     cv2.putText(out, label, (u+8, v-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    # optional: draw a line from image center (agent’s optical center) to the object
    # H, W = frame_bgr.shape[:2]
    # cv2.line(out, (W//2, H//2), (u, v), color, 1)
    return out

def get_fov(val, f):
    fov = 2 * np.arctan(val/(2*f))
    return np.degrees(fov)

def get_other_fov(H, W, fov):
    fov_other = 2 * np.arctan((H / W) * np.tan(np.deg2rad(fov / 2)))
    return np.degrees(fov_other)

def calculate_focal_length(val, fov):
    f = val / (2 * np.tan(np.deg2rad(fov/2)))
    return f

def get_center_point(bbox):
    xc = np.average([bbox[0], bbox[2]])
    yc = np.average([bbox[1], bbox[3]])
    return xc, yc

# -------- Example usage --------
if __name__ == "__main__":
    # Image & FOV
    W, H = 396, 224
    FOV = 59  # vertical FOV in degrees

    # Agent pose (global)
    agent_pos = (5.85, 0.9, 2.55)          # (ag-pos-x, ag-pos-y, ag-pos-z)
    agent_rot = (0.0, 210.0, 0.0)         # (pitch, yaw, roll) deg

    # fx, fy, cx, cy = intrinsics_from_fov(W, H, FOV_V)
    # K = np.array([
    #     [fx, 0, cx],
    #     [0, fy, cy],
    #     [0, 0, 1]
    # ])
    # R = np.eye(3)
    
    # Object global position
    obj_worlds = np.array([
        [5.06, 0.94, 1.91],
        [5.72, 0.97, 1.84],
        [5.5, 0.01, 1.86],
        [4.79, 0.94, 1.81],
        [5.94, 0.96, 1.96],
        [5.28, 0.94, 1.71]
    ])
    obj_bboxes = np.array([
        [242, 114, 300, 142],
        [96, 135, 162, 170],
        [0, 96, 349, 223],
        [281, 106, 288, 109],
        [17, 142, 87, 223],
        [202, 92, 218, 121]
    ])
    # obj_worlds = np.array([
    #                     #   [10.55, 0.0, 1.85], # chair-dining
    #                     #   [9.76, 0.78, 1.72],  # Objawrench
    #                     #   [9.97, 0.79, 1.95],   # ObjaToken
    #                       [10.16, 0.78, 1.94],  # Plate
    #                     #   [10.16, 0.78, 2.17],  # ObjaCoffePot
    #                     #   [9.86, 0.78, 1.72],   # ObjaGarlic
    #                       [10.06, 0.83, 1.72],  # Bread
    #                       [10.06, 0.78, 1.5],   #Vase
    #                      [9.86, 0.85, 2.17],   # Lettuce
    #                     #   [10.06, 0.0, 2.17]
    #                     ])  # DiningTable
    # R_wc, Pw, C, Pl = world_to_local(agent_pos, agent_rot, obj_worlds[0])
    # t = -(R_wc @ C)
    
    # obj_worlds = obj_worlds.T
    # projected_2d_points = project_3d_to_2d(obj_worlds.T, K, R, t)
    # print(projected_2d_points)


    # Pl = world_to_local(agent_pos, agent_rot, obj_worlds)
    # print(Pl)

    # fx = calculate_focal_length(W, FOV)
    # print("fx:", fx)
    # fov_v = get_other_fov(H, W, FOV)
    # print("fov_v:", fov_v)
    # fy = calculate_focal_length(H, fov_v)
    # print("fy", fy)
    # print("fov_v:", get_fov(H, fy))
    # fov_h = get_fov(W, fx)
    # print("fov_h", fov_h)
    # # fov_v = get_fov(H, fy)
    # # print(fov_v)
    # # fxp, fyp, cx, cy = intrinsics_from_fov(W, H, FOV)
    # # print(fxp, fyp, cx, cy)

    # u, v = projection_with_local_vector(Pl, W, H, fov_h, fov_v)
    # print("Considering FOV como FOV_h=59", u, v)

    # # alpha = calculate_angle(Pl[0], Pl[2])
    # # x_prima = (W * alpha) / fov_h
    # # betha = calculate_angle(Pl[1], Pl[2])
    # # y_prima = (H * betha) / fov_v
    # # print(x_prima, y_prima)
    print("Now considering FOV default is vertical")
    fy = calculate_focal_length(H, FOV)
    print("fy", fy)
    fov_h = get_other_fov(W, H, FOV)
    print("fov_h", fov_h)
    fx = calculate_focal_length(W, fov_h)
    print("fx", fx)
    fov_v = get_fov(H, fy)
    print("fov_v", fov_v)
    # u_p, v_p = projection_with_local_vector(Pl, W, H, fov_h, fov_v)
    # print("Considering FOV_v=59", u_p, v_p)

    uvs = []
    for i, obj_world in enumerate(obj_worlds):
        x_l, y_l, z_l = world_to_local(agent_pos, agent_rot, obj_world)
        # x_l, y_l, z_l = obj_world
        alpha = calculate_angle(x_l, z_l)
        betha = calculate_angle(y_l, z_l)
        u, v = get_center_point(obj_bboxes[i])
        # u_p, v_p = projection_with_angles(fx, fy, alpha, betha, cx, cy)
        u_p, v_p = projection_with_local_vector((x_l, y_l, z_l), W, H, fov_h, fov_v)
        # u = u + (W//2)
        # v = -v + (H // 2)
        print(f"Alpha: {alpha}, Betha: {betha}")
        # print(f"Answer with angles: u:{u}, v:{v}")
        print(f"Answer with local vector: u:{u}, v:{v}")
        print(f"Answer with bboxes: u_p:{u_p}, v_p:{v_p}")
        uvs.append((u, v))
    # # Compute local coords and angles (if you want them)
    # azimuth_deg   = np.degrees(np.arctan2(x_l, z_l))   # +right/-left
    # elevation_deg = np.degrees(np.arctan2(y_l, z_l))   # +up/-down
    # print("Local (x,y,z):", (x_l, y_l, z_l))
    # print("Angles (azimuth,elevation):", (azimuth_deg, elevation_deg))

    # # # Pixel projection
    # # uv = local_to_pixel((x_l, y_l, z_l), W, H, FOV_V)
    # # print("Pixel (u,v):", uv)
    # uvs = [(u+(3 * W/4), v+(H/2))]

    # //////////////////////
    # uvs = [(x_prima + (3 * W / 4), y_prima + (3 * H / 4))]
    # uvs = [(u+W, v), (u_p+W, v_p)]
    # print(uvs)
    palette = sns.color_palette("Set2", n_colors=10)
    # # # for p in palette:
    # # #     print(p)
    # # image_path = '/home/andreina/Documents/Programs/Dataset/Generated/navigation/10_21_2025_12_02_45_153326/images/img_0.png'
    image_path = "/home/andreina/Documents/Programs/Dataset/Generated/navigation/11_04_2025_12_30_24_941806/images/img_0.png"
    # # Draw on a dummy frame
    frame = cv2.imread(image_path)
    frame = draw_on_frame(frame, uvs, W, H, palette)
    filename = 'new_img.png'
    cv2.imwrite(filename, frame)
