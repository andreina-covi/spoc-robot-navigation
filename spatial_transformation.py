import numpy as np

def calculate_focal_length(val, fov):
    f = val / (2 * np.tan(np.deg2rad(fov/2)))
    return f

def get_other_fov(val1, val2, fov):
    fov_other = 2 * np.arctan((val1 / val2) * np.tan(np.deg2rad(fov / 2)))
    return np.degrees(fov_other)

def get_focal_length(w, h, fov_v):
    fy = calculate_focal_length(h, fov_v)
    fov_h = get_other_fov(w, h, fov_v)
    fx = calculate_focal_length(w, fov_h)
    return fx, fy, fov_h

def get_center_point(bbox):
    xc = np.mean([bbox[0], bbox[2]])
    yc = np.mean([bbox[1], bbox[3]])
    return xc, yc

def calculate_angle(coord1, coord2):
    angle = np.arctan2(coord1, coord2)
    return np.degrees(angle)

def world_to_local(agent_pos, agent_rot_deg, object_pos):
    # agent_rot_deg = (pitch_x, yaw_y, roll_z) in degrees
    pitch, yaw, roll = np.deg2rad(agent_rot_deg)
    Rx = np.array([[1,0,0],
                   [0,np.cos(pitch),-np.sin(pitch)],
                   [0,np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[ np.cos(yaw),0,np.sin(yaw)],
                   [0,1,0],
                   [-np.sin(yaw),0,np.cos(yaw)]])
    # Rz = np.array([[np.cos(roll),-np.sin(roll),0],
    #                [np.sin(roll), np.cos(roll),0],
    #                [0,0,1]])
    #R_cw = Ry @ Rx #@ Rz         # camera->world
    #R_wc = R_cw.T               # world->camera
    R_wc = Rx @ Ry
    Pw = np.asarray(object_pos).reshape(3,1)
    C  = np.asarray(agent_pos).reshape(3,1)
    Pl = (R_wc @ (Pw - C)).flatten()  # local (Xl,Yl,Zl)
    return Pl  # (x_l, y_l, z_l)

def projection_with_local_vector(local_xyz, c_point, foc_l, hyperparams):
    xl, yl, zl = local_xyz
    if zl <= hyperparams['epsilon_z']:
        # print(f"Z none: {zl}")
        raise ValueError(f"Z is too small: {zl}")
        # zl = np.absolute(zl)
    u = foc_l[0] * (xl / zl) + c_point[0]
    v = foc_l[1] * (yl / zl) + c_point[1]
    return float(u), float(v)

def transform_3d_to_2d(obj1_pos, obj1_rot, obj2_pos, c_point, foc_l, hyperparams):
    x_l, y_l, z_l = world_to_local(obj1_pos, obj1_rot, obj2_pos)
    alpha = calculate_angle(x_l, z_l)
    betha = calculate_angle(y_l, z_l)
    u_l, v_l = projection_with_local_vector((x_l, y_l, z_l), c_point, foc_l, hyperparams)
    return (x_l, y_l, z_l), (u_l, v_l), alpha, betha

def transform3d_to_2d(obj1_data, obj2_data, hyperparams):
    obj1_pos = obj1_data['position']
    obj1_rot = obj1_data['rotation']
    obj2_pos = obj2_data['position']
    w, h = hyperparams['w'], hyperparams['h']
    fov_v = hyperparams['fov_v']
    fx, fy, fov_h = get_focal_length(w, h, fov_v)
    hyperparams['fov_h'] = fov_h
    c_point = (w // 2, h // 2)
    return transform_3d_to_2d(obj1_pos, obj1_rot, obj2_pos, c_point, (fx, fy), hyperparams)

def main():
    W, H = 800, 600
    FOV_V = 90
    hyperparams = {
        'w': W,
        'h': H,
        'fov_v': FOV_V,
        'epsilon': 1/3,
        'k_neighbors': 3,
        'epsilon_z': 1e-6
    }
    obj1_data = {
        'position': [0, 0, 0],
        'rotation': [0, 0, 0]
    }
    obj2_data = {
        'position': [1, 0, 10],
        'rotation': [0, 0, 0]
    }
    local_coords, pixel_coords, alpha, betha = transform3d_to_2d(obj1_data, obj2_data, hyperparams)
    print(f"Local coordinates: {local_coords}, Pixel coordinates: {pixel_coords}, Alpha: {alpha:.2f}, Betha: {betha:.2f}")

if __name__ == '__main__':
    main()