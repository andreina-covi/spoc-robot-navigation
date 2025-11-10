import math
import argparse
import json
import numpy as np
import pandas as pd
import cv2
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Online evaluation")
    parser.add_argument("--csv_path", type=str)
    # parser.add_argument("--jsonl_out", default="train.jsonl", type=str)
    # parser.add_argument("--max_eps_len", default=-1, type=int)
    # parser.add_argument("--det_type", default="gt", help="gt or detic", choices=["gt", "detic"])
    # parser.add_argument("--total_num_videos", type=int, default=8200)

    args = parser.parse_args()
    return args

# # === 2. Helper functions ===
# def cardinal_from_yaw(yaw):
#     dirs = ["north","north-east","east","south-east","south","south-west","west","north-west"]
#     idx = int(((yaw % 360) + 22.5) // 45) % 8
#     return dirs[idx]

def dir8_global(A, B):
    # A=(Ax,Ay) agent pos, B=(Bx,By) target pos; directions vs. world axes
    dx, dy = B[0]-A[0], B[1]-A[1]
    # angle w.r.t +y is convenient for “front” pointing up; shift by 22.5° for 8 sectors
    ang = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    idx = int(ang // 22.5)
    labels = [
        "front","front-right","right", "right", 
        "right", "right", "back-right","back",
        "back", "back-left","left", "left", 
        "left", "left", "front-left", "front"]
    return labels[idx]
        

def cardinal_from_bbox(bbox_point, p_point):
    # print(bbox_point, p_point)
    (cmin, rmin, cmax, rmax) = bbox_point
    x = np.mean([cmin, cmax])
    y = np.mean([rmin, rmax])
    direction = dir8_global(p_point, (x, y))
    return direction

def split_name(name):
    return name.split("|")

# def bearing_from_agent(ax, az, ox, oz, ayaw_deg):
#     vx, vz = ox - ax, oz - az
#     obj_ang = math.degrees(math.atan2(vx, vz))
#     rel = (obj_ang - ayaw_deg + 540) % 360 - 180
#     if -22.5 <= rel <= 22.5: return "front"
#     if 22.5 < rel <= 67.5: return "front-right"
#     if 67.5 < rel <= 112.5: return "right"
#     if 112.5 < rel <= 157.5: return "back-right"
#     if rel > 157.5 or rel <= -157.5: return "back"
#     if -157.5 < rel <= -112.5: return "back-left"
#     if -112.5 < rel <= -67.5: return "left"
#     return "front-left"

# def relational_position(ag_pos, obj_pos):
#     rel_pos =  obj_pos - ag_pos
#     rel_x = "right" if rel_pos[0] > 0 else "left"
#     rel_y = "forward" if rel_pos[1] > 0 else "backward"
#     rel_z = "up" if rel_pos[2] > 0 else "down"
#     return (rel_x, rel_y, rel_z), rel_pos

def data_from_world(pos):
    x, y, z = pos
    distance = math.sqrt(x**2 + y**2 + z**2)
    azimuth = math.degrees(math.atan2(x, z))       # Left/right (°)
    elevation = math.degrees(math.atan2(y, math.sqrt(x**2 + z**2)))  # Up/down (°)
    return distance, azimuth, elevation

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

# def intrinsics_from_fov(w, h, fov_h, fov_v):
#     fy = (h/2.0) / np.tan(np.deg2rad(fov_v)/2.0)
#     fx = (w/2.0) / np.tan(np.deg2rad(fov_h)/2.0)
#     fx = fy * (w/h)
#     cx, cy = w/2.0, h/2.0
#     return fx, fy, cx, cy

def projection_with_local_vector(local_xyz, c_point, foc_l):
    xl, yl, zl = local_xyz
    if zl <= 0:
        print(f"Z none: {zl}")
        # return None  # behind camera
        # zl = 0.001
        zl = np.absolute(zl)
    # fx, fy, cx, cy = intrinsics_from_fov(W, H, fov_h, fov_v)
    u = foc_l[0] * (xl / zl) + c_point[0]
    v = foc_l[1] * (yl / zl) + c_point[1]
    # if 0 <= u < W and 0 <= v < H:
    #     return (float(u), float(v))
    return float(u), float(v)

def projection_with_angles(foc_l, alpha, betha, c_point):
    u = foc_l[0] * np.tan(np.deg2rad(alpha)) + c_point[0]
    v = foc_l[1] * np.tan(np.deg2rad(betha)) + c_point[1]
    return float(u), float(v)

def transform_3d_to_2d(ag_pos, ag_rot, obj_pos, bbox, c_point, foc_l):
    x_l, y_l, z_l = world_to_local(ag_pos, ag_rot, obj_pos)
    alpha = calculate_angle(x_l, z_l)
    betha = calculate_angle(y_l, z_l)
    u_b, v_b = get_center_point(bbox)
    u_a, v_a = projection_with_angles(foc_l, alpha, betha, c_point)
    u_l, v_l = projection_with_local_vector((x_l, y_l, z_l), c_point, foc_l)
    return (u_b, v_b), (u_a, v_a), (u_l, v_l), alpha, betha

def calculate_focal_length(val, fov):
    f = val / (2 * np.tan(np.deg2rad(fov/2)))
    return f

def get_other_fov(val1, val2, fov):
    fov_other = 2 * np.arctan((val1 / val2) * np.tan(np.deg2rad(fov / 2)))
    return np.degrees(fov_other)

def get_focal_length(w, h, fov_v):
    fy = calculate_focal_length(h, fov_v)
    # print("fy", fy)
    fov_h = get_other_fov(w, h, fov_v)
    # print("fov_h", fov_h)
    fx = calculate_focal_length(w, fov_h)
    return fx, fy

def get_spatial_relations(dict_navigation, w, h, fov_v):
    fx, fy = get_focal_length(w, h, fov_v)
    c_point = (w // 2, h // 2)
    # palette = sns.color_palette("Set2", n_colors=20)
    for i, (d_agent, d_objects) in enumerate(dict_navigation.items()):
        path, action, ag_pos, ag_rot = d_agent
        ag_pos = np.array(ag_pos)
        ag_rot = np.array(ag_rot)
        print(f"{i}: {action}")
        uvs = []
        for d_object in d_objects:
            print("++++++++")
            obj = d_object["Object"]
            obj_pos = d_object["Position"]
            dist = d_object["Distance"]
            bbox = d_object["BBox"]
            uvs.append(get_center_point(bbox))
            obj_pos = np.array(obj_pos)
            # text_pos, rel_pos = relational_position(ag_pos, obj_pos)
            # print(text_pos, rel_pos)
            p_bbox, p_a, p_l, alpha, betha = transform_3d_to_2d(ag_pos, ag_rot, obj_pos, bbox, c_point, (fx, fy)) 
            # rel_dist, az, el = data_from_world(rel_pos)
            
            dir = cardinal_from_bbox(bbox, c_point)
            print(f"Object: {obj}, Dist: {dist}") #, Rel pos: {rel_pos}, Text pos: {text_pos}")
            print(f"BBox: {p_bbox}, Point angle: {p_a}, Point local vector: {p_l}")
            print(f"Angle from 3d to 2d: alpha: {alpha}, betha: {betha}, dir: {dir}")
            print("++++++++")
        print("----------------------------------")
        # frame = cv2.imread(path)
        # ot_frame = draw_on_frame(frame, uvs, p_point, palette)
        # filename = 'new_img' + str(i) + ".png"
        # cv2.imwrite(filename, ot_frame)

def draw_on_frame(frame_bgr, uvs, p_point, palette):
    if uvs is None:
        return frame_bgr
    # u, v = int(round(uv[0])), int(round(uv[1]))
    out = frame_bgr.copy()
    # cv2.circle(out, (0, 0), 6, (255, 0, 0), -1)
    cv2.circle(out, p_point, 6, (0, 0, 255), -1)
    for i, (u, v) in enumerate(uvs):
        u, v = int(round(u)), int(round(v))
        (r, g, b) = palette[i%10]
        (r, g, b) = (int(255*r), int(255*g), int(255*b))
        cv2.circle(out, (u, v), 6, (r, g, b), -1)
        cv2.putText(out, str(i), (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out

def get_records(csv_path):
    df = pd.read_csv(csv_path)
    dict_navigation = {}
    for _, row in df.iterrows():
        action = row.get("ag-action")
        ag_pos_x = row.get("ag-pos-x")
        ag_pos_y = row.get("ag-pos-y")
        ag_pos_z = row.get("ag-pos-z")
        ag_rot_x = row.get("ag-rot-x")
        ag_rot_y = row.get("ag-rot-y")
        ag_rot_z = row.get("ag-rot-z")
        obj = row.get("obj-id")
        obj_pos_x = row.get("obj-pos-x")
        obj_pos_y = row.get("obj-pos-y")
        obj_pos_z = row.get("obj-pos-z")
        dist = row["obj-distance"]
        cmin = row.get("cmin")
        rmin = row.get("rmin")
        cmax = row.get("cmax")
        rmax = row.get("rmax")
        path = row.get("path")
        key = (path, action, (ag_pos_x, ag_pos_y, ag_pos_z), (ag_rot_x, ag_rot_y, ag_rot_z))
        if key not in dict_navigation:
            dict_navigation[key] = []
        dict_navigation[key].append({
            "Object": obj, 
            "Position": (obj_pos_x, obj_pos_y, obj_pos_z), 
            "Distance": dist,
            "BBox": (cmin, rmin, cmax, rmax)
        })
    return dict_navigation

def write_json(json_path, records):
    with open(json_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main(args):
    csv_path = args.csv_path
    # jsonl_out = args.jsonl_out
    records = get_records(csv_path)
    w, h = 396, 224
    # p_point = (w//2, h//2)
    fov_v = 59
    get_spatial_relations(records, w, h, fov_v)
    # write_json(jsonl_out, records)

# print(f"✅ Saved {len(records)} examples to {jsonl_out}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
