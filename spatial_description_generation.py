import os
import ast
import re
import json
import math
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import cv2


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
    Rz = np.array([[np.cos(roll),-np.sin(roll),0],
                   [np.sin(roll), np.cos(roll),0],
                   [0,0,1]])
    R_cw = Ry @ Rx @ Rz         # camera->world
    R_wc = R_cw.T               # world->camera
    Pw = np.asarray(object_pos).reshape(3,1)
    C  = np.asarray(agent_pos).reshape(3,1)
    Pl = (R_wc @ (Pw - C)).flatten()  # local (Xl,Yl,Zl)
    return Pl  # (x_l, y_l, z_l)

def projection_with_local_vector(local_xyz, c_point, foc_l):
    xl, yl, zl = local_xyz
    if zl <= 0:
        # print(f"Z none: {zl}")
        zl = np.absolute(zl)
    u = foc_l[0] * (xl / zl) + c_point[0]
    v = foc_l[1] * (yl / zl) + c_point[1]
    return float(u), float(v)

def transform_3d_to_2d(ag_pos, ag_rot, obj_pos, c_point, foc_l):
    x_l, y_l, z_l = world_to_local(ag_pos, ag_rot, obj_pos)
    alpha = calculate_angle(x_l, z_l)
    betha = calculate_angle(y_l, z_l)
    u_l, v_l = projection_with_local_vector((x_l, y_l, z_l), c_point, foc_l)
    return (x_l, y_l, z_l), (u_l, v_l), alpha, betha

def get_x_direction(alpha, fov_h, epsilon=1/3):
    t_alpha = epsilon * (fov_h / 2)
    x_dir = "front"
    if alpha >t_alpha:
        x_dir = "right"
    elif alpha < t_alpha * -1:
        x_dir = "left"
    return x_dir

def get_y_direction(betha, fov_v, epsilon=1/3):
    t_betha = epsilon * (fov_v / 2)
    y_dir = ""
    if betha > t_betha:
        y_dir = "above"
    elif betha < (t_betha * -1):
        y_dir = "below"
    return y_dir

def get_z_direction(distance):
    return distance

def get_direction(alpha, betha, distance, fov_h, fov_v, epsilon=1/3):
    x_dir = get_x_direction(alpha, fov_h, epsilon) #rirght, left
    y_dir = get_y_direction(betha, fov_v, epsilon) # above, below
    z_dir = get_z_direction(distance) # front, behind
    return x_dir, y_dir, z_dir

def transform_text2list(text):
    t = ast.literal_eval(text)
    l = list(t)
    return l

def camel_to_words(text):
    new_word = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    words = new_word.split(" ")
    new_word = words[1] if len(words) > 1 else words[0]
    return new_word.lower()

def get_horizontal_phrase(text):
    new_text = "in front of" if text == "front" else f"to the {text} of"
    return new_text

def get_vertical_phrase(text):
    new_text = f" and {text} eye level" if text else text
    return new_text

def get_dist_phrase(text):
    new_text = text
    if text == "medium":
        new_text = "at a medium distance"
    elif text == "far":
        new_text = "far away"
    return new_text

def get_distance_text(number):
    text = ""
    if number < 1.5:
        text = "near"
    elif number > 1.5 and number < 2.5:
        text = "medium"
    else:
        text = "far"
    return text

def get_spatial_descriptions(object_dict):
    name = object_dict["object_name"]
    dir = object_dict["egocentric_label"]
    phr_dir_x = get_horizontal_phrase(dir["horiz"])
    phr_dir_y = get_vertical_phrase(dir["vert"])
    phr_dir_z = get_dist_phrase(dir["dist"])
    if phr_dir_y:
        text = f"From the agent's viewpoint, the {name} is {phr_dir_x} the agent, {phr_dir_z}{phr_dir_y}"
    else:
        text = f"From the agent's viewpoint, the {name} is {phr_dir_x} the agent and {phr_dir_z}"
    return text

# create a json file for training a LLM
def get_spatial_relations(dict_navigation, episode_id, df_obj, w, h, fov_v, epsilon=1/3,):
    spatial_data = {
        "episode": episode_id,
        "agent_frame": "egocentric",
        "sequence": []
    }
    spatial_annotation = {
        "episode": episode_id,
        "sequence": []
    }
    trajectories = {
        "episode": episode_id
    }
    fx, fy, fov_h = get_focal_length(w, h, fov_v)
    c_point = (w // 2, h // 2)
    # deg = 30
    for i, (d_agent, d_objects) in enumerate(dict_navigation.items()):
        action, deg, ag_pos, ag_rot, path = d_agent
        ag_pos = np.array(ag_pos)
        ag_rot = np.array(ag_rot)
        step_dict = {
            "step": i,
            "action": "initialize" if i == 0 else action,
            # "degrees": deg if i > 0 else 0,
            "objects": {}
        }
        step_dict["degrees"] = deg if step_dict["action"] not in ["initialize", "move_ahead"] else 0
        description_dict = {
            "step": i,
            "objects": {}
        }
        for d_object in d_objects:
            data_json_object = {}
            obj_id = d_object["object"]
            if obj_id not in trajectories:
                trajectories[obj_id] = []
            obj_dist = d_object["distance"]
            data_object = df_obj[df_obj['obj-id'] == obj_id].iloc[0]
            obj_pos = tuple(data_object[['obj-pos-x', 'obj-pos-y', 'obj-pos-z']])
            recep_objs = transform_text2list(data_object['receptacleObjectIds'])
            filtered_recep_objs = []
            if recep_objs:
                for recep_obj in recep_objs:
                    if df_obj['obj-id'].isin([recep_obj]).any():
                        filtered_recep_objs.append(recep_obj)
            obj_pos = np.array(obj_pos)
            w_to_l, p_l, alpha, betha = transform_3d_to_2d(ag_pos, ag_rot, obj_pos, c_point, (fx, fy)) 
            dir_x, dir_y, dir_z = get_direction(alpha, betha, obj_dist, fov_h, fov_v, epsilon)
            if obj_id.startswith("window") or obj_id.startswith("wall"):
                dir_y = ""
            data_json_object["object_name"] = camel_to_words(data_object['obj-type'])
            data_json_object["egocentric_label"] = {
                "horiz": dir_x, 
                "vert": dir_y,
                "dist": get_distance_text(dir_z)
            }
            data_json_object["relations"] = {
                "contains": filtered_recep_objs
            }
            step_dict["objects"][obj_id] = data_json_object
            description_dict["objects"][obj_id] = get_spatial_descriptions(data_json_object)
            trajectories[obj_id].append({"step": i, "horiz": dir_x, "vert": dir_y, "dist": get_distance_text(dir_z)})
        spatial_data["sequence"].append(step_dict)
        spatial_annotation["sequence"].append(description_dict)
    return spatial_data, spatial_annotation, trajectories

def get_records_navigation(csv_path):
    df = pd.read_csv(csv_path)
    dict_navigation = {}
    for _, row in df.iterrows():
        action = row.get("ag-action")
        degrees = row.get("degrees")
        ag_pos_x = row.get("ag-pos-x")
        ag_pos_y = row.get("ag-pos-y")
        ag_pos_z = row.get("ag-pos-z")
        ag_rot_x = row.get("ag-rot-x")
        ag_rot_y = row.get("ag-rot-y")
        ag_rot_z = row.get("ag-rot-z")
        obj = row.get("obj-id")
        dist = row["obj-distance"]
        path = row.get("path")
        key = (action, degrees, (ag_pos_x, ag_pos_y, ag_pos_z), (ag_rot_x, ag_rot_y, ag_rot_z), path)
        if key not in dict_navigation:
            dict_navigation[key] = []
        dict_navigation[key].append({
            "object": obj, 
            "distance": dist,
        })
    return dict_navigation

def export_to_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Online evaluation")
    parser.add_argument("--csv_path_navigation", type=str)
    parser.add_argument("--csv_path_objects", type=str)
    parser.add_argument("--json_path_navigation", type=str)
    parser.add_argument("--json_path_spatial_rels", type=str)
    parser.add_argument("--json_path_trajectories", type=str)
    parser.add_argument("--episode_key", type=str)
    args = parser.parse_args()
    return args

def main(args):
    path_navigation = args.csv_path_navigation
    path_objects = args.csv_path_objects
    path_json_nav = args.json_path_navigation
    path_json_spat_rels = args.json_path_spatial_rels
    path_json_traj = args.json_path_trajectories
    episode = args.episode_key
    W, H = 396, 224
    FOV_V = 59
    dict_nav = get_records_navigation(path_navigation)
    df_obj = pd.read_csv(path_objects)
    sp_data, spat_ann, traj = get_spatial_relations(dict_nav, episode, df_obj, W, H, FOV_V)
    export_to_json(os.path.join(path_json_nav), sp_data)
    export_to_json(os.path.join(path_json_spat_rels), spat_ann)
    export_to_json(os.path.join(path_json_traj), traj)

if __name__ == '__main__':
    args = parse_args()
    main(args)