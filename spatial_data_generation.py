import os
import ast
import re
import json
import argparse
import math
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

from spatial_transformation import transform3d_to_2d, world_to_local, transform_3d_to_2d_with_fov
# from drawer import draw_graph

def get_x_direction(x_pos, thr_x):
    x_dir = "" # center / ignore
    if x_pos > thr_x:
        x_dir = "right"
    elif x_pos < thr_x * -1:
        x_dir = "left"
    return x_dir

def get_y_direction(y_pos, thr_y):
    y_dir = "" # center / ignore
    if y_pos > thr_y:
        y_dir = "above"
    elif y_pos < thr_y * -1:
        y_dir = "below"
    return y_dir

def get_z_direction(z_pos, thr_z):
    z_dir = "" # same depth / ignore
    if z_pos > thr_z:
        z_dir = "front"
    elif z_pos < thr_z * -1:
        z_dir = "behind"
    return z_dir

def get_x_direction_angle(angle_xz, hyperparams):
    ambiguity = hyperparams['angle_threshold_xz']
    abs_angle = abs(angle_xz)
    if abs_angle < ambiguity or abs_angle > (180 - ambiguity):
        # nearly straight ahead or straight behind → no left/right
        x_relation = ""
    elif angle_xz > 0:
        x_relation = "right"
    else:
        x_relation = "left"
    return x_relation

def get_z_direction_angle(angle_xz, hyperparams):
    ambiguity = hyperparams['angle_threshold_xz']
    abs_angle = abs(angle_xz)
    if abs_angle < (90 - ambiguity):
        # nearly straight ahead or straight behind → no above/below
        z_relation = "front"
    elif abs_angle > (90 + ambiguity):
        z_relation = "behind"
    else:
        z_relation = ""
    return z_relation

def get_direction(local_position, hyperparams): #alpha, betha, z, hyperparams):
    x_dir = get_x_direction(local_position[0], hyperparams['ex']) #right, left
    y_dir = get_y_direction(local_position[1], hyperparams['ey']) # above, below
    z_dir = get_z_direction(local_position[2], hyperparams['ez']) # front, behind
    return (x_dir, y_dir, z_dir)

def transform_text2list(text):
    t = ast.literal_eval(text)
    l = list(t)
    return l

def camel_to_words(text):
    new_word = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    words = new_word.split(" ")
    new_word = words[1] if len(words) > 1 else words[0]
    return new_word.lower()

def get_distance_text(number, hyperparams):
    # this method can be improved by using the actual distribution of distances in the dataset 
    # to define the thresholds for near, medium, and far. For now, we are using fixed thresholds for simplicity.
    min_dist = hyperparams['min_distance']
    max_dist = hyperparams['max_distance']
    if number <= min_dist:
        text = "close"
    elif number <= max_dist:
        text = "medium"
    else:
        text = "far"
    return text

def get_records_navigation(csv_path):
    df = pd.read_csv(csv_path)
    dict_navigation = {}
    for _, row in df.iterrows():
        timestep = row.get('timestep')
        obj = row.get("obj-id")
        if timestep not in dict_navigation:
            dict_navigation[timestep] = {
                "action": row.get("ag-action"),
                "degrees": row.get("degrees"),
                # "ag_pos": (row.get("ag-pos-x"), row.get("ag-pos-y"), row.get("ag-pos-z")),
                "ag_pos": (row.get("camera-pos-x"), row.get("camera-pos-y"), row.get("camera-pos-z")),
                "ag_rot": (row.get("camera-horizon"), row.get("ag-rot-y"), row.get("ag-rot-z")),
                "path": row.get("path"),
                "objects": [],
                'bboxes': [],
            }
        if (type(obj) == str and obj is not None) or (type(obj) == float and not math.isnan(obj)):
            dict_navigation[timestep]['objects'].append(obj)
            dict_navigation[timestep]['bboxes'].append((row.get("cmin"), row.get("rmin"), row.get("cmax"), row.get("rmax")))
        if obj == 'nan':
            print(f"Warning: NaN value found for obj-id at timestep {timestep}")
    return dict_navigation

def get_records_objects(csv_path):
    df = pd.read_csv(csv_path)
    dict_objects = {}
    for _, row in df.iterrows():
        dict_objects[row.get('obj-id')] = {
            'obj-type': row.get('obj-type'),
            'obj-pos': (row.get('obj-pos-x'), row.get('obj-pos-y'), row.get('obj-pos-z')),
            'obj-rot': (row.get('obj-rot-x'), row.get('obj-rot-y'), row.get('obj-rot-z')),
            'receptacleObjectIds': transform_text2list(row.get('receptacleObjectIds')),
            'bbox': (row.get("cmin"), row.get("rmin"), row.get("cmax"), row.get("rmax"))
        }
    return dict_objects

def get_knn(obj_id, data, hyperparams):
    obj_pos = np.array(data[obj_id]['local_position'])
    neighbors = []
    k = hyperparams['k_neighbors']
    radius = hyperparams['radius']
    for other_id, other_data in data.items():
        if other_id != obj_id:
            other_pos = np.array(other_data['local_position'])
            (alpha, betha) = data[other_id]['angles']
            dist = np.linalg.norm(obj_pos - other_pos)
            if dist <= radius and alpha is not None: # after evaluate if alpha and betha can be compared with fov
                neighbors.append((other_id, dist))
    neighbors.sort(key=lambda x: x[1])
    knn_neighbors = [n[0] for n in neighbors[:k]]
    return knn_neighbors

def get_direction_angle(diff, hyperparams):
    x, y, z = diff
    angle_xz = math.atan2(x, z) * 180 / math.pi
    # angle_yz = math.atan2(y, z) * 180 / math.pi
    x_dir = get_x_direction_angle(angle_xz, hyperparams)
    z_dir = get_z_direction_angle(angle_xz, hyperparams)
    y_dir = get_y_direction(y, hyperparams['ey'])
    return (x_dir, y_dir, z_dir)

def edges_btw_neighbors(obj_id, neighbors, data, hyperparams, last_seen=-1, extra_data=None):
    edges = []
    local_position_object = np.array(data[obj_id]['local_position'])
    for neighbor in neighbors:
        local_position_neighbor = np.array(data[neighbor]['local_position'])
        dist = np.linalg.norm(local_position_object - local_position_neighbor)
        dist_text = get_distance_text(dist, hyperparams)
        diff = local_position_neighbor - local_position_object
        direction = get_direction(diff, hyperparams)
        angle_direction = get_direction_angle(diff, hyperparams)
        edge = {
            'source': obj_id,
            'target': neighbor,
            'distance': dist_text,
            'relation': direction,
            'angle_relation': angle_direction,
            'inferred': False if last_seen < 0 else True# this can be calculated based on distance or other factors
        }
        if last_seen >= 0:
            edge['last_seen'] = last_seen
        if extra_data is not None:
            extra_data['all_distances'].append(dist)
        edges.append(edge)
    return edges

def get_visible_objects(data, dict_obj, hyperparams):
    visible_objs = {}
    ag_pos = data['ag_pos']
    ag_rot = data['ag_rot']
    objects = data['objects']
    for obj in objects:
        dict_object = {}
        data_object = dict_obj[obj]
        obj_pos = data_object['obj-pos']
        w_to_l, p_l, alpha, betha = transform_3d_to_2d_with_fov(ag_pos, ag_rot, obj_pos, hyperparams)
        dict_object = {
            'category': data_object['obj-type'],
            'position': obj_pos,
            'local_position': tuple(np.round(w_to_l, 3)),
            'local_point': tuple(np.round(p_l, 3)) if p_l[0] is not None and p_l[1] is not None else (None, None),
            'angles': (np.round(alpha, 3), np.round(betha, 3))
        }
        visible_objs[obj] = dict_object
    return visible_objs

def update_memory(memory, visible_objs, timestep):
    for obj_id, obj_data in visible_objs.items():
        memory[obj_id] = {
            'category': obj_data['category'],
            'position': obj_data['position'],
            'last_seen_step': timestep
        }

def create_edges_for_visible_objects(visible_objs, hyperparams, extra_data=None):
    edges = []
    agent_data = {'agent': {'local_position': (0, 0, 0)}} # the agent is always at the origin in the local coordinate system and has no angles with itself
    visible_objs_with_agent = {**visible_objs, **agent_data}
    edges.extend(edges_btw_neighbors('agent', visible_objs.keys(), visible_objs_with_agent, hyperparams, extra_data=extra_data))
    for obj_id in visible_objs:
        neighbors = get_knn(obj_id, visible_objs, hyperparams)
        edges.extend(edges_btw_neighbors(obj_id, neighbors, visible_objs, hyperparams, extra_data=extra_data))
    return edges

def create_edges_for_inferred_objects(non_visible_objs, memory, data, hyperparams):
    edges = []
    ag_pos = data['ag_pos']
    ag_rot = data['ag_rot']
    for obj_id, obj_memory in non_visible_objs.items():
        aux_dict = {obj_id: {}, 'agent': {}}
        obj_pos = obj_memory['position']
        w_to_l, p_l, alpha, betha = transform_3d_to_2d_with_fov(ag_pos, ag_rot, obj_pos, hyperparams)
        aux_dict[obj_id]['local_position'] = w_to_l
        aux_dict['agent']['local_position'] = (0, 0, 0) # the agent is always at the origin in the local coordinate system
        edges.extend(edges_btw_neighbors('agent', [obj_id], aux_dict, hyperparams, \
            last_seen=memory[obj_id]['last_seen_step']))
    return edges
 
def collect_episode_data(dict_navigation, df_obj, hyperparams, extra_data=None):
    episode_dict = {
        'scene': extra_data['scene'],
        # 'episode_id': extra_data['episode_id'],
        'steps': []
    }
    seen_objects_memory = {}
    for timestep, data in dict_navigation.items():
        step_dict = {
            'step': timestep,
            'action': data['action'],
            'degrees': data['degrees'],
            'agent': {
                'position': data['ag_pos'],
                'rotation': data['ag_rot']
            }
        }
        visible_objs = get_visible_objects(data, df_obj, hyperparams)
        update_memory(seen_objects_memory, visible_objs, timestep)
        visible_ids = set(visible_objs)
        non_visible_objs = {
            obj_id: memory_data \
            for obj_id, memory_data in seen_objects_memory.items() \
            if obj_id not in visible_ids
        }
        edges_visible = create_edges_for_visible_objects(visible_objs, hyperparams, extra_data=extra_data)
        edges_inferred = create_edges_for_inferred_objects(non_visible_objs, seen_objects_memory, data, hyperparams)
        step_dict['visible_objects'] = visible_objs
        step_dict['non_visible_objects'] = non_visible_objs
        step_dict['edges_visible'] = edges_visible
        step_dict['edges_inferred'] = edges_inferred

        episode_dict['steps'].append(step_dict)
    return episode_dict

def export_to_json(path_dict, data, json_filename="structured_data.json"):
    os.makedirs(path_dict, exist_ok=True)
    filename = os.path.join(path_dict, json_filename)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Online evaluation")
    parser.add_argument("--csv_path_navigation", type=str)
    parser.add_argument("--csv_path_objects", type=str)
    parser.add_argument("--json_path_dict", type=str)
    parser.add_argument("--other_folder_path", type=str, default="")
    parser.add_argument("--json_filename", type=str, default="structured_data.json")
    args = parser.parse_args()
    return args

def draw_distances(distances):
    if distances:
        plt.figure(figsize=(10, 6))
        sns.histplot(distances, bins=30, kde=True)
        plt.title("Distribution of Distances Between Objects and Agent")
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("No distances to plot.")

def main(args):
    path_navigation = args.csv_path_navigation
    path_objects = args.csv_path_objects
    path_json_dict = args.json_path_dict
    other_folder = args.other_folder_path
    json_filename = args.json_filename
    W, H = 396, 224
    FOV_V = 59
    dict_nav = get_records_navigation(path_navigation)
    df_obj = get_records_objects(path_objects)
    hyperparams = {
        'w': W,
        'h': H,
        'fov_v': FOV_V,
        'epsilon': 1/3,
        'k_neighbors': 3,
        'radius': 1.5, # this can be defined based on the distribution of distances in the dataset, for now we are using a fixed value for simplicity
        'fraction_threshold': 0.15, #0.1 or 0.2
        'ex': 0.1, # threshold for horizontal direction based on image width
        'ey': 0.1, # threshold for vertical direction based on image height
        'ez': 0.15,
        'angle_threshold_xz': 15, # threshold for ambiguity in angle-based direction (in degrees)
        'min_distance': 0.5, # threshold for close distance,
        'max_distance': 1.5 # threshold for far distance
    }
    extra_data = {
        'palette': sns.color_palette("Set2", n_colors=10),
        'other_folder_path': other_folder,
        'scene': path_navigation.split("/")[-1].split("-")[1][:-4], # this can be improved by using a more robust method to extract the episode id from the path
        'all_distances': [] # this can be used to store all the distances between objects and the agent in the episode, which can be useful to define the thresholds for near, medium, and far distances based on the actual distribution of distances in the dataset
        # 'scene': path_navigation.split("-")[-3] # this can be improved by using a more robust method to extract the scene name from the path
    }        
    episode_dict = collect_episode_data(dict_nav, df_obj, hyperparams, extra_data)
    # print(f"Episode data collected: {episode_dict}")
    # draw_distances(extra_data['all_distances'])
    export_to_json(path_json_dict, episode_dict, json_filename)

if __name__ == '__main__':
    args = parse_args()
    main(args)