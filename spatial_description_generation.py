import os
import ast
import re
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import seaborn as sns

from spatial_transformation import transform3d_to_2d, world_to_local, transform_3d_to_2d_with_fov
from drawer import draw_graph

def get_thresholds(fov, fraction_threshold):
    thr = fraction_threshold * (fov / 2)
    return thr

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

def get_distance_text(number, hyperparams):
    # this method can be improved by using the actual distribution of distances in the dataset 
    # to define the thresholds for near, medium, and far. For now, we are using fixed thresholds for simplicity.
    min_dist = hyperparams['min_distance']
    max_dist = hyperparams['max_distance']
    if number <= min_dist:
        text = "close"
    elif number >= max_dist:
        text = "far"
    else:
        text = ""
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
        dict_navigation[timestep]['objects'].append(obj)
        dict_navigation[timestep]['bboxes'].append((row.get("cmin"), row.get("rmin"), row.get("cmax"), row.get("rmax")))
    return dict_navigation

def get_records_objects(csv_path):
    df = pd.read_csv(csv_path)
    dict_objects = {}
    for _, row in df.iterrows():
        obj_id = row.get('obj-id')
        dict_objects[obj_id] = {
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

# def get_x_direction_bbox(x1, x2, hyperparams):
#     ex = hyperparams['ex']
#     dif = x2 - x1
#     if dif < -ex:
#         # dir_x = "left"
#         dir_x = "l"
#     elif np.abs(dif) <= ex:
#         # dir_x = "center-horizontal"
#         dir_x = "c-h"
#     else:
#         # dir_x = "right"
#         dir_x = "r"
#     return dir_x

# def get_y_direction_bbox(y1, y2, hyperparams):
#     ey = hyperparams['ey']
#     dif = y2 - y1
#     if dif < -ey:
#         # dir_y = "above"
#         dir_y = "a"
#     elif np.abs(dif) <= ey:
#         # dir_y = "center-vertical"
#         dir_y = "c-v"
#     else:
#         # dir_y = "below"
#         dir_y = "b"
#     return dir_y

# def get_direction_bbox(pos1, pos2, hyperparams):
#     x_dir = get_x_direction_bbox(pos1[0], pos2[0], hyperparams) #right, left
#     y_dir = get_y_direction_bbox(pos1[1], pos2[1], hyperparams) # above, below
#     # z_dir = get_z_direction(dist, hyperparams) # front, behind
#     return x_dir, y_dir

# def get_spatial_direction_bbox(obj1, obj2, dist, dict_obj, hyperparams):
#     if obj1 == 'agent':
#         pos1 = hyperparams['w'] / 2, hyperparams['h'] / 2, 0
#         # pos2 = get_box_center(dict_obj[obj2]['bbox'])
#         pos2 = dict_obj[obj2]['bbox']
#     elif obj2 == 'agent':
#         # pos1 = get_box_center(dict_obj[obj1]['bbox'])
#         pos1 = dict_obj[obj1]['bbox']
#         pos2 = hyperparams['w'] / 2, hyperparams['h'] / 2, 0
#     else:
#         # pos1 = get_box_center(dict_obj[obj1]['bbox'])
#         # pos2 = get_box_center(dict_obj[obj2]['bbox'])
#         pos1 = dict_obj[obj1]['bbox']
#         pos2 = dict_obj[obj2]['bbox']
#     direction_xy = get_direction_bbox(pos1, pos2, hyperparams)
#     xl, yl, zl = world_to_local(dict_obj[obj1]['position'], dict_obj[obj1]['rotation'], dict_obj[obj2]['position'])
#     dir_z = get_z_direction(zl, hyperparams)
#     return direction_xy[0], direction_xy[1], dir_z

def get_box_center(bbox):
    x1, y1, x2, y2 = bbox
    c_x = (x1 + x2) / 2
    c_y = (y1 + y2) / 2
    return c_x, c_y

def get_box_center_node(obj_id, dict_obj, hyperparams):
    if obj_id == 'agent':
        c_p = hyperparams['w'] / 2, hyperparams['h'] / 2
    else:
        c_p = get_box_center(dict_obj[obj_id]['bbox'])
    return c_p

def get_difference(p1, p2, p_l):
    dif = np.linalg.norm(np.array(p_l) - np.array(p1))
    dif2 = np.linalg.norm(np.array(p_l) - np.array(p2))
    return dif, dif2

def get_spatial_direction(obj1, obj2, dict_obj, hyperparams):
    # try:
        # if obj1 == 'agent':
        #     w_to_l, p_l, alpha, betha = transform3d_to_2d(dict_obj[obj1], dict_obj[obj2], hyperparams)
        # elif obj2 == 'agent':
        #     w_to_l, p_l, alpha, betha = transform3d_to_2d(dict_obj[obj2], dict_obj[obj1], hyperparams)
        # else:
        #     w_to_l_1, p_l_1, alpha_1, betha_1 = transform3d_to_2d(dict_obj['agent'], dict_obj[obj1], hyperparams)
        #     w_to_l_2, p_l_2, alpha_2, betha_2 = transform3d_to_2d(dict_obj['agent'], dict_obj[obj2], hyperparams)
        #     w_to_l =  np.array(w_to_l_2) - np.array(w_to_l_1)
    w_to_l1 = dict_obj[obj1]['local_position']
    w_to_l2 = dict_obj[obj2]['local_position']
    w_to_l = np.array(w_to_l2) - np.array(w_to_l1)
    # direction = None
    direction = get_direction(w_to_l, hyperparams)
        # p1 = get_box_center_node(obj1, dict_obj, hyperparams)
        # p2 = get_box_center_node(obj2, dict_obj, hyperparams)
        # dif = get_difference(p_l, p1, p2)
        # print(f"Difference btw transformation 3d to 2d: {p_l}, and box center: obj1 {obj1} with {p1}, obj2 {obj2} with {p2}.")
        # text_dist = get_distance_text(dist)
    # except ValueError as v_e:
    #     # print(f"Error calculating spatial direction between {obj1} and {obj2}: {v_e}")
    #     direction = None
    #     raise v_e
    # except Exception as e:
    #     print(f"Unexpected error: {e}")
    #     direction = None
    #     raise e
    return direction

def edges_btw_neighbors(obj_id, neighbors, data, hyperparams):
    edges = []
    obj_pos = np.array(data[obj_id]['local_position'])
    for neighbor in neighbors:
        dist = np.linalg.norm(obj_pos - np.array(data[neighbor]['local_position']))
        dist_text = get_distance_text(dist, hyperparams)
        direction = get_spatial_direction(obj_id, neighbor, data, hyperparams)
        # try:
        edge = {
            'source': obj_id,
            'target': neighbor,
            'distance': dist_text,
            'relation': direction,
            'confidence': 1.0 # this can be calculated based on distance or other factors
        }
        edges.append(edge)
        # except ValueError as v_e:
        #     print(f"Error creating edge between {obj_id} and {neighbor}: {v_e}")
        # except Exception as e:
        #     print(f"Unexpected error: {e}")
        # break
    return edges

def draw_points(timestep, path_image, dict_points, extra_data, dict_objects):
    # path = extra_data['path_folder']
    other_path = extra_data['other_folder_path']
    image_name = "image_" + str(timestep) + ".png"
    frame = cv2.imread(path_image)
    # u, v = int(round(uv[0])), int(round(uv[1]))
    out = frame.copy()
    cv2.circle(out, (0, 0), 6, (255, 0, 0), -1)
    # cv2.circle(out, (W//2, H//2), 6, (0, 0, 255), -1)
    for object_id, point in dict_points.items():
        i = dict_objects[object_id]
        u, v = point
        u, v = int(round(u)), int(round(v))
        (r, g, b) = extra_data['palette'][i%10]
        (r, g, b) = (int(255*r), int(255*g), int(255*b))
        cv2.circle(out, (u, v), 6, (r, g, b), -1)
        cv2.putText(out, str(i), (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(other_path, image_name), out)

def draw_local_positions(timestep, path_image, data_nodes, hyperparams, extra_data, dict_objects):
    dict_points = {}
    for obj_id, obj_data in data_nodes.items():
        if obj_id != 'agent':
            try:
                w_to_l, p_l, alpha, betha = transform3d_to_2d(data_nodes['agent'], obj_data, hyperparams)
                # p_l = get_box_center(obj_data['bbox'])
                # print(f"Object {obj_id} local position: {w_to_l}, alpha: {alpha}, betha: {betha}")
                # xb, yb, zb = obj_data['bbox']
                # dict_points[obj_id] = (xb, yb)
                dict_points[obj_id] = p_l
            except ValueError as v_e:
                print(f"Error drawing local position for {obj_id}: {v_e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
    # print(f"Drawing for timestep: {timestep}, points: {points}, obj_ids: {obj_ids}")
    draw_points(timestep, path_image, dict_points, extra_data, dict_objects)

def draw_local_positions_scenes(dict_navigation, nodes, hyperparams, extra_data, dict_objects):
    for timestep, data in nodes.items():
        path_image = dict_navigation[timestep]['path']
        draw_local_positions(timestep, path_image, data, hyperparams, extra_data, dict_objects)

def create_edges(nodes, hyperparams):
    edges = {}
    for timestep, data in nodes.items():
        # print(f"Creating edges for timestep {timestep} with objects: {list(data.keys())}")
        # create edges btw agent and objects
        # draw_local_positions(timestep, dict_navigation[timestep]['path'], data, hyperparams, extra_data)
        edges[timestep] = []
        for obj_id in data.keys():
            if obj_id == 'agent':
                neighbors = data.keys() - {'agent'}
            else:
                neighbors = get_knn(obj_id, data, hyperparams)
            edges[timestep].extend(edges_btw_neighbors(obj_id, neighbors, data, hyperparams))
            # break
        # break
    return edges

# def create_nodes(dict_navigation, df_obj, hyperparams):
#     nodes = {}
#     unique_objects = set()
#     unique_objects.add('agent')
#     for timestep, data in dict_navigation.items():
#         nodes[timestep] = {}
#         ag_pos = data['ag_pos']
#         ag_rot = data['ag_rot']
#         nodes[timestep]['agent'] = {
#             'category': 'agent',
#             'position': ag_pos,
#             'rotation': ag_rot, 
#             'visible': 0, # the agent itself is not visible, but we can save its position and rotation for future use if needed
#             'bbox': (0, 0, 0, 0), # the agent does not have a bbox, but we can save a placeholder for future use if needed
#             'local_position': (0, 0, 0), # the agent is always in the center of the local coordinate system, so its local position is (0, 0, 0)
#             # 'bbox3d': (0, 0, 0) # the agent does not have a bbox, but we can save a placeholder for future use if needed
#             'local_point': (hyperparams['w'] / 2, hyperparams['h'] / 2), # the agent is always in the center of the image, so its local point is (w/2, h/2)
#             'angles': (0, 0) # the agent is always facing forward, so its angles are (0, 0)
#         }
#         objects = data['objects']
#         bboxes = data['bboxes']
#         for obj, bbox in zip(objects, bboxes):
#             unique_objects.add(obj)
#             data_object = df_obj[df_obj['obj-id'] == obj].iloc[0]
#             obj_pos = tuple(data_object[['obj-pos-x', 'obj-pos-y', 'obj-pos-z']])
#             # obj_rot = tuple(data_object[['obj-rot-x', 'obj-rot-y', 'obj-rot-z']])
#             # recep_objs = transform_text2list(data_object['receptacleObjectIds'])
#             w_to_l, p_l, alpha, betha = transform_3d_to_2d_with_fov(ag_pos, ag_rot, obj_pos, hyperparams)
#             nodes[timestep][obj] = {
#                 'category': data_object['obj-type'],
#                 'position': obj_pos,
#                 'rotation': tuple(data_object[['obj-rot-x', 'obj-rot-y', 'obj-rot-z']]),  # for objects is not necessary the riotation, but we can save it for future use if needed
#                 'visible': 1, #int(obj in data['objects']),
#                 # 'bbox3d': tuple(data_object[['objOrBBox-x', 'objOrBBox-y', 'objOrBBox-z']]),
#                 'bbox': bbox,
#                 'local_position': w_to_l,
#                 'local_point': p_l,
#                 'angles': (alpha, betha)
#             }
#     return nodes, unique_objects

def get_agent_data(data):
    ag_data = {
        'position': data['ag_pos'],
        'rotation': data['ag_rot']
    }
    return ag_data

def get_visible_objects(data, df_obj, hyperparams):
    visible_objs = []
    ag_pos = data['ag_pos']
    ag_rot = data['ag_rot']
    objects = data['objects']
    for obj in objects:
        dict_object = {}
        data_object = df_obj[df_obj['obj-id'] == obj].iloc[0]
        obj_pos = data_object[['obj-pos']]
        w_to_l, p_l, alpha, betha = transform_3d_to_2d_with_fov(ag_pos, ag_rot, obj_pos, hyperparams)
        dict_object = {
            'category': data_object['obj-type'],
            'position': obj_pos,
            'local_position': w_to_l,
            'local_point': p_l,
            'angles': (alpha, betha)
        }
        visible_objs.append(dict_object)
    return visible_objs

def update_memory(memory, visible_objs, timestep):
    for obj_id, obj_data in visible_objs.items():
        memory[obj_id] = {
            'category': obj_data['category'],
            'position': obj_data['position'],
            'last_seen_step': timestep
        }

def create_edges_for_visible_objects(visible_objs, hyperparams):
    edges = []
    for obj_id in visible_objs:
        neighbors = get_knn(obj_id, visible_objs, hyperparams)
        edges.extend(edges_btw_neighbors(obj_id, neighbors, hyperparams))
    return edges
 
def collect_episode_data(dict_navigation, df_obj, hyperparams, extra_data):
    episode_dict = {
        'scene': extra_data['scene'],
        'episode_id': extra_data['episode_id'],
        'steps': []
    }
    # set_objects, dict_objects = get_objects(df_obj)
    seen_objects_memory = {}
    for timestep, data in dict_navigation.items():
        step_dict = {
            'step': timestep,
            'action': data['ag-action'],
            'degrees': data['degrees'],
            'agent': get_agent_data(data),

        }
        visible_objs = get_visible_objects(data, df_obj, hyperparams)
        update_memory(seen_objects_memory, visible_objs, timestep)
        visible_ids = set(visible_objs)
        non_visible_objs = {
            obj_id: memory_data \
            for obj_id, memory_data in seen_objects_memory.items() \
            if obj_id not in visible_ids
        }
        # non_visible_objs = {} if timestep == 0 else set_objects - set(visible_objs)
        edges_visible = create_edges_for_visible_objects(visible_objs, hyperparams)
        edges_inferred = create_edges_for_inferred_objects(non_visible_objs, dict_navigation, df_obj, hyperparams)
        step_dict['visible_objects'] = visible_objs
        step_dict['non_visible_objects'] = non_visible_objs
        step_dict['edges_visible'] = edges_visible
        step_dict['edges_inferred'] = edges_inferred

        episode_dict['steps'].append(step_dict)
    # print("Unique objects in the dataset:", set_objects)
    # objects = {obj_id: i for i, obj_id in enumerate(list(set_objects))}
    # print(f"Unique objects in the dataset: {dict_objects}")
    # draw_local_positions_scenes(dict_navigation, nodes, hyperparams, extra_data, dict_objects)
    # crear las aristas dependiendo de la relacion y distancia entre objetos y agente (e.g. distancia, visibilidad, etc.)
    # episode_dict['edges'] = create_edges(episode_dict['nodes'], hyperparams)
    return episode_dict

def export_to_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Online evaluation")
    parser.add_argument("--csv_path_navigation", type=str)
    parser.add_argument("--csv_path_objects", type=str)
    # parser.add_argument("--json_path_navigation", type=str)
    # parser.add_argument("--json_path_spatial_rels", type=str)
    # parser.add_argument("--json_path_trajectories", type=str)
    parser.add_argument("--json_path_dict", type=str)
    # parser.add_argument("--folder_path", type=str, default="data/")
    parser.add_argument("--other_folder_path", type=str, default="")
    # parser.add_argument("--episode_key", type=str)
    args = parser.parse_args()
    return args

def main(args):
    path_navigation = args.csv_path_navigation
    path_objects = args.csv_path_objects
    # path_json_nav = args.json_path_navigation
    # path_json_spat_rels = args.json_path_spatial_rels
    # path_json_traj = args.json_path_trajectories
    path_json_dict = args.json_path_dict
    # path_folder = args.folder_path
    other_folder = args.other_folder_path
    # episode = args.episode_key
    W, H = 396, 224
    FOV_V = 59
    dict_nav = get_records_navigation(path_navigation)
    # print(f"Navigation data: {dict_nav}")
    df_obj = get_records_objects(path_objects)
    # sp_data, spat_ann, traj = get_spatial_relations(dict_nav, episode, df_obj, W, H, FOV_V)
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
        'min_distance': 1.5, # threshold for close distance,
        'max_distance': 3.0 # threshold for far distance
    }
    extra_data = {
        'palette': sns.color_palette("Set2", n_colors=10),
        # 'folder_path': path_folder,
        'other_folder_path': other_folder,
        'episode_id': path_navigation.split("/")[-2], # this can be improved by using a more robust method to extract the episode id from the path
        'scene': path_navigation.split("/")[-3] # this can be improved by using a more robust method to extract the scene name from the path
    }        
    episode_dict = collect_episode_data(dict_nav, df_obj, hyperparams, extra_data)
    # draw_graph(graph)
    # print(f"Graph created with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges.")
    # print(f"Graph: {graph}")
    export_to_json(path_json_dict, episode_dict)
    # print(f"Graph: {graph}, nodes: {len(graph['nodes'])}, edges: {len(graph['edges'])}")
    # export_to_json(os.path.join(path_json_nav), graph)
    # export_to_json(os.path.join(path_json_nav), sp_data)
    # export_to_json(os.path.join(path_json_spat_rels), spat_ann)
    # export_to_json(os.path.join(path_json_traj), traj)

if __name__ == '__main__':
    args = parse_args()
    main(args)