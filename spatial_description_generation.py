import ast
import re
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import seaborn as sns

from spatial_transformation import transform3d_to_2d

def get_thresholds(fov, fraction_threshold):
    thr = fraction_threshold * (fov / 2)
    return thr

def get_x_direction(alpha, hyperparams):
    fov_h = hyperparams['fov_h']
    thr_h = get_thresholds(fov_h, hyperparams['fraction_threshold'])
    x_dir = "center-horizontal"
    if alpha >thr_h:
        x_dir = "right"
    elif alpha < thr_h * -1:
        x_dir = "left"
    return x_dir

def get_y_direction(betha, hyperparams):
    fov_v = hyperparams["fov_v"]
    thr_v = get_thresholds(fov_v, hyperparams['fraction_threshold'])
    y_dir = "center-vertical"
    if betha > thr_v:
        y_dir = "above"
    elif betha < thr_v * -1:
        y_dir = "below"
    return y_dir

def get_z_direction(z, hyperparams):
    z_dir = "undefined"
    epsilon = hyperparams["epsilon_z"]
    if z > epsilon:
        z_dir = "front"
    elif z < epsilon * -1:
        z_dir = "behind"
    return z_dir

def get_direction(alpha, betha, z, hyperparams):
    x_dir = get_x_direction(alpha, hyperparams) #rirght, left
    y_dir = get_y_direction(betha, hyperparams) # above, below
    z_dir = get_z_direction(z, hyperparams) # front, behind
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

def get_distance_text(number):
    # this method can be improved by using the actual distribution of distances in the dataset 
    # to define the thresholds for near, medium, and far. For now, we are using fixed thresholds for simplicity.
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

def get_records_navigation(csv_path):
    df = pd.read_csv(csv_path)
    dict_navigation = {}
    for _, row in df.iterrows():
        timestep = row.get('timestep')
        action = row.get("ag-action")
        degrees = row.get("degrees")
        ag_pos_x = row.get("ag-pos-x")
        ag_pos_y = row.get("ag-pos-y")
        ag_pos_z = row.get("ag-pos-z")
        ag_rot_x = row.get("ag-rot-x")
        ag_rot_y = row.get("ag-rot-y")
        ag_rot_z = row.get("ag-rot-z")
        obj = row.get("obj-id")
        path = row.get("path")
        if timestep not in dict_navigation:
            dict_navigation[timestep] = {
                "action": action,
                "degrees": degrees,
                "ag_pos": (ag_pos_x, ag_pos_y, ag_pos_z),
                "ag_rot": (ag_rot_x, ag_rot_y, ag_rot_z),
                "path": path,
                "objects": [],
            }
        dict_navigation[timestep]['objects'].append(obj)
    return dict_navigation

def create_nodes(dict_navigation, df_obj):
    nodes = {}
    for timestep, data in dict_navigation.items():
        nodes[timestep] = {}
        ag_pos = data['ag_pos']
        ag_rot = data['ag_rot']
        nodes[timestep]['agent'] = {
            'category': 'agent',
            'position': ag_pos,
            'rotation': ag_rot
        }
        objects = data['objects']
        for obj in objects:
            data_object = df_obj[df_obj['obj-id'] == obj].iloc[0]
            obj_pos = tuple(data_object[['obj-pos-x', 'obj-pos-y', 'obj-pos-z']])
            # obj_rot = tuple(data_object[['obj-rot-x', 'obj-rot-y', 'obj-rot-z']])
            # recep_objs = transform_text2list(data_object['receptacleObjectIds'])
            nodes[timestep][obj] = {
                'category': data_object['obj-type'],
                'position': obj_pos,
                'rotation': tuple(data_object[['obj-rot-x', 'obj-rot-y', 'obj-rot-z']]),
                'visible': 1, #int(obj in data['objects']),
                'bbox': tuple(data_object[['objOrBBox-x', 'objOrBBox-y', 'objOrBBox-z']])
            }
    return nodes

def get_knn(obj_id, data, k=3):
    obj_pos = np.array(data[obj_id]['position'])
    neighbors = []
    for other_id, other_data in data.items():
        if other_id != obj_id:
            other_pos = np.array(other_data['position'])
            dist = np.linalg.norm(obj_pos - other_pos)
            neighbors.append((other_id, dist))
    neighbors.sort(key=lambda x: x[1])
    knn_neighbors = [n[0] for n in neighbors[:k]]
    return knn_neighbors

def get_spatial_direction(obj1, obj2, dist, dict_obj, hyperparams):
    try:
        w_to_l, p_l, alpha, betha = transform3d_to_2d(dict_obj[obj1], dict_obj[obj2], hyperparams)
        (x_l, y_l, z_l) = w_to_l 
        direction = get_direction(alpha, betha, z_l, hyperparams)
        text_dist = get_distance_text(dist)
    except ValueError as v_e:
        # print(f"Error calculating spatial direction between {obj1} and {obj2}: {v_e}")
        direction = None
        raise v_e
    except Exception as e:
        print(f"Unexpected error: {e}")
        direction = None
        raise e
    return direction

def edges_btw_neighbors(obj_id, neighbors, data, hyperparams):
    edges = []
    for neighbor in neighbors:
        dist = np.linalg.norm(np.array(data[obj_id]['position']) - np.array(data[neighbor]['position']))
        try:
            edge = {
                'source': obj_id,
                'target': neighbor,
                'distance': dist,
                'relation': get_spatial_direction(obj_id, neighbor, dist, data, hyperparams),
                'confidence': 1.0 # this can be calculated based on distance or other factors
            }
            edges.append(edge)
        except ValueError as v_e:
            print(f"Error creating edge between {obj_id} and {neighbor}: {v_e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
    return edges

def draw_local_positions(path, data_nodes, hyperparams, extra_data):
    points = []
    obj_ids = []
    for obj_id, obj_data in data_nodes.items():
        if obj_id != 'agent':
            try:
                w_to_l, p_l, alpha, betha = transform3d_to_2d(data_nodes['agent'], obj_data, hyperparams)
                # print(f"Object {obj_id} local position: {w_to_l}, alpha: {alpha}, betha: {betha}")
                points.append(p_l)
                obj_ids.append(obj_id)
            except ValueError as v_e:
                print(f"Error drawing local position for {obj_id}: {v_e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
    print(f"Drawing points for path {path}, points: {points}, obj_ids: {obj_ids}")
    draw_points(path, points, extra_data)

def draw_points(path, points, extra_data):
    # path = extra_data['path']
    other_path = extra_data['other_path']
    frame = cv2.imread(path)
    # u, v = int(round(uv[0])), int(round(uv[1]))
    out = frame.copy()
    cv2.circle(out, (0, 0), 6, (255, 0, 0), -1)
    # cv2.circle(out, (W//2, H//2), 6, (0, 0, 255), -1)
    for i, (u, v) in enumerate(points):
        u, v = int(round(u)), int(round(v))
        (r, g, b) = extra_data['palette'][i%10]
        (r, g, b) = (int(255*r), int(255*g), int(255*b))
        cv2.circle(out, (u, v), 6, (r, g, b), -1)
        cv2.putText(out, str(i), (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(other_path, out)

def create_edges(dict_navigation, nodes, hyperparams, extra_data):
    edges = {}
    for timestep, data in nodes.items():
        print(f"Creating edges for timestep {timestep} with objects: {list(data.keys())}")
        # create edges btw agent and objects
        draw_local_positions(dict_navigation[timestep]['path'], data, hyperparams, extra_data)
        edges[timestep] = []
        for obj_id in data.keys():
            if obj_id == 'agent':
                neighbors = data.keys() - {'agent'}
            else:
                neighbors = get_knn(obj_id, data, k=hyperparams['k_neighbors'])
            edges[timestep] = edges_btw_neighbors(obj_id, neighbors, data, hyperparams)
        break
    return edges
        
def create_graph(dict_navigation, df_obj, hyperparams, extra_data):
    graph = {'nodes': {}, 'edges': {}}
    graph['nodes'] = create_nodes(dict_navigation, df_obj)
    # crear las aristas dependiendo de la relacion y distancia entre objetos y agente (e.g. distancia, visibilidad, etc.)
    graph['edges'] = create_edges(dict_navigation, graph['nodes'], hyperparams, extra_data)
    return graph

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
    parser.add_argument("--image_path", type=str, default="image.png")
    # parser.add_argument("--episode_key", type=str)
    args = parser.parse_args()
    return args

def main(args):
    path_navigation = args.csv_path_navigation
    path_objects = args.csv_path_objects
    path_json_nav = args.json_path_navigation
    path_json_spat_rels = args.json_path_spatial_rels
    path_json_traj = args.json_path_trajectories
    path_image = args.image_path
    # episode = args.episode_key
    W, H = 396, 224
    FOV_V = 59
    dict_nav = get_records_navigation(path_navigation)
    # print(f"Navigation data: {dict_nav}")
    df_obj = pd.read_csv(path_objects)
    # sp_data, spat_ann, traj = get_spatial_relations(dict_nav, episode, df_obj, W, H, FOV_V)
    hyperparams = {
        'w': W,
        'h': H,
        'fov_v': FOV_V,
        'epsilon': 1/3,
        'k_neighbors': 3,
        'epsilon_z': 1e-6,
        'fraction_threshold': 0.15 #0.1 or 0.2
    }
    extra_data = {
        'palette': sns.color_palette("Set2", n_colors=10),
        'other_path': path_image   
    }        
    graph = create_graph(dict_nav, df_obj, hyperparams, extra_data)
    # print(f"Graph: {graph}, nodes: {len(graph['nodes'])}, edges: {len(graph['edges'])}")
    # export_to_json(os.path.join(path_json_nav), graph)
    # export_to_json(os.path.join(path_json_nav), sp_data)
    # export_to_json(os.path.join(path_json_spat_rels), spat_ann)
    # export_to_json(os.path.join(path_json_traj), traj)

if __name__ == '__main__':
    args = parse_args()
    main(args)