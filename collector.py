import os
import datetime

import numpy as np
import pandas as pd
import cv2

from utils.constants.objaverse_data_dirs import OBJAVERSE_NAVIGATION_PATH
from utils.constants.stretch_initialization_utils import AGENT_ROTATION_DEG
from utils.type_utils import THORActions

class Collector:

    def __init__(self):
        self.dict_agent = {}
        self.data_objects = {'objects': set()}
        self.index = 0
        dt = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f")
        self.image_path = os.path.join(OBJAVERSE_NAVIGATION_PATH, dt, 'images')
        self.objects_path = os.path.join(OBJAVERSE_NAVIGATION_PATH, dt, 'objects.csv')
        self.navigation_path = os.path.join(OBJAVERSE_NAVIGATION_PATH, dt,'navigation.csv')

        os.makedirs(self.image_path, exist_ok=True)

    def round_number(self, arr_numbers, n_round):
        rounded_arr = []
        if type(arr_numbers) is dict:
            rounded_arr = [arr_numbers['x'], arr_numbers['y'], arr_numbers['z']]
        else:
            rounded_arr = arr_numbers
        return tuple([np.round(number, n_round) for number in rounded_arr])
    
    def get_min_by_axis(self, bbox):
        array = np.array(bbox)
        assert array.shape == (8, 3)
        x_min = np.min(array[:, 0])
        y_min = np.min(array[:, 1])
        z_min = np.min(array[:, 2])
        return (x_min, y_min, z_min)
    
    def get_object_data(self, arr_objects):
        objects = set()
        cond_objs = []
        # detections = controller.last_event.instance_detections2D

        for obj_dict in arr_objects:
            if not obj_dict['visible']:
                continue
            # if not detections.__contains__(obj_dict['objectId']):
            #     continue
            # bbox = detections.__getitem__(obj_dict['objectId'])
            cond_objs.append([
                # class label
                # obj_dict['objectType'],
                # id
                obj_dict['objectId'],
                # name
                # obj_dict['name'],
                # global position 
                # tuple(self.round_number(obj_dict['position'], 2)),
                # local rotation
                # tuple(self.round_number(obj_dict['rotation'], 2)),
                # euclidean distance from the center-point to the agent
                np.round(obj_dict['distance'], 4),
                # bounding box
            ])
            # print("Object: ", obj_dict)
            bbox_name = 'objectOrientedBoundingBox' if obj_dict['objectOrientedBoundingBox'] != None else 'axisAlignedBoundingBox'
            objects.add((
                obj_dict['objectType'],
                # self.get_name_object(obj_dict['name']),
                obj_dict['objectId'],
                # obj_dict['name'],
                tuple(self.round_number(obj_dict['position'], 2)),
                tuple(self.round_number(obj_dict['rotation'], 2)),
                # obj_dict['parentReceptacles'],
                tuple(obj_dict['receptacleObjectIds']) if obj_dict['receptacleObjectIds'] != None else (),
                self.get_min_by_axis(obj_dict[bbox_name]["cornerPoints"])
            ))
        return cond_objs , objects
    
    def save_data_by_axis(self, dict_data, base_name, array):
        axis_names = ['-x', '-y', '-z']
        for (axis, item) in zip(axis_names, array):
            dict_data[base_name + axis].append(item)

    # def save_bbox(self, dict_data, bbox):
    #     assert len(bbox) == 4, "size error of bbox, it must be of size 4"
    #     dict_data['cmin'].append(bbox[0])
    #     dict_data['rmin'].append(bbox[1])
    #     dict_data['cmax'].append(bbox[2])
    #     dict_data['rmax'].append(bbox[3])

    def save_data_navigation(self, dict_navigation, key, objects_data, path_image, degrees):
        # print("key: ", key)
        for object_data in objects_data:
            dict_navigation['ag-action'].append(key[0])
            self.save_data_by_axis(dict_navigation, 'ag-pos', key[1])
            self.save_data_by_axis(dict_navigation, 'ag-rot', key[2])
            dict_navigation['degrees'].append(degrees)
            # dict_navigation['obj-type'].append(object_data[0])
            dict_navigation['obj-id'].append(object_data[0])
            # self.save_data_by_axis(dict_navigation, 'obj-pos', object_data[2])
            # self.save_data_by_axis(dict_navigation, 'obj-rot', object_data[3])
            dict_navigation['obj-distance'].append(object_data[1])
            # self.save_bbox(dict_navigation, object_data[5])
            dict_navigation['path'].append(path_image) 

    def save_image(self, image_name, event):
        cv2.imwrite(image_name, event.cv2img) 
    
    def get_dict_navigation(self):
        # dict_navigation = {
        #     'ag-action': [], 'ag-pos-x': [], 'ag-pos-y': [], 'ag-pos-z': [], 
        #     'ag-rot-x': [], 'ag-rot-y': [], 'ag-rot-z': [], 'obj-type': [], 
        #     'obj-id': [], 'obj-pos-x': [], 'obj-pos-y': [], 'obj-pos-z': [], 
        #     'obj-rot-x': [], 'obj-rot-y': [], 'obj-rot-z': [], 'obj-distance': [],
        #     'cmin': [], 'rmin': [], 'cmax': [], 'rmax': [], 'path': []
        #     }
        dict_navigation = {
            'ag-action': [], 'degrees': [], 'ag-pos-x': [], 'ag-pos-y': [], 'ag-pos-z': [], 
            'ag-rot-x': [], 'ag-rot-y': [], 'ag-rot-z': [], #'obj-type': [], 
            'obj-id': [], 'obj-distance': [], 'path': []
            }
        for key in self.dict_agent:
            object_data = self.dict_agent[key]['objects']
            image_path = self.dict_agent[key]['image']
            degrees = self.dict_agent[key]['degrees']
            self.save_data_navigation(dict_navigation, key, object_data, image_path, degrees)
        return dict_navigation
    
    # def save_data_by_axis_bbox(self, dict_objects, base_name, bbox):
    #     array = np.array(bbox)
    #     print(array, array.shape)
    #     assert array.shape == (8, 3)
    #     x_min = np.min(array[:, 0])
    #     y_min = np.min(array[:, 1])
    #     z_min = np.min(array[:, 2])
    #     self.save_data_by_axis(dict_objects, base_name, [x_min, y_min, z_min])

    def get_dict_objects(self):
        dict_objects = {
            'obj-type': [], 'obj-id': [], 'obj-pos-x': [], 'obj-pos-y': [], 'obj-pos-z': [],
            'obj-rot-x': [], 'obj-rot-y': [], 'obj-rot-z': [], #'parentReceptacles': [],
            'receptacleObjectIds': [], 'objOrBBox-x': [], 'objOrBBox-y': [], 'objOrBBox-z': []
        }
        for t in self.data_objects['objects']:
            dict_objects['obj-type'].append(t[0])
            dict_objects['obj-id'].append(t[1])
            self.save_data_by_axis(dict_objects, 'obj-pos', t[2])
            self.save_data_by_axis(dict_objects, 'obj-rot', t[3])
            # dict_objects['parentReceptacles'].append(t[4])
            dict_objects['receptacleObjectIds'].append(t[4])
            # self.save_data_by_axis_bbox(dict_objects, 'objOrBBox', t[5])
            self.save_data_by_axis(dict_objects, 'objOrBBox', t[5])
        return dict_objects

    def collect_data(self, event, action, v_objects):
        # print("Visible objects: ", v_objects)
        position = self.round_number(event.metadata['agent']['position'], 2)
        rotation = self.round_number(event.metadata['agent']['rotation'], 2)
        action_name = THORActions.get_action_name(action)
        key = (action_name, position, rotation)
        if key not in self.dict_agent:
            self.dict_agent[key] = {'objects': [], 'image': ''}
            # cond_objs = self.get_object_data(objects)
            self.dict_agent[key]["degrees"] = AGENT_ROTATION_DEG
            cond_objs, objects = self.get_object_data(v_objects)
            self.dict_agent[key]['objects'] = cond_objs
            if self.data_objects['objects']:
                self.data_objects['objects'].update(objects)
            else:
                self.data_objects['objects'] = objects
            # save image
            image_name = os.path.join(self.image_path, 'img_' + str(self.index) + '.png')
            self.dict_agent[key]['image'] = image_name
            self.save_image(image_name, event)
            self.index += 1

    def save_data(self):
        dict_navigation = self.get_dict_navigation()
        df_navigation = pd.DataFrame(dict_navigation)
        df_navigation.to_csv(self.navigation_path)
        dict_objects = self.get_dict_objects()
        df_objects = pd.DataFrame(dict_objects)
        df_objects.to_csv(self.objects_path)

    