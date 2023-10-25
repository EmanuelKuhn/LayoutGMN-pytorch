from collections import defaultdict
import os
import pickle

import numpy as np


class SGDataset:
    def __init__(self, use_box_feats, box_info_list_pkl="layoutgmn_data/FP_box_info_list.pkl", sg_geometry_dir='../training_scripts/fp_data/geometry-directed/') -> None:
        self.sg_geometry_dir = sg_geometry_dir

        with open(box_info_list_pkl, 'rb') as f:
            self.info = pickle.load(f)

        self.id2index = defaultdict(dict)

        for ix in range(len(self.info)):
            img = self.info[ix]['id']
            self.id2index[img] = ix
        
        self.use_box_feats = use_box_feats
        
    
    def get_graph_data_by_id(self, image_id):
        geometry_path = os.path.join(self.sg_geometry_dir, image_id + '.npy')
        rela = np.load(geometry_path, allow_pickle=True)[()]  # dict contains keys of edges and feats

        index = self.id2index[image_id]
        assert (image_id == self.info[index]['id'])

        obj = self.info[index]['class_id']
        obj = np.reshape(obj, (-1, 1))
        #one_hot_encoded_obj = self.class_labels_to_one_hot(obj)

        box = self.info[index]['xywh']

        if self.use_box_feats:
            box_feats = self.get_box_feats(box)
            #box_feats = np.concatenate((box_feats, one_hot_encoded_obj), axis=-1)
            sg_data = {'obj': obj, 'box_feats': box_feats, 'rela': rela, 'box':box}
        else:
            sg_data = {'obj': obj,  'rela': rela, 'box':box}

        '''
        new_sg_data = {}
        new_sg_data['box_feats'] = sg_data['box_feats']
        new_sg_data['room_ids'] = sg_data['obj']
        new_sg_data['rela_edges'] = sg_data['rela']['edges']
        new_sg_data['rela_feats'] = sg_data['rela']['feats']

        return new_sg_data
        '''

        return sg_data
    
    @staticmethod
    def get_box_feats(box):
        boxes = np.array(box)
        W, H = 1440, 2560  # We know the height and weight for all semantic UIs are 2560 and 1400
        
        x1, y1, w, h = np.hsplit(boxes,4)
        x2, y2 = x1+w, y1+h 
        
        box_feats = np.hstack((0.5 * (x1 + x2) / W, 0.5 * (y1 + y2) / H, w/W, h/H, w*h/(W*H)))
        #box_feats = box_feat / np.linalg.norm(box_feats, 2, 1, keepdims=True)
        return box_feats

    def get_batch(self, image_ids):
        """Return a batch in the form of batch_sg(sg_batch) of the specified image_ids"""

        assert isinstance(image_ids, list), "Expected image_id to be a list"
        assert isinstance(image_ids[0], str), "Expected list of ids, where each id is a string"

        sg_datas = [self.get_graph_data_by_id(id) for id in image_ids]

        return self.batch_sg(sg_datas)

    
    def batch_sg(self, sg_batch):
        """Helper function to create a batch of a list of sg_data dictionaries.

        E.g. sg_batch = sg_dataset.batch_sg([sg_dataset[0], sg_dataset[1]])
        
        batching object, attribute, and relationship data"""
        obj_batch = [_['obj'] for _ in sg_batch]
        rela_batch = [_['rela'] for _ in sg_batch]
        # box_batch = [_['box'] for _ in sg_batch]

        sg_data = []
        for i in range(len(obj_batch)):
            sg_data.append(dict())

        if self.use_box_feats:
            box_feats_batch = [_['box_feats'] for _ in sg_batch]
            # sg_data['box_feats'] = []
            for i in range(len(box_feats_batch)):
                sg_data[i]['box_feats'] = box_feats_batch[i]
                sg_data[i]['room_ids'] = obj_batch[i]

            for i in range(len(rela_batch)):
                sg_data[i]['rela_edges'] = rela_batch[i]['edges']
                sg_data[i]['rela_feats'] = rela_batch[i]['feats']

        return sg_data

