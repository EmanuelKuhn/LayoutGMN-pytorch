#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:05:38 2019
    - Dataloader for triplet of graph-data.
    - trainset only includes ids that has valid positive-pairs (iou > threshold.) e.g. 0.6
    - randomly sample anchors [same as previous]
    - for selected anchor, find an positive from positive set (randomly choose if multiple exits)
    - To find the negative,
        1) randomly choose any image except from the pos list
        2) only choose images whose iou is beteen some range (l_iou, h_iou) --> (0.2-0.4)
           The higher the iou the harder is the negative.
        3) Only choose hard examples just below the postive threshold, and above some iou e.g. (0.4-0.7))
"""

import torch
from torch.utils.data import Dataset
import torch.utils.data as data
import os
from PIL import Image

from torchvision import transforms
import numpy as np
import random
import pickle
import torch.nn.functional as F
from collections import defaultdict
import random


def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)
        print('Saved to {}.'.format(fname))

def pickle_load(fname):
    with open(fname, 'rb') as pf:
         data = pickle.load(pf)
         print('Loaded {}.'.format(fname))
         return data


def get_id_from_path(path):
    return path.split("/")[-1].split(".")[0]

def get_splits(path="layoutgmn_data_changed_splits/FP_data.p"):
    fp_data = pickle_load(path)


    splits = dict()

    for key in fp_data:
        splits[key] = list(map(get_id_from_path, fp_data[key]))

    return splits

class RICO_TripletDataset(Dataset):
    
    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, if_shuffle = (split=='train'))
        self.iterators[split] = 0
   
    def __init__(self,opt):
        self.opt = opt
        
        self.info = pickle.load(open('layoutgmn_data/FP_box_info_list.pkl', 'rb'))

        self.sg_geometry_dir = 'fp_data/geometry-directed/'                
        print('\nLoading geometric graphs and features from {}\n'.format(self.sg_geometry_dir))
        
        self.batch_size = self.opt.batch_size
        self.transform = None
       
        self.geometry_relation = True
        self.geom_feat_size = 8
        
        #% get the anchor-positive-negative apn_dict dictionary 
        self.apn_dict = pickle_load(self.opt.apn_dict_path)

        train_uis = list(self.apn_dict.keys())
        
        # Separate out indexes for the train and test 
        splits = get_splits("layoutgmn_data_changed_splits/FP_data.p")

        
        # # Donot use the images with large number of components. 
        # uis_ncomponent_g100 = pickle.load(open('data/ncomponents_g100_imglist.pkl', 'rb'))
        self.orig_train_uis = list(set(splits["train_fps"]) & set([x['id'] for x in self.info]))  #some img (e.g. img with no comp are removed in info)
        # self.orig_train_uis = list(set(self.orig_train_uis) - set(uis_ncomponent_g100))
        
        train_uis = list(set(train_uis) & set([x['id'] for x in self.info]))
        
        #Instantiate the ix
        self.split_ix = {'train': [],  'gallery': [], 'val':[]}
        
        # id2index: the dataset is indexed with indicies of the list info:
        self.id2index = defaultdict(dict)
        
        train_uis_set = set(train_uis)
        val_fps_set = set(splits["val_fps"])

        for ix in range(len(self.info)):
            img = self.info[ix]['id']

            assert isinstance(img, str), f"img is not a string: {img}"

            self.id2index[img] = ix  
            if img in train_uis_set:
                self.split_ix['train'].append(ix)
            elif img in val_fps_set:
                self.split_ix['val'].append(ix)
        
        self.iterators = {'train': 0,  'val': 0}
        
        for split in self.split_ix.keys():
            print('assigned %d images to split %s'%(len(self.split_ix[split]), split))
        
        self._prefetch_process = {} # The three prefetch process 
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train', num_workers = 4)
        
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

    def __len__(self):
        """Returns the number of triplet anchors"""
        return len(self.split_ix['train'])

    def __getitem__(self, index):
#        ix = index #self.split_ix[index]
        
        sg_data = self.get_graph_data(index)
   
        return (sg_data, 
                # img,
                index)   
    
     
    def get_pairs(self, ix):
        id_a = self.info[ix]['id'] 
        pos_pool = self.apn_dict[id_a]['ids_pos']
        c_ind = random.choice(range(len(pos_pool)))
        id_p = pos_pool[c_ind]
        #iou_p = self.anc_pos_dict[id_a]['pos_ious'][c_ind]
        iou_p_norm = self.apn_dict[id_a]['ious_pos'][c_ind] 
        
        if self.opt.hardmining:
            #print('hard_negative mining')
            neg_pool = set(self.apn_dict[id_a]['ids_b2040']) | set(self.apn_dict[id_a]['ids_b4050']) | set(self.apn_dict[id_a]['ids_b5060'])

            assert len(neg_pool & set(set(self.apn_dict[id_a]['ids_pos']))) == 0

            neg_pool = list(neg_pool)
            
            if len(neg_pool) == 0:
                # ids_b5060 = self.apn_dict[id_a]['ids_b5060']
                # ids_b4050 = self.apn_dict[id_a]['ids_b4050']
                ids_iou1 = self.apn_dict[id_a]['ids_iou1']
                # sample from any image except pos, anchor-itself, and ids with iou between [0.4-0.6]
                neg_pool = list(set(self.orig_train_uis) - set(pos_pool) - set([id_a]) -  set(ids_iou1))   
        else:
            ids_b5060 = self.apn_dict[id_a]['ids_b5060']
            ids_b4050 = self.apn_dict[id_a]['ids_b4050']
            ids_iou1 = self.apn_dict[id_a]['ids_iou1']
            # sample from any image except pos, anchor-itself, and ids with iou between [0.4-0.6]
            neg_pool = list(set(self.orig_train_uis) - set(pos_pool) - set([id_a]) - set(ids_b5060) - set(ids_b4050) -  set(ids_iou1))
        
        id_n = random.choice(neg_pool)
        iou_n_norm = 0 # Need to implement this, may be useful for hard negative
        
        
        # Hard negative say ids with ious between 20-50: 
#        ids_b2040 = self.apn_dict[id_a]['ids_b2040']  # Note: these are not norm_iou
#        ids_b4050 = self.apn_dict[id_a]['ids_b4050']
#        neg_pool = ids_b2040 + ids_b4050
#        id_n = random.choice(neg_pool)
#        iou_n_norm = 0
       
        return id_a, id_p, id_n,  iou_p_norm, iou_n_norm
            
       
            
    def get_graph_data(self, index):
        #self.opt.use_box_feats = True
        image_id = self.info[index]['id']
#        sg_use = np.load(self.sg_data_dir + image_id + '.npy', encoding='latin1', allow_pickle=True)[()]
    
        geometry_path = os.path.join(self.sg_geometry_dir, image_id + '.npy')
        rela = np.load(geometry_path, allow_pickle=True)[()] # dict contains keys of edges and feats
        
        obj = self.info[index]['class_id']
        obj = np.reshape(obj, (-1, 1))
    
        box = self.info[index]['xywh']                
        
        if self.opt.use_box_feats:
            box_feats = self.get_box_feats(box)
            sg_data = {'obj': obj, 'box_feats': box_feats, 'rela': rela, 'box':box}
        else:
            sg_data = {'obj': obj,  'rela': rela, 'box':box}
    
        return sg_data
    
    
    def get_graph_data_by_id(self, image_id):  
        # combines get_graph_data & getitem functions.
        geometry_path = os.path.join(self.sg_geometry_dir, image_id + '.npy')
        rela = np.load(geometry_path, allow_pickle=True)[()] # dict contains keys of edges and feats
        
        index = self.id2index[image_id]
        assert(image_id == self.info[index]['id'])
        
        obj = self.info[index]['class_id']
        obj = np.reshape(obj, (-1, 1))
    
        box = self.info[index]['xywh']                
        
        if self.opt.use_box_feats:
            box_feats = self.get_box_feats(box)
            sg_data = {'obj': obj, 'box_feats': box_feats, 'rela': rela, 'box':box}
        else:
            sg_data = {'obj': obj,  'rela': rela, 'box':box}
               
        return (sg_data, 
                # img,
                index)       
    
    
    def get_box_feats(self,box):

        boxes = np.array(box)
        W, H = 256, 256  # We know the height and weight for all semantic UIs are 2560 and 1400
        
        x1, y1, w, h = np.hsplit(boxes,4)
        x2, y2 = x1+w, y1+h 
        
        box_feats = np.hstack((0.5 * (x1 + x2) / W, 0.5 * (y1 + y2) / H, w/W, h/H, w*h/(W*H)))
        #box_feats = box_feat / np.linalg.norm(box_feats, 2, 1, keepdims=True)
        return box_feats
    
    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size
        sg_batch_a = []
        sg_batch_p = []
        sg_batch_n = []
        

#        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')
        infos = []
        
        #images_a = []
        #images_p = []
        #images_n = []
        
        wrapped = False
        
        for i in range(batch_size):
            # fetch image
            #tmp_sg_a, tmp_img_a, ix_a, tmp_wrapped = self._prefetch_process[split].get()
            tmp_sg_a, ix_a, tmp_wrapped = self._prefetch_process[split].get()

            # id_p, id_n, iou_p, iou_p_norm, iou_n, iou_n_norm  = self.get_pairs(ix_a)
            id_a, id_p, id_n,  iou_p_norm, iou_n_norm = self.get_pairs(ix_a)

            '''
            tmp_sg_p, tmp_img_p, ix_p = self.get_graph_data_by_id(id_p)
            tmp_sg_n, tmp_img_n, ix_n = self.get_graph_data_by_id(id_n)
            '''

            #tmp_sg_a_new, ix_a_new = self.get_graph_data_by_id(id_a)
            tmp_sg_p, ix_p = self.get_graph_data_by_id(id_p)
            tmp_sg_n, ix_n = self.get_graph_data_by_id(id_n)
             
            sg_batch_a.append(tmp_sg_a)
            #images_a.append(tmp_img_a)
            
            sg_batch_p.append(tmp_sg_p)
            #images_p.append(tmp_img_p)
            
            sg_batch_n.append(tmp_sg_n)
            #images_n.append(tmp_img_n)
            
          
            
           # record associated info as well
            info_dict = {}
            info_dict['ix_a'] = ix_a
            info_dict['id_a'] = self.info[ix_a]['id']
            info_dict['id_p'] = id_p
            info_dict['id_n'] = id_n
            #info_dict['iou_p'] = iou_p
            #info_dict['iou_n'] = iou_n
            info_dict['iou_p_norm'] = iou_p_norm
            info_dict['iou_n_norm'] = iou_n_norm
            
            infos.append(info_dict)
            
            if tmp_wrapped:
                wrapped = True
                break
            
        data = {}

        '''
        max_box_len_a = max([_['obj'].shape[0] for _ in sg_batch_a])
        max_box_len_p = max([_['obj'].shape[0] for _ in sg_batch_p])
        max_box_len_n = max([_['obj'].shape[0] for _ in sg_batch_n])
        '''


        
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        data['sg_data_a'] = self.batch_sg(sg_batch_a)#, max_box_len_a)
        data['sg_data_p'] = self.batch_sg(sg_batch_p)#, max_box_len_p)
        data['sg_data_n'] = self.batch_sg(sg_batch_n)#, max_box_len_n)

        #data['images_a'] = torch.stack(images_a)
        #data['images_p'] = torch.stack(images_p)
        #data['images_n'] = torch.stack(images_n)
        
        return data
    
    def batch_sg(self, sg_batch):#, max_box_len):
        "batching object, attribute, and relationship data"
        obj_batch = [_['obj'] for _ in sg_batch]
        rela_batch = [_['rela'] for _ in sg_batch]
        #box_batch = [_['box'] for _ in sg_batch]
        
        sg_data = []
        for i in range(len(obj_batch)):
            sg_data.append(dict())

        '''
        # obj labels, shape: (B, No, 1)
        sg_data['obj_labels'] = []
        for i in range(len(obj_batch)):
            sg_data['obj_labels'].append(obj_batch[i])
        sg_data['obj_labels'] = np.array(sg_data['obj_labels'])



        sg_data['obj_boxes'] = []
        for i in range(len(box_batch)):
            sg_data['obj_boxes'].append(box_batch[i])
        sg_data['obj_boxes'] = np.array(sg_data['obj_boxes'])
        '''
        
        if self.opt.use_box_feats:
            box_feats_batch = [_['box_feats'] for _ in sg_batch]
            #obj_labels = -1*np.ones([max_box_len, 1], dtype = 'int')
            #sg_data['box_feats'] = []
            for i in range(len(box_feats_batch)):
                sg_data[i]['box_feats'] = box_feats_batch[i]
                #obj_labels[:obj_batch[i].shape[0]] = obj_batch[i]
                sg_data[i]['room_ids'] = obj_batch[i]

            for i in range(len(rela_batch)):
                sg_data[i]['rela_edges'] = rela_batch[i]['edges']
                sg_data[i]['rela_feats'] = rela_batch[i]['feats']

            #sg_data['box_feats'] = np.array(sg_data['box_feats'])
        '''
        # rela
        sg_data['rela_edges'] = []
        sg_data['rela_feats'] = []

        for i in range(len(rela_batch)):
            sg_data['rela_edges'].append(rela_batch[i]['edges'])
            sg_data['rela_feats'].append(rela_batch[i]['feats'])

        sg_data['rela_edges'] = np.array(sg_data['rela_edges'])
        sg_data['rela_feats'] = np.array(sg_data['rela_feats'])
        '''

        return sg_data    
#%%
class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle, num_workers = 4):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
#        self.opt =opt
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle
        self.num_workers = num_workers

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers= self.num_workers,#1, # 4 is usually enough
                                            worker_init_fn=None,
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = next(self.split_loader)
        if wrapped:
            self.reset()

        assert tmp[-1] == ix, "ix not equal"

        tmp = list(tmp)

        return tmp + [wrapped]
    
    
#%%
class SubsetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
        #return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)