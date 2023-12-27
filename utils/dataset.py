"""
Dataset class for organizing datasets with:
Pointcloud + Pointcloud_o3d
Keypointindex + Keypoint
PCpairs + pairgt
"""


import os
import numpy as np
import abc

import torch
from torch.utils.data import Dataset
from utils.r_eval import compute_R_diff,quaternion_from_matrix
from utils.utils import read_pickle, make_non_exists_dir
import open3d as o3d

class EvalDataset(abc.ABC):
    @abc.abstractmethod
    def get_pair_ids(self):
        pass

    @abc.abstractmethod
    def get_cloud_ids(self):
        pass

    @abc.abstractmethod
    def get_pc_dir(self,cloud_id):
        pass
    
    @abc.abstractmethod
    def get_key_dir(self,cloud_id):
        pass

    @abc.abstractmethod
    def get_transform(self,id0,id1):
        # note the order!
        # target: id0, source: id1
        # R @ pts1 + t = pts0
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    @abc.abstractmethod
    def get_kps(self,cloud_id):
        pass

#The dataset class for original/ground truth datas
class ThrDMatchPartDataset(EvalDataset):
    def __init__(self,root_dir,stationnum,gt_dir=None):
        self.root=root_dir
        if gt_dir==None:
            self.gt_dir=f'{self.root}/PointCloud/gt.log'
        else:
            self.gt_dir=gt_dir
        self.kps_pc_fn=[f'{self.root}/Keypoints_PC/cloud_bin_{k}Keypoints.npy' for k in range(stationnum)]
        self.kps_fn=[f'{self.root}/Keypoints/cloud_bin_{k}Keypoints.txt' for k in range(stationnum)]
        self.pc_ply_paths=[f'{self.root}/PointCloud/cloud_bin_{k}.ply' for k in range(stationnum)]
        self.pc_npy_paths=[f'{self.root}/PointCloud/cloud_bin_{k}.npy' for k in range(stationnum)]
        self.pc_txt_paths=[f'{self.root}/PointCloud/cloud_bin_{k}.txt' for k in range(stationnum)]
        self.pair_id2transform=self.parse_gt_fn(self.gt_dir)
        self.pair_ids=[tuple(v.split('-')) for v in self.pair_id2transform.keys()]
        self.pc_ids=[str(k) for k in range(stationnum)]
        self.pair_num=self.get_pair_nums()
        self.name='3dmatch/kitchen'

    #function for gt(input: gt.log)
    @staticmethod
    def parse_gt_fn(fn):
        with open(fn,'r') as f:
            lines=f.readlines()
            pair_num=len(lines)//5
            pair_id2transform={}
            for k in range(pair_num):
                id0,id1=np.fromstring(lines[k*5],dtype=np.float32,sep='\t')[0:2]
                id0=int(id0)
                id1=int(id1)
                row0=np.fromstring(lines[k*5+1],dtype=np.float32,sep=' ')
                row1=np.fromstring(lines[k*5+2],dtype=np.float32,sep=' ')
                row2=np.fromstring(lines[k*5+3],dtype=np.float32,sep=' ')
                transform=np.stack([row0,row1,row2],0)
                pair_id2transform['-'.join((str(id0),str(id1)))]=transform

            return pair_id2transform

    def get_pair_ids(self):
        return self.pair_ids

    def get_pair_nums(self):
        return len(self.pair_ids)

    def get_cloud_ids(self):
        return self.pc_ids

    def get_pc_dir(self,cloud_id):
        return self.pc_ply_paths[int(cloud_id)]

    def get_pc(self,pc_id):
        if os.path.exists(self.pc_ply_paths[int(pc_id)]):
            pc=o3d.io.read_point_cloud(self.pc_ply_paths[int(pc_id)])
            return np.array(pc.points)
        else:
            pc=np.load(self.pc_npy_paths[int(pc_id)])
            return pc
    
    def get_pc_o3d(self,pc_id):
        return o3d.io.read_point_cloud(self.pc_ply_paths[int(pc_id)])
            
    def get_key_dir(self,cloud_id):
        return self.kps_fn[int(cloud_id)]

    def get_transform(self, id0, id1):
        return self.pair_id2transform['-'.join((id0,id1))]

    def get_name(self):
        return self.name

    def get_kps(self, cloud_id):
        if not os.path.exists(self.kps_pc_fn[int(cloud_id)]):
            pc=self.get_pc(cloud_id)
            key_idxs=np.loadtxt(self.kps_fn[int(cloud_id)]).astype(np.int)
            keys=pc[key_idxs]
            make_non_exists_dir(f'{self.root}/Keypoints_PC')
            np.save(self.kps_pc_fn[int(cloud_id)],keys)
            return keys
        return np.load(self.kps_pc_fn[int(cloud_id)])

#Get dataset items with the dataset name(output: dict)
def get_dataset_name(dataset_name,origin_data_dir):

    if dataset_name=='demo':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=['kitchen']
        stationnums=[2]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets

    if dataset_name=='kitti':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=['8','9','10']
        stationnums=[4071,1591,1201]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets

    if dataset_name=='CS':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=['8','9']
        stationnums=[4071,4071]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets
    
    if dataset_name=='CTCS':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=['10']
        stationnums=[10000]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets

    else:
        raise NotImplementedError


def get_dataset(cfg,training=True):
    if training:
        dataset_name=cfg.trainset_name
    else:
        dataset_name=cfg.testset_name
    origin_dir=cfg.origin_data_dir
    return get_dataset_name(dataset_name,origin_dir)
    

#train dataset 
class Enhanced_train_dataset(Dataset):
    def __init__(self,cfg,is_training=True):
        self.cfg=cfg
        self.output_dir= f'{self.cfg.output_cache_fn}/train'
        self.is_training=is_training
        self.Rgroup=np.load(f'{self.cfg.SO3_related_files}/Rotation_8.npy').astype(np.float32)
        if self.is_training:
            self.name_pair_ids=read_pickle(cfg.train_pcpair_list_fn) #list: name id0 id1 pt1 pt2
        else:
            self.name_pair_ids=read_pickle(cfg.val_pppair_list_fn)[0:384000]   #list: name id0 id1 pt1 pt2

    def R2DR_id(self,R):
        min_diff=180
        best_id=0
        for R_id in range(self.Rgroup.shape[0]):
            R_diff=compute_R_diff(self.Rgroup[R_id],R)
            if R_diff<min_diff:
                min_diff=R_diff
                best_id=R_id
        return best_id

    def __getitem__(self,index):
        if self.is_training:
            item=torch.load(f'{self.output_dir}/train_val_batch/trainset/{index}.pth')
            return item
        
        else:
            item=torch.load(f'{self.output_dir}/train_val_batch/valset/{index}.pth')
            return item
        

    def __len__(self):
        return len(self.name_pair_ids)


        
name2traindataset={
    "Enhanced_train_dataset":Enhanced_train_dataset
}

