"""
Feature extractor for feature extraction:
(1)extractor: group feature;
(2)extractor_dr_index: group feature pair-->coarse rotation(index in 8);
"""



import os,sys
sys.path.append('..')
import torch
import glob
import numpy as np
from tqdm import tqdm
from utils.utils import make_non_exists_dir,to_cuda
from utils.network import name2network


class extractor():
    def __init__(self,cfg):
        self.cfg=cfg
        self.network=name2network[f'{self.cfg.test_network_type}'](self.cfg).cuda()
        self.model_fn=f'{self.cfg.model_fn}/{self.cfg.train_network_type}/model.pth'
        self.best_model_fn=f'{self.cfg.model_fn}/{self.cfg.train_network_type}/model_best.pth'

    #Model
    def _load_model(self):
        best_para,start_step=0,0
        if os.path.exists(self.best_model_fn):
            checkpoint=torch.load(self.best_model_fn)
            best_para = checkpoint['best_para']
            self.network.load_state_dict(checkpoint['network_state_dict'])
            print(f'Resuming best para {best_para}')
        else:
            raise ValueError("No model exists")

    #Extract
    def Extract(self,dataset):
        #data input 5000*32*8
        #output: 5000*32*8->save
        self._load_model()
        self.network.eval()
        FCGF_input_dir=f'{self.cfg.output_cache_fn}/test/{dataset.name}/FCGF_Input_Group_feature'
        MSReg_output_dir=f'{self.cfg.output_cache_fn}/test/{dataset.name}/MSReg_Output_Group_feature'
        make_non_exists_dir(MSReg_output_dir)
        print(f'Extracting the MSReg descriptors on {dataset.name}')
        pc_fns = glob.glob(f'{FCGF_input_dir}/*.npy')
        for fn in tqdm(pc_fns):
            pc_id = str.split(fn,'/')[-1][:-4]
            if os.path.exists(f'{MSReg_output_dir}/{pc_id}.npy'):continue
            Input_feature=np.load(f'{FCGF_input_dir}/{pc_id}.npy') #5000*32*60
            output_feature=[]
            bi=0
            while(bi*self.cfg.test_batch_size<Input_feature.shape[0]):
                start=bi*self.cfg.test_batch_size
                end=(bi+1)*self.cfg.test_batch_size
                batch=torch.from_numpy(Input_feature[start:end,:,:].astype(np.float32)).cuda()
                with torch.no_grad():
                    batch_output=self.network(batch)
                output_feature.append(batch_output['eqv'].cpu().numpy())
                bi+=1
            output_feature=np.concatenate(output_feature,axis=0)
            np.save(f'{MSReg_output_dir}/{pc_id}.npy',output_feature) #5000*32*60

class extractor_dr_index():
    def __init__(self,cfg):
        self.cfg=cfg
        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/8_8.npy').reshape([-1]).astype(np.int)).cuda()

    def Des2R_torch(self,des1_eqv,des2_eqv):#beforerot afterrot
        des1_eqv=des1_eqv[:,self.Nei_in_SO3].reshape([-1,8,8])
        cor=torch.einsum('fag,fg->a',des1_eqv,des2_eqv)
        return torch.argmax(cor)

    def Batch_Des2R_torch(self,des1_eqv,des2_eqv):#beforerot afterrot
        B,F,G=des1_eqv.shape
        des1_eqv=des1_eqv[:,:,self.Nei_in_SO3].reshape([B,F,8,8])
        cor=torch.einsum('bfag,bfg->ba',des1_eqv,des2_eqv)
        return torch.argmax(cor,dim=1)
  
    def Rindex(self,dataset):
        match_dir=f'{self.cfg.output_cache_fn}/test/{dataset.name}/Match'
        Save_dir=f'{match_dir}/DR_index'
        make_non_exists_dir(Save_dir)
        datasetname=dataset.name
        Feature_dir=f'{self.cfg.output_cache_fn}/test/{datasetname}/MSReg_Output_Group_feature'
        
        print(f'extract the drindex of the matches on {dataset.name}')
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            if os.path.exists(f'{Save_dir}/{id0}-{id1}.npy'):continue
            match_pps=np.load(f'{match_dir}/{id0}-{id1}.npy')
            feats0=np.load(f'{Feature_dir}/{id0}.npy') #5000,32,60
            feats1=np.load(f'{Feature_dir}/{id1}.npy') #5000,32,60
            feats0=torch.from_numpy(feats0[match_pps[:,0]].astype(np.float32)).cuda()
            feats1=torch.from_numpy(feats1[match_pps[:,1]].astype(np.float32)).cuda()
            pre_idxs=self.Batch_Des2R_torch(feats1,feats0).cpu().numpy()
            np.save(f'{Save_dir}/{id0}-{id1}.npy',pre_idxs)
   
name2extractor={
    'extractor':extractor
}