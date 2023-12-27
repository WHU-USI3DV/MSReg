"""
Evaluator class for the whole pipeline of MSReg containing:
(1) Given FCGF group feature from ./testset.py.
(2) MSReg group feature-->MSReg-Desc                      extractor
(3) MSReg-Desc-->inv-->matmul matcher-->pps               matcher
(4) pps+MSReg-Desc-->coarse rotation                      extractor
(5) MSReg                                                 estimator
or namely, tester.
"""


import os,sys
sys.path.append('..')
import numpy as np
from tqdm import tqdm
from utils.dataset import get_dataset
from utils.utils import transform_points
from utils.r_eval import compute_R_diff
from test.extractor import name2extractor,extractor_dr_index
from test.matcher import name2matcher
from test.estimator import name2estimator

class evaluator:
    def __init__(self,cfg,max_iter):
        self.max_iter=max_iter
        self.cfg=cfg
        self.extractor=name2extractor[self.cfg.extractor](self.cfg)
        self.matcher=name2matcher[self.cfg.matcher](self.cfg)
        self.drindex_extractor=extractor_dr_index(self.cfg)
        est=self.cfg.estimator
        if self.max_iter>500:
            est='estimator_mul'
        self.estimator=name2estimator[est](self.cfg)

    def run_onescene(self,dataset):
        self.extractor.Extract(dataset)
        self.matcher.match(dataset)
        self.drindex_extractor.Rindex(dataset)
        self.estimator.ransac(dataset,self.max_iter)

    def Feature_match_Recall(self,dataset):
        pair_fmrs=[]
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            #match
            matches=np.load(f'{self.cfg.output_cache_fn}/test/{dataset.name}/Match/{id0}-{id1}.npy')
            keys0=dataset.get_kps(id0)[matches[:,0],:]
            keys1=dataset.get_kps(id1)[matches[:,1],:]
            #gt
            gt=dataset.get_transform(id0,id1)
            #ratio
            keys1=transform_points(keys1,gt)
            dist=np.sqrt(np.sum(np.square(keys0-keys1),axis=-1))
            pair_fmr=np.mean(dist<self.cfg.ok_match_dist_threshold) #ok ratio in one pair
            pair_fmrs.append(pair_fmr)                              
        pair_fmrs=np.array(pair_fmrs)                               #ok ratios in one scene
        return pair_fmrs

    def eval(self):
        datasets=get_dataset(self.cfg,False)
        max_iter=1000
        all_pair_fmrs=[]
        for scene,dataset in datasets.items():
            if scene=='wholesetname':continue
            self.run_onescene(dataset)
            print(f'eval the FMR result on {dataset.name}')
            pair_fmrs=self.Feature_match_Recall(dataset)
            all_pair_fmrs.append(pair_fmrs)
        all_pair_fmrs=np.concatenate(all_pair_fmrs,axis=0)

        #RR
        datasetname=datasets['wholesetname']
        RRs=[]
        whole_ok_num=0
        whole_all_num=0
        whole_rre=[]
        whole_rte=[]
        for name,dataset in datasets.items():
            if name=='wholesetname':
                continue
            oknum=0
            wholenum=0
            for pair in dataset.pair_ids:
                writer=open(f'{self.cfg.output_cache_fn}/test/{dataset.name}/Match/MSReg/{self.cfg.max_iter}iters/pre_RRE&RTE.log','a')
                id0,id1=pair
                wholenum+=1
                gt=dataset.get_transform(id0,id1)
                pre=np.load(f'{self.cfg.output_cache_fn}/test/{dataset.name}/Match/MSReg/{self.cfg.max_iter}iters/{id0}-{id1}.npz')['trans']
                tdiff = np.linalg.norm(pre[0:3,-1]-gt[0:3,-1])
                Rdiff=compute_R_diff(gt[0:3,0:3],pre[0:3,0:3])
                
                if tdiff<=2 and Rdiff<=5:
                    oknum+=1
                    writer.write(f'{int(id0)}\t{int(id1)}\tSucceed!\t')
                    writer.write(f'RRE:{Rdiff}\tRTE:{tdiff}\n')
                    if Rdiff<5:
                        whole_rre.append(Rdiff)
                    if tdiff<2:
                        whole_rte.append(tdiff)
                else:
                    writer.write(f'{int(id0)}\t{int(id1)}\tFailed...\t')
                    writer.write(f'RRE:{Rdiff}\tRTE:{tdiff}\n')
                writer.close()
            RRs.append(oknum/wholenum)
            whole_ok_num+=oknum
            whole_all_num+=wholenum
        Mean_Registration_Recall = np.mean(np.array(RRs))
        rre = np.mean(np.array(whole_rre))
        rte = np.mean(np.array(whole_rte))
        #print and save:
        msg=f'{datasetname}-{self.cfg.descriptor}-{self.cfg.extractor}-{self.cfg.matcher}-{self.cfg.estimator}-{self.max_iter}iterations\n'
        msg+=f'correct ratio avg {np.mean(all_pair_fmrs):.5f}\n' \
            f'Mean_Registration_Recall {Mean_Registration_Recall}\n' \
            f'RRE {rre}\n' \
            f'RTE {rte}\n'

        with open('data/results.log','a') as f:
            f.write(msg)
        print(msg)

name2evaluator={
    'evaluator':evaluator,
}