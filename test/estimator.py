import numpy as np
from tqdm import tqdm
from utils.r_eval import compute_R_diff
from utils.utils import transform_points,make_non_exists_dir,GetRotationAngles
from functools import partial
from multiprocessing import Pool

def R_pre_log(dataset,save_dir):
    writer=open(f'{save_dir}/pre.log','w')
    pair_num=int(len(dataset.pc_ids))
    for pair in dataset.pair_ids:
        pc0,pc1=pair
        ransac_result=np.load(f'{save_dir}/{pc0}-{pc1}.npz',allow_pickle=True)
        transform_pr=ransac_result['trans']
        writer.write(f'{int(pc0)}\t{int(pc1)}\t{pair_num}\n')
        writer.write(f'{transform_pr[0][0]}\t{transform_pr[0][1]}\t{transform_pr[0][2]}\t{transform_pr[0][3]}\n')
        writer.write(f'{transform_pr[1][0]}\t{transform_pr[1][1]}\t{transform_pr[1][2]}\t{transform_pr[1][3]}\n')
        writer.write(f'{transform_pr[2][0]}\t{transform_pr[2][1]}\t{transform_pr[2][2]}\t{transform_pr[2][3]}\n')
        writer.write(f'{0.0}\t{0.0}\t{0.0}\t{1.0}\n')
    writer.close()


class refiner:
    def __init__(self):
        pass

    def center_cal(self,key_m0,key_m1,scores):
        key_m0=np.mean(key_m0,axis=0)
        key_m1=np.mean(key_m1,axis=0)
        return key_m0,key_m1

    def SVDR_w(self,beforerot,afterrot,scores):# beforerot afterrot Scene2,Scene1
        weight=np.diag(scores)
        H=np.matmul(np.matmul(np.transpose(afterrot),weight),beforerot)
        U,Sigma,VT=np.linalg.svd(H)
        return np.matmul(U,VT)

    def R_cal(self,key_m0,key_m1,center0,center1,scores):
        key_m0=key_m0-center0[None,:]
        key_m1=key_m1-center1[None,:]
        return self.SVDR_w(key_m1,key_m0,scores)

    def t_cal(self,key_m0,key_m1,scores,R):
        scores = scores / np.sum(scores)
        t_matches = key_m0 - key_m1@R.T
        t = np.sum(scores[:,None]*t_matches, axis=0)
        return t

    def Rt_cal(self,key_m0,key_m1,scores):
        center0,center1=self.center_cal(key_m0,key_m1,scores)
        R=self.R_cal(key_m0,key_m1,center0,center1,scores)
        t=self.t_cal(key_m0,key_m1,scores,R)
        return R,t
    
    def Refine_trans(self,key_m0,key_m1,T,scores,inlinerdist=None):
        key_m1_t=transform_points(key_m1,T)
        diff=np.sum(np.square(key_m0-key_m1_t),axis=-1)
        diff_w=diff*scores
        overlap=np.where(diff_w<inlinerdist*inlinerdist)[0]
            
        scores=scores[overlap]
        key_m0=key_m0[overlap]
        key_m1=key_m1[overlap]
        R,t=self.Rt_cal(key_m0, key_m1, scores)
        Tnew=np.eye(4)
        Tnew[0:3,0:3]=R
        Tnew[0:3,3]=t
        return Tnew


class estimator:
    def __init__(self,cfg):
        self.cfg=cfg
        self.inliner_dist=cfg.ransac_c_inlinerdist
        self.refiner = refiner()


    def DR_statictic(self,DR_indexs):
        R_index_pre_statistic={}
        for i in range(8):
            R_index_pre_statistic[i]=[]
        for t in range(DR_indexs.shape[0]):
            R_index_pre_statistic[DR_indexs[t]].append(t)
        R_index_pre_probability=[]
        for i in range(8):
            if len(R_index_pre_statistic[i])<2:
                R_index_pre_probability.append(0)
            else:
                num=float(len(R_index_pre_statistic[i]))/100.0
                R_index_pre_probability.append(num*(num-0.01)*(num-0.02))
        R_index_pre_probability=np.array(R_index_pre_probability)
        if np.sum(R_index_pre_probability)<1e-4:
            return None,None
        R_index_pre_probability=R_index_pre_probability/np.sum(R_index_pre_probability)
        return R_index_pre_statistic,R_index_pre_probability

    def Threepps2Tran_weight(self,kps0_init,kps1_init,weight):
        weight_c = weight / np.sum(weight)
        center0=np.mean(kps0_init,0,keepdims=True)
        center1=np.mean(kps1_init,0,keepdims=True)
        kps0 = kps0_init-center0
        kps1 = kps1_init-center1
        m = (kps1 * weight[:,None]).T @ kps0
        U,S,VT = np.linalg.svd(m)
        rotation = VT.T @ U.T   #predicted RT
        t_matches = kps0_init - kps1_init@rotation.T #(3,3)
        offset = np.sum(weight_c[:,None]*t_matches, axis=0) #(3,)
        transform = np.eye(4)
        transform[0:3,0:3] = rotation
        transform[0:3,3] = offset
        return transform #3*4

    def Threepps2Tran(self,kps0_init,kps1_init):
        center0=np.mean(kps0_init,0,keepdims=True)
        center1=np.mean(kps1_init,0,keepdims=True)
        m = (kps1_init-center1).T @ (kps0_init-center0)
        U,S,VT = np.linalg.svd(m)
        rotation = VT.T @ U.T   #predicted RT
        offset = center0 - (center1 @ rotation.T)
        transform=np.concatenate([rotation,offset.T],1)
        return transform #3*4

    def overlap_cal_weight(self,key_m0,key_m1,T,weight):
        key_m1=transform_points(key_m1,T)
        diff=np.sum(np.square(key_m0-key_m1),axis=-1)
        diff_w = diff*weight
        overlap=np.mean(diff_w<self.inliner_dist*self.inliner_dist)
        return overlap
    
    def overlap_cal(self,key_m0,key_m1,T):
        key_m1=transform_points(key_m1,T)
        diff=np.sum(np.square(key_m0-key_m1),axis=-1)
        overlap=np.mean(diff<self.inliner_dist*self.inliner_dist)
        return overlap

    def transdiff(self,gt,pre):
        Rdiff=compute_R_diff(gt[0:3:,0:3],pre[0:3:,0:3])
        tdiff=np.sqrt(np.sum(np.square(gt[0:3,3]-pre[0:3,3])))
        return Rdiff,tdiff


    def ransac(self,dataset,max_iter=1000):
        match_dir=f'{self.cfg.output_cache_fn}/test/{dataset.name}/Match'
        Index_dir=f'{match_dir}/DR_index'
        Save_dir=f'{match_dir}/MSReg/{max_iter}iters'
        make_non_exists_dir(Save_dir)

        datasetname=dataset.name
        Keys_dir=f'{self.cfg.origin_data_dir}/{datasetname}/Keypoints_PC'

        print(f'Ransac with MSReg on {dataset.name}:')
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            #if os.path.exists(f'{Save_dir}/{id0}-{id1}.npz'):continue
            gt=dataset.get_transform(id0,id1)
            #Keypoints
            Keys0=np.load(f'{Keys_dir}/cloud_bin_{id0}Keypoints.npy')
            Keys1=np.load(f'{Keys_dir}/cloud_bin_{id1}Keypoints.npy')
            #Key_pps
            pps=np.load(f'{match_dir}/{id0}-{id1}.npy')
            Keys_m0=Keys0[pps[:,0]]
            Keys_m1=Keys1[pps[:,1]]
            #Disparities
            if datasetname[0:2] == 'CS' or datasetname[0:4] == 'CTCS':
                disps1 = np.load(f'{self.cfg.origin_data_dir}/{datasetname}/Disparity/disp_{id1}.npy')
                disps_m1=disps1[pps[:,1]]
                disps_m1 = 1/disps_m1
                d_max = np.max(disps_m1)
                d_min = np.min(disps_m1)
                disps_m1 = (disps_m1-d_min)/(d_max-d_min)
            #Indexs
            Index=np.load(f'{Index_dir}/{id0}-{id1}.npy')
            #DR_statistic
            R_index_pre_statistic,R_index_pre_probability=self.DR_statictic(Index)
  
            if R_index_pre_probability is None:
                np.savez(f'{Save_dir}/{id0}-{id1}.npz',trans=np.eye(4), center=0,axis=0,recalltime=50001)
            else:
                #RANSAC
                iter_ransac=0
                recall_time=0
                best_overlap=0
                best_trans_ransac=np.eye(4)
                best_3p_in_0=np.ones([3,3])
                best_3p_in_1=np.ones([3,3])
                max_time=50000
                exec_time=0
                while iter_ransac<max_iter:
                    if exec_time>max_time:break
                    exec_time+=1
                    R_index=np.random.choice(range(8),p=R_index_pre_probability)
                    if (len(R_index_pre_statistic[R_index])<2):
                        continue
                    iter_ransac+=1
                    idxs_init=np.random.choice(np.array(R_index_pre_statistic[R_index]),3) #guarantee the same index
                    kps0_init=Keys_m0[idxs_init]
                    kps1_init=Keys_m1[idxs_init]
                    #if not ok:continue
                    if datasetname[0:2] == 'CS' or datasetname[0:4] == 'CTCS':
                        if self.cfg.weight == False:
                            trans=self.Threepps2Tran(kps0_init,kps1_init)
                            overlap=self.overlap_cal(Keys_m0,Keys_m1,trans)
                        else:
                            disps1_init=disps_m1[idxs_init]
                            weight_init = np.exp(-np.sqrt(disps1_init))
                            weight_m1 = np.exp(-np.sqrt(disps_m1))
                            trans=self.Threepps2Tran_weight(kps0_init,kps1_init,weight_init)
                            overlap=self.overlap_cal_weight(Keys_m0,Keys_m1,trans,weight_m1)
                    else:
                        trans=self.Threepps2Tran(kps0_init,kps1_init)
                        overlap=self.overlap_cal(Keys_m0,Keys_m1,trans)
                    R_x,R_y,R_z=GetRotationAngles(trans[0:3,0:3])
                    R_xy=abs(R_x)+abs(R_y)
                    if R_xy<=15 and overlap>best_overlap:
                        best_overlap=overlap
                        best_trans_ransac=trans
                        best_3p_in_0=kps0_init
                        best_3p_in_1=kps1_init
                        recall_time=iter_ransac
                # refine:
                refine_times = 10
                if datasetname[0:2] == 'CS' or datasetname[0:4] == 'CTCS':
                    if self.cfg.weight == False:
                        scores = np.ones([Keys_m0.shape[0]])
                    else:
                        scores = np.exp(-np.sqrt(disps_m1))
                else:
                    scores = np.ones([Keys_m0.shape[0]])
                for refine_i in range(refine_times):
                    thres = 0.5 + ((2-0.5)/refine_times)*(refine_times-refine_i)
                    best_trans_ransac=self.refiner.Refine_trans(Keys_m0,Keys_m1,best_trans_ransac,scores,inlinerdist=self.inliner_dist*thres)
                #save:
                np.savez(f'{Save_dir}/{id0}-{id1}.npz',trans=best_trans_ransac, center=np.concatenate([best_3p_in_0,best_3p_in_1],axis=0),recalltime=recall_time)
                
        R_pre_log(dataset,Save_dir)

    

class estimator_mul:
    def __init__(self,cfg):
        self.cfg=cfg
        self.inliner_dist=cfg.ransac_c_inlinerdist
        self.refiner = refiner()


    def DR_statictic(self,DR_indexs):
        R_index_pre_statistic={}
        for i in range(8):
            R_index_pre_statistic[i]=[]
        for t in range(DR_indexs.shape[0]):
            R_index_pre_statistic[DR_indexs[t]].append(t)
        R_index_pre_probability=[]
        for i in range(8):
            if len(R_index_pre_statistic[i])<2:
                R_index_pre_probability.append(0)
            else:
                num=float(len(R_index_pre_statistic[i]))/100.0
                R_index_pre_probability.append(num*(num-0.01)*(num-0.02))
        R_index_pre_probability=np.array(R_index_pre_probability)
        if np.sum(R_index_pre_probability)<1e-4:
            return None,None
        R_index_pre_probability=R_index_pre_probability/np.sum(R_index_pre_probability)
        return R_index_pre_statistic,R_index_pre_probability



    def Threepps2Tran(self,kps0_init,kps1_init):
        center0=np.mean(kps0_init,0,keepdims=True)
        center1=np.mean(kps1_init,0,keepdims=True)
        m = (kps1_init-center1).T @ (kps0_init-center0)
        U,S,VT = np.linalg.svd(m)
        rotation = VT.T @ U.T   #predicted RT
        offset = center0 - (center1 @ rotation.T)
        transform=np.concatenate([rotation,offset.T],1)
        return transform #3*4

    def Threepps2Tran_weight(self,kps0_init,kps1_init,weight):
        weight_c = weight / np.sum(weight)
        center0=np.mean(kps0_init,0,keepdims=True)
        center1=np.mean(kps1_init,0,keepdims=True)
        kps0 = kps0_init-center0
        kps1 = kps1_init-center1
        m = (kps1 * weight[:,None]).T @ kps0
        U,S,VT = np.linalg.svd(m)
        rotation = VT.T @ U.T   #predicted RT
        t_matches = kps0_init - kps1_init@rotation.T #(3,3)
        offset = np.sum(weight_c[:,None]*t_matches, axis=0) #(3,)
        transform = np.eye(4)
        transform[0:3,0:3] = rotation
        transform[0:3,3] = offset
        return transform #3*4

    def overlap_cal(self,key_m0,key_m1,T):
        key_m1=transform_points(key_m1,T)
        diff=np.sum(np.square(key_m0-key_m1),axis=-1)
        overlap=np.mean(diff<self.inliner_dist*self.inliner_dist)
        return overlap

    def overlap_cal_weight(self,key_m0,key_m1,T,weight):
        key_m1=transform_points(key_m1,T)
        diff=np.sum(np.square(key_m0-key_m1),axis=-1)
        diff_w = diff*weight
        overlap=np.mean(diff_w<self.inliner_dist*self.inliner_dist)
        return overlap

    def transdiff(self,gt,pre):
        Rdiff=compute_R_diff(gt[0:3:,0:3],pre[0:3:,0:3])
        tdiff=np.sqrt(np.sum(np.square(gt[0:3,3]-pre[0:3,3])))
        return Rdiff,tdiff


    def ransac_once(self,dataset,max_iter,pair):
        match_dir=f'{self.cfg.output_cache_fn}/test/{dataset.name}/Match'
        Index_dir=f'{match_dir}/DR_index'
        Save_dir=f'{match_dir}/MSReg/{max_iter}iters'
        datasetname=dataset.name

        id0,id1=pair
        #if os.path.exists(f'{Save_dir}/{id0}-{id1}.npz'):continue
        gt=dataset.get_transform(id0,id1)
        #Keypoints
        Keys0=dataset.get_kps(id0)
        Keys1=dataset.get_kps(id1)
        #Key_pps
        pps=np.load(f'{match_dir}/{id0}-{id1}.npy')
        Keys_m0=Keys0[pps[:,0]]
        Keys_m1=Keys1[pps[:,1]]
        if datasetname[0:2] == 'CS' or datasetname[0:4] == 'CTCS':
            disps1 = np.load(f'{self.cfg.origin_data_dir}/{datasetname}/Disparity/disp_{id1}.npy')
            disps_m1=disps1[pps[:,1]]
            disps_m1 = 1/disps_m1
            d_max = np.max(disps_m1)
            d_min = np.min(disps_m1)
            disps_m1 = (disps_m1-d_min)/(d_max-d_min)
        #Indexs
        Index=np.load(f'{Index_dir}/{id0}-{id1}.npy')
        #DR_statistic
        R_index_pre_statistic,R_index_pre_probability=self.DR_statictic(Index)
  
        if R_index_pre_probability is None:
            np.savez(f'{Save_dir}/{id0}-{id1}.npz',trans=np.eye(4), center=0,axis=0,recalltime=50001)
            
        else:
            #RANSAC
            iter_ransac=0
            recall_time=0
            best_overlap=0
            best_trans_ransac=np.eye(4)
            best_3p_in_0=np.ones([3,3])
            best_3p_in_1=np.ones([3,3])
            max_time=50000
            exec_time=0
            while iter_ransac<max_iter:
                if exec_time>max_time:break
                exec_time+=1
                R_index=np.random.choice(range(8),p=R_index_pre_probability)
                if (len(R_index_pre_statistic[R_index])<2):
                    continue
                iter_ransac+=1
                idxs_init=np.random.choice(np.array(R_index_pre_statistic[R_index]),3) #guarantee the same index
                kps0_init=Keys_m0[idxs_init]
                kps1_init=Keys_m1[idxs_init]
                if datasetname[0:2] == 'CS' or datasetname[0:4] == 'CTCS':
                    if self.cfg.weight == False:
                        trans=self.Threepps2Tran(kps0_init,kps1_init)
                        overlap=self.overlap_cal(Keys_m0,Keys_m1,trans)
                    else:
                        disps1_init=disps_m1[idxs_init]
                        weight_init = np.exp(-np.sqrt(disps1_init))
                        weight_m1 = np.exp(-np.sqrt(disps_m1))
                        trans=self.Threepps2Tran_weight(kps0_init,kps1_init,weight_init)
                        overlap=self.overlap_cal_weight(Keys_m0,Keys_m1,trans,weight_m1)
                else:
                    trans=self.Threepps2Tran(kps0_init,kps1_init)
                    overlap=self.overlap_cal(Keys_m0,Keys_m1,trans)
                R_x,R_y,R_z=GetRotationAngles(trans[0:3,0:3])
                R_xy=abs(R_x)+abs(R_y)
                if R_xy<=15 and overlap>best_overlap:
                    best_overlap=overlap
                    best_trans_ransac=trans
                    best_3p_in_0=kps0_init
                    best_3p_in_1=kps1_init
                    recall_time=iter_ransac
            # refine:
            refine_times = 10
            if datasetname[0:2] == 'CS' or datasetname[0:4] == 'CTCS':
                if self.cfg.weight == False:
                    scores = np.ones([Keys_m0.shape[0]])
                else:
                    scores = np.exp(-np.sqrt(disps_m1))
            else:
                scores = np.ones([Keys_m0.shape[0]])
            for refine_i in range(refine_times):
                thres = 0.5 + ((2-0.5)/refine_times)*(refine_times-refine_i)
                best_trans_ransac=self.refiner.Refine_trans(Keys_m0,Keys_m1,best_trans_ransac,scores,inlinerdist=self.inliner_dist*thres)
            np.savez(f'{Save_dir}/{id0}-{id1}.npz',trans=best_trans_ransac, center=np.concatenate([best_3p_in_0,best_3p_in_1],axis=0),recalltime=recall_time)


    def ransac(self,dataset,max_iter=1000):
        match_dir=f'{self.cfg.output_cache_fn}/test/{dataset.name}/Match'
        Index_dir=f'{match_dir}/DR_index'
        Save_dir=f'{match_dir}/MSReg/{max_iter}iters'
        make_non_exists_dir(Save_dir)
        print(f'Ransac with MSReg on {dataset.name}:')
        pair_ids=dataset.pair_ids
        pool = Pool(len(pair_ids))
        func = partial(self.ransac_once,dataset,max_iter)
        pool.map(func,pair_ids)
        pool.close()
        pool.join()
        R_pre_log(dataset,Save_dir)
        print('Done')


name2estimator={
    'estimator':estimator,
    'estimator_mul':estimator_mul
}
