
from asyncore import read
import os
import sys
from tokenize import group

from pyparsing import col
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils.generate_vrptw import create_VRPTW_dataset,get_episode_data_fn,augment_vrp_data,read_standard_dataset,getDirFile,getSol_value
from ModelCode.VRPTWModel_GEAQ import VRPTWModel
from ModelCode.VRPTWEnv import group_state,group_env
from tqdm import tqdm
import math 
import torch
import numpy as np
from torch.optim import Adam as optimizer
import pandas as pd 


model_params = {
    "epochs":1,
    "batch_size":25,
    #"batch_size":1,
    "instance_size":10000,
    "problem_size":100,
    #"problem_size":20,
    "AUG_S":8,
    #"group_s":2,
    "group_s":100,
    "round_distance":False,
    "depo_capacity":50,
    
    # 模型参数
    "encoder_layer_num":3,
    "embedding_dim":128,
    "head_num":8,
    "qkv_dim":16,
    "hidden_dim":512,
    "sqrt_embedding_dim":128**(0.5),
    "logit_clipping":10,
    "use_log":False ,
    "method":"softmax",
    "TINY":1e-15,
    "optimizer":{
            "lr":1e-4,
            "weight_decay":1e-8

        },
    "load_model_dir":"ModelSave/vrptw_use_model.pt",
   
    "instance_dir":"instances/vrp100_test_seed1234.pkl",
    "vrptw_file_dir":"Vrp-Set-Solomon",
    
    "max_iter":10,
    "servicetime":None
}



# dataset_couple = create_VRPTW_dataset(100,100)
# vrptwmodel_timewait = VRPTWModel(**model_params).to(device="cuda:0")
vrptwmodel_reward = VRPTWModel(**model_params).to(device="cuda:0")

# 判断 是否 继续 训练

state_high = torch.load(model_params['load_model_dir'])
vrptwmodel_reward.load_state_dict(state_high['model'])

# 模型 的训练 学习率 1e-4   
# myoptimizer = optimizer(vrptwmodel_reward.parameters(),**model_params['optimizer'])

filelist = getDirFile(model_params['vrptw_file_dir'])
file_dict={}
for curr_file in filelist:
    currfilepath = model_params['vrptw_file_dir'] +'/'+curr_file
    filename = curr_file.split(".")[0]
    filetype = curr_file.split(".")[1]
    if filename not in file_dict:
        file_dict[filename]={}
    print(curr_file)
    
    if filetype=='txt':
    # batch_data = generate_vrp_data(batch_size=model_params['instance_size'] ,problem_size= model_params['problem_size'])
    # dataset_couple = create_VRPTW_dataset(batch_size=model_params['instance_size'],problem_size=model_params['problem_size'])
        dataset_couple = read_standard_dataset(currfilepath,model_params['batch_size'])
        # dataset_couple = create_VRPTW_dataset(batch_size=model_params['instance_size'],problem_size=model_params['problem_size'],instance_dir=model_params['instance_dir'])
        data_size = dataset_couple[0][0].shape[0]  
        all_route=[]
        model_params['depo_capacity'] = dataset_couple[2]
        model_params['servicetime'] = dataset_couple[3]
        episode_data = (dataset_couple[0]),(dataset_couple[1])
            #episode_data = get_episode_data_fn(dataset_couple,episode*model_params['batch_size'],model_params['batch_size'])
        problem_size = episode_data[0][0].shape[1]
        model_params['problem_size'] = problem_size 
        episode_data = augment_vrp_data(episode_data,problem_size,model_params['AUG_S'])

        batch_r  =  model_params['batch_size']
        batch_s  =  batch_r * model_params['AUG_S']
        group_s =  model_params["group_s"]
        AUG_S = model_params['AUG_S']
        #problem_size = 
        env = group_env(episode_data,problem_size,model_params['round_distance'],model_params['depo_capacity'],model_params['servicetime']) 
        incumbent_solutions = torch.zeros(batch_r, problem_size * 2, dtype=torch.int)
        curr_best=None
        for iter_ in range(model_params['max_iter']):
            step = 0
            group_state,reward,done = env.reset(group_size=group_s)
            incumbent_solutions_expanded = incumbent_solutions.repeat(AUG_S, 1)
            first_action = torch.cuda.LongTensor(np.zeros((batch_s,group_s)))

            group_state,reward,done = env.step(first_action)

            step+=1
            second_action = torch.cuda.LongTensor(np.arange(group_s) % problem_size)[None, :].expand(batch_s, group_s).clone()
            
            if iter_ > 0:
                second_action[:, -1] = incumbent_solutions_expanded[:, step]
            group_state, reward, done = env.step(second_action)
            step += 1
          
            vrptwmodel_reward.reset(group_state)
            vrptwmodel_reward.set_v_matrix(problem_size)
            # prob_list = torch.zeros((batch_s,group_s,0)).to(device="cuda:0")
            
            while not done:
                # _ , last_probs = vrptwmodel_timewait(group_state)
                decoder_probs , _ = vrptwmodel_reward(group_state)
                #decoder_probs , _ = vrptwmodel_timepenalty(group_state)
                
                batch_s = decoder_probs.shape[0]
                ant_size = decoder_probs.shape[1]
                if model_params["use_log"] == True:
                    # sampling
                    #torch.multinomial(logits.exp(), self.n_samples)
                    actions = decoder_probs.reshape(batch_s*ant_size,-1).multinomial(1).squeeze(1).reshape(batch_s,ant_size)
                    # greedy
                    #torch.topk(logits, self.n_samples, dim = 1)[1]
                else:
                    if model_params['method']=="softmax":
                        actions = torch.multinomial(decoder_probs.reshape(batch_s*ant_size,-1), 1).reshape(batch_s,ant_size)
                        #actions = decoder_probs.reshape(batch_s*ant_size,-1).multinomial(1).squeeze(1).reshape(batch_s,ant_size)
                        # actions[group_state.finished]=0
                    else:
                        # actions = decoder_probs.reshape(batch_s*ant_size,-1).multinomial(1).squeeze(1).reshape(batch_s,ant_size)
                        actions = torch.topk(decoder_probs.reshape(batch_s*ant_size,-1), 1, dim = 1)[1].reshape(batch_s,ant_size)
                
                group_state,reward,done = env.step(actions)

      

            group_reward = reward.reshape(AUG_S, batch_r, group_s)
            max_reward, _ = group_reward.max(dim=2)
            max_reward, _ = max_reward.max(dim=0)

            reward_g = group_reward.permute(1, 0, 2).reshape(batch_r, -1)
            iter_max_k, iter_best_k = torch.topk(reward_g, k=1, dim=1)
            solutions = group_state.selected_node_list.reshape(AUG_S, batch_r, group_s, -1)
            solutions = solutions.permute(1, 0, 2, 3).reshape(batch_r, AUG_S * group_s, -1)
            best_solutions_iter = torch.gather(solutions, 1,
                                                iter_best_k.unsqueeze(2).expand(-1, -1, solutions.shape[2])).squeeze(
                1)
            incumbent_solutions[:, :best_solutions_iter.shape[1]] = best_solutions_iter

            data = dataset_couple[0][0][0]
            test_route = best_solutions_iter[iter_max_k.max(dim=0)[1]].squeeze(0)

            iter_max_k_best = iter_max_k.max().item()
            gathering_index = test_route[:,None].expand(-1,2)
            ordered_seq = data.gather(dim=0,index=gathering_index)
            rolled_seq = ordered_seq.roll(dims=0,shifts=-1)
            segment_lengths = ((ordered_seq-rolled_seq)**2)
            a_data = segment_lengths.sum(dim=1).sqrt() 
            travel_distances  = a_data.sum(dim=0)

            if curr_best==None:
                curr_best =iter_max_k_best
                curr_best_solution = best_solutions_iter[iter_max_k.max(dim=0)[1]].cpu().numpy()
                str_solution = ""
                for cbs in curr_best_solution[0]:
                    
                    str_solution +=str(cbs)+","
                str_solution=str_solution[0:len(str_solution)-1]
                
            else:
                if abs(curr_best)>abs(iter_max_k_best):
                    curr_best=iter_max_k_best
                    curr_best_solution = best_solutions_iter[iter_max_k.max(dim=0)[1]].cpu().numpy()
                    str_solution = ""
                    for cbs in curr_best_solution[0]:
                        
                        str_solution +=str(cbs)+","
                    str_solution=str_solution[0:len(str_solution)-1]
            #curr_best = abs(max_reward_iter.max().item())
            test = 0
        file_dict[filename]['model'] =abs(curr_best) 
        file_dict[filename]['route'] =str_solution
        
    elif filetype=='sol':

        best_solution = getSol_value(currfilepath)
        file_dict[filename]['sol'] =best_solution 

