
import os
import sys
from tokenize import group
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils.generate_vrptw import create_VRPTW_dataset,get_episode_data_fn,augment_vrp_data
from ModelCode.VRPTWModel_GEAQ import VRPTWModel
from ModelCode.VRPTWEnv import group_state,group_env
from tqdm import tqdm
import math 
import torch
import numpy as np
from torch.optim import Adam as optimizer
from utils.Logs import MyLogs


model_params = {
    "epochs":100,
    "batch_size":25,
    "instance_size":10000,
    "problem_size":100,
    "AUG_S":8,
    "group_s":30,
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
   
    "instance_dir":"instances/vrp100_test_seed1234.pkl",
    "vrptw_file_dir":"Vrp-Set-Solomon",
    "max_iter":10,
    "servicetime":None,
    "filepath":"Logs/vrptw_gen5_100.txt"
}



# dataset_couple = create_VRPTW_dataset(100,100)

vrptwmodel_reward = VRPTWModel(**model_params).to(device="cuda:0")
#state_high = torch.load(model_params['load_model_dir'])
#vrptwmodel_reward.load_state_dict(state_high['model'])



# TINY = model_params['TINY']
# 模型 的训练 学习率 1e-4    
myoptimizer = optimizer(vrptwmodel_reward.parameters(),**model_params['optimizer'])
mylogs = MyLogs(model_params['filepath'])
for epoch in tqdm(range(model_params['epochs'])):
    # batch_data = generate_vrp_data(batch_size=model_params['instance_size'] ,problem_size= model_params['problem_size'])
    dataset_couple = create_VRPTW_dataset(batch_size=model_params['instance_size'],problem_size=model_params['problem_size'])
    data_size = dataset_couple[0][0].shape[0]  
   
    for episode in range(math.ceil(data_size/model_params['batch_size'])):
        episode_data = get_episode_data_fn(dataset_couple,episode*model_params['batch_size'],model_params['batch_size'])
        problem_size = episode_data[0][0].shape[1]
        model_params['problem_size'] = problem_size 
        episode_data = augment_vrp_data(episode_data,problem_size,model_params['AUG_S'])

        batch_r  =  model_params['batch_size']
        batch_s  =  batch_r * model_params['AUG_S']
        group_s =  model_params["group_s"]
        AUG_S = model_params['AUG_S']

        incumbent_solutions = torch.zeros(batch_r, problem_size * 2, dtype=torch.int)
        
        env = group_env(episode_data,problem_size,model_params['round_distance'],model_params['depo_capacity'],servicetime=0) 
          
        

        for iter_ in range(model_params['max_iter']):
            step = 0
            group_state,reward,done = env.reset(group_size=group_s)
        
            first_action = torch.cuda.LongTensor(np.zeros((batch_s,group_s)))
            incumbent_solutions_expanded = incumbent_solutions.repeat(AUG_S, 1)

            group_state,reward,done = env.step(first_action)
           
            step += 1
            second_action = torch.cuda.LongTensor(np.arange(group_s) % problem_size)[None, :].expand(batch_s, group_s).clone()
            

            if iter_ > 0:
                second_action[:, -1] = incumbent_solutions_expanded[:, step]
            group_state, reward, done = env.step(second_action)
            step += 1
            # solutions.append(second_action.unsqueeze(2))

            prob_list = torch.zeros((batch_s,group_s,0)).to(device="cuda:0")
        
            vrptwmodel_reward.reset(group_state)
            vrptwmodel_reward.set_v_matrix(problem_size)
            # vrptwmodel_timepenalty.reset(group_state)
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

                action_probs = decoder_probs[group_state.batch_idx_mat,group_state.ant_idx_mat,group_state.current_node].reshape(batch_s,group_s)
                # 结束的不再考虑概率
                action_probs[group_state.finished]=1
                prob_list = torch.cat((prob_list,action_probs[:,:,None]),dim=2)
                step += 1


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

            group_reward = reward[:, :group_s - 1]
            group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)
            if model_params['use_log']==False:
                # log_probs = (prob_list+TINY).log().sum(dim=2) 
                log_probs = (prob_list).log().sum(dim=2) 
            group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

            group_loss = -group_advantage * log_probs[:, :group_s - 1]

            loss_1 = group_loss.mean()  # Reinforcement learning loss
            loss_2 = -prob_list[:, group_s - 1].mean()  # Imitation learning loss
            loss = loss_1 + loss_2 * 0.013


            vrptwmodel_reward.zero_grad()
       
            loss.backward()
        
            myoptimizer.step()
            if episode%5==0 and iter_%9==0:
                str_log = " epoch = {} , episode = {} , loss_mean = {},probelm_size={} ,reward_mean = {}".format(
                    epoch,episode,loss.item(),problem_size,reward.mean().item()
                )
                #print(str_log)
                mylogs.log(str_log)
            debug = 0
    debug = 0
    state={
        "epoch":epoch,
        "model":vrptwmodel_reward.state_dict(),
        "optimizer":myoptimizer.state_dict()
    }
    save_root = "ModelSave/vrptw_model_100.pt"
    torch.save(state,save_root)
    pass
a = 0 