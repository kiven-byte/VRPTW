
from asyncore import read
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
    "epochs":5,
    "batch_size":25,
    "instance_size":10000,
    "problem_size":125,
    #"problem_size":20,
    "AUG_S":8,
    # "group_s":2,
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
    # "load_model_dir":"ModelSave/vrptw_reward_100.pt",
    "load_model_dir":"ModelSave/vrptw_use_model.pt",
    "instance_dir":"instances/vrp100_test_seed1234.pkl"
}



# dataset_couple = create_VRPTW_dataset(100,100)
# vrptwmodel_timewait = VRPTWModel(**model_params).to(device="cuda:0")
vrptwmodel_reward = VRPTWModel(**model_params).to(device="cuda:0")

# 判断 是否 继续 训练

state_high = torch.load(model_params['load_model_dir'])
vrptwmodel_reward.load_state_dict(state_high['model'])

# 模型 的训练 学习率 1e-4    不用 学习率衰减 Scheduler 
#myoptimizer = optimizer(vrptwmodel_reward.parameters(),**model_params['optimizer'])

result_dict = {}

for my_idx in range(0,1):
    model_params['problem_size']=100
    # if my_idx==0:
    #     model_params['problem_size']=125
    #     pass
    # elif my_idx==1:
    #     model_params['problem_size']=150
    #     pass
    # elif my_idx==2: 
    #     model_params['problem_size']=175
    # elif my_idx==3: 
    #     model_params['problem_size']=200
    #     pass


    for epoch in tqdm(range(model_params['epochs'])):
       
        # batch_data = generate_vrp_data(batch_size=model_params['instance_size'] ,problem_size= model_params['problem_size'])
        #dataset_couple = create_VRPTW_dataset(batch_size=model_params['instance_size'],problem_size=model_params['problem_size'],
        #seed=1234
        #)

        dataset_couple = create_VRPTW_dataset(batch_size=model_params['instance_size'],problem_size=model_params['problem_size'],
        seed=1234)
        data_size = dataset_couple[0][0].shape[0]  
        all_route=[]

        reward_list = []
        
        for episode in range(math.ceil(data_size/model_params['batch_size'])):
            episode_data = get_episode_data_fn(dataset_couple,episode*model_params['batch_size'],model_params['batch_size'])
            problem_size = episode_data[0][0].shape[1]
            model_params['problem_size'] = problem_size 
            episode_data = augment_vrp_data(episode_data,problem_size,model_params['AUG_S'])

            batch_r  =  model_params['batch_size']
            batch_s  =  batch_r * model_params['AUG_S']
            group_s =  model_params["group_s"]
            AUG_S = model_params['AUG_S']

            #env = group_env(episode_data,problem_size,model_params['round_distance'],model_params['depo_capacity']) 
            incumbent_solutions = torch.zeros(batch_r, problem_size * 2, dtype=torch.int)
            curr_best=None
            
            with torch.no_grad():
                
                prob_matrix = torch.ones((batch_r, problem_size * problem_size), device="cuda")  # Matrix Q
                incumbent_edges = [[]] * batch_r
                incumbent_edges_probs = [[]] * batch_r
                max_reward = torch.full((batch_r,), -np.inf, device="cuda")
                incumbent_solutions = torch.zeros(batch_r, problem_size * 2, dtype=torch.long)
                # import time
                # t_start = time.time()
                max_iter = 10
                env = group_env(episode_data,problem_size,model_params['round_distance'],model_params['depo_capacity']) 

                for iter in range(max_iter): 
                    group_state,reward,done = env.reset(group_size=group_s)

                    incumbent_solutions_expanded = incumbent_solutions.repeat(AUG_S, 1)
                    first_action = torch.cuda.LongTensor(np.zeros((batch_s,group_s)))
                    last_action = first_action
                    group_state,reward,done = env.step(first_action)

                    prob_matrix_expanded = prob_matrix.repeat(AUG_S, 1).unsqueeze(1).expand(batch_s, group_s, -1).reshape(
                        batch_s, group_s,
                        problem_size,
                        problem_size)  # Expand Q because the same Q matrix is used for all 8 augmentations of an instance

                    solutions = [first_action.unsqueeze(2)]
                    solutions_iter_edges_prob = []

                    vrptwmodel_reward.reset(group_state)
                    vrptwmodel_reward.set_v_matrix(problem_size)
                    param_alpha =0.539
                    param_sigma = 9.55
                    while not done:
                        # _ , last_probs = vrptwmodel_timewait(group_state)
                        decoder_probs , _ = vrptwmodel_reward(group_state)
                        #decoder_probs , _ = vrptwmodel_timepenalty(group_state)
                        idx = last_action.unsqueeze(2).unsqueeze(3).expand(batch_s, group_s, 1, problem_size)
                        matrix_probs = torch.gather(prob_matrix_expanded, 2, idx).squeeze()
                        action_probs = decoder_probs ** param_alpha * matrix_probs

                        batch_s = decoder_probs.shape[0]
                        ant_size = decoder_probs.shape[1]
                        if model_params["use_log"] == True:
                            # sampling
                            #torch.multinomial(logits.exp(), self.n_samples)
                            actions = action_probs.reshape(batch_s*ant_size,-1).multinomial(1).squeeze(1).reshape(batch_s,ant_size)
                            # greedy
                            #torch.topk(logits, self.n_samples, dim = 1)[1]
                        else:
                            if model_params['method']=="softmax":
                                actions = torch.multinomial(action_probs.reshape(batch_s*ant_size,-1), 1).reshape(batch_s,ant_size)
                                #actions = decoder_probs.reshape(batch_s*ant_size,-1).multinomial(1).squeeze(1).reshape(batch_s,ant_size)
                                # actions[group_state.finished]=0
                            else:
                                # actions = decoder_probs.reshape(batch_s*ant_size,-1).multinomial(1).squeeze(1).reshape(batch_s,ant_size)
                                actions = torch.topk(action_probs.reshape(batch_s*ant_size,-1), 1, dim = 1)[1].reshape(batch_s,ant_size)
                                acb = 0
                        group_state,reward,done = env.step(actions)
                        solutions.append(actions.unsqueeze(2))
                        solutions_iter_edges_prob.append(torch.gather(decoder_probs, 2, actions.unsqueeze(2)))

                        last_action = actions
                    
                    group_reward = reward.reshape(AUG_S, int(batch_s / AUG_S), group_s)
                    solutions = torch.cat(solutions, dim=2)
                    solutions_iter_edges_prob = torch.cat(solutions_iter_edges_prob, dim=2)

                    max_reward_iter, _ = group_reward.max(dim=2)
                    max_reward_iter, max_reward_idx = max_reward_iter.max(dim=0)
                    improved_idx = max_reward < max_reward_iter
                    # print("iter = ",iter)
                    # print("max_reward_iter = ",max_reward_iter.max()) # 15.5176
                    # print("max_reward_iter_mean = ",max_reward_iter.mean())
                    # print("max_reward_mean = ",max_reward.mean())
                    # print("improved_idx = ",improved_idx)
                    if improved_idx.any():
                        # Update incumbent rewards of the search
                        max_reward[improved_idx] = max_reward_iter[improved_idx] 
                        # 实际上是修改 max_reward中  improved_idx 中为 True 的值
                        # Find the best solutions per instance (over all augmentations)
                        reward_g = group_reward.permute(1, 0, 2).reshape(batch_r, -1)[improved_idx]
                        iter_max_k, iter_best_k = torch.topk(reward_g, k=1, dim=1) # 找出每一组最好的 共25组 最好的解以及所在的索引值
                        solutions = solutions.reshape(AUG_S, batch_r, group_s, -1)
                        allsolutions = solutions.clone()
                        solutions = solutions.permute(1, 0, 2, 3).reshape(batch_r, AUG_S * group_s, -1)[[improved_idx]] # 试了一下似乎使用 [improved_idx ]  或者是 [[ improved_idx ]] 结果是一样的 , 超过 [[[  则会报错  ]]]
                        best_solutions_iter = torch.gather(solutions, 1, iter_best_k.unsqueeze(2).expand(-1, -1,
                                                                                                        solutions.shape[
                                                                                                            2])).squeeze(
                            1)
                        


                        
                        all_reward_g = group_reward.permute(1, 0, 2).reshape(batch_r, -1)
                        
                        all_iter_max_k, all_iter_best_k = torch.topk(all_reward_g, k=1, dim=1) # 找出每一组最好的 共25组 最好的解以及所在的索引值
                        

                        allsolutions = allsolutions.permute(1, 0, 2, 3).reshape(batch_r, AUG_S * group_s, -1) # 试了一下似乎使用 [improved_idx ]  或者是 [[ improved_idx ]] 结果是一样的 , 超过 [[[  则会报错  ]]]
                            
                        all_best_solutions_iter = torch.gather(allsolutions, 1, all_iter_best_k.unsqueeze(2).expand(-1, -1,
                                                                                                        allsolutions.shape[
                                                                                                            2])).squeeze(
                            1)




                        # Update incumbent solution storage
                        incumbent_solutions[improved_idx, :best_solutions_iter.shape[1]] = best_solutions_iter.cpu()
                        # incumbent_solutions 本身的shape 就是 [ 25 , 200 ]  其中 25 是 imporved_idx 为 True 的数量
                        # Find the edge ids that are part of best solutions of this iteration
                        best_solutions_iter = best_solutions_iter.unsqueeze(1).unsqueeze(3) # [ 25 , 1 , 100 , 1 ]
                        best_solutions_iter_edges = torch.cat(
                            [best_solutions_iter[:, :, :-1], best_solutions_iter[:, :, 1:]], dim=3)
                        # 尝试 roll 来 看看 能不能解出来 
                        # best_solutions_iter.roll(dims=2,shifts=-1)
                        best_solutions_iter_edges = best_solutions_iter_edges[:, :, :,
                                                    0] * problem_size + best_solutions_iter_edges[:, :, :, 1]

                        # Find the probability assigned to each edge of the incumbent solutions by the network
                        solutions_iter_edges_prob = solutions_iter_edges_prob.reshape(AUG_S, round(batch_s / AUG_S),
                                                                                    group_s, -1)
                        solutions_iter_edges_prob = \
                            solutions_iter_edges_prob.permute(1, 0, 2, 3).reshape(batch_r, AUG_S * group_s, -1)[
                                [improved_idx]]
                        best_solutions_iter_edge_prob = torch.gather(solutions_iter_edges_prob, 1,
                                                                    iter_best_k.unsqueeze(2).expand(-1, -1,
                                                                                                    solutions_iter_edges_prob.shape[
                                                                                                        2]))
                        # Update the incumbent edges and their probability
                        for j, idx in enumerate(improved_idx.nonzero()):
                            incumbent_edges[idx] = best_solutions_iter_edges[j, 0]
                            incumbent_edges_probs[idx] = best_solutions_iter_edge_prob.squeeze(1)[j]

                        # Update the matrix Q based on the incumbent edges and their probability
                        prob_matrix *= 0
                        for i in range(batch_r):
                            prob_matrix[i, incumbent_edges[i]] = (
                                    param_sigma / (incumbent_edges_probs[i] ** param_alpha))
                        prob_matrix = torch.clamp(prob_matrix, 1)


                        if curr_best==None:
                            curr_best_list = max_reward_iter
                            curr_best =abs(max_reward_iter.mean().item())
                        
                            #curr_best_solution = best_solutions_iter[iter_max_k.max(dim=0)[1]].squeeze().cpu().numpy()
                            
                            #str_solution = ""
                            #for cbs in curr_best_solution:
                                
                            #    str_solution +=str(cbs)+","
                            #str_solution=str_solution[0:len(str_solution)-1]
                    
                        else:
                            if abs(curr_best_list.mean())>=abs(max_reward_iter.mean().item()):
                                curr_best=abs(max_reward_iter.mean().item())
                                curr_best_list = max_reward_iter
                                

                reward_list.extend(list(curr_best_list.cpu().numpy()))
                #a= 0

        result_dict[epoch] = np.mean(reward_list)
        
        debug2 = 0
        #state={
        #    "epoch":epoch,
        #    "model":vrptwmodel_timepenalty.state_dict(),
        #    "optimizer":myoptimizer.state_dict()
        #}

    value_list  = [result_dict[mykey] for mykey in result_dict.keys()]
    import pandas as pd 

    df=pd.DataFrame()
    df.index = [i for i in range(0,5)]
    df['tab'] = value_list
    
    