
from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
import os
import random
import pickle
import copy
from src.utils.get_action_space import *
import gym
from gym import spaces

action_space, action_bound, action_bottom =None, None, None

m_type_dicts = {1:"CONV", 2:"DSCONV"}
class MaestroEnvironment(gym.Env):


    def __init__(self, model_defs, finish_reward=100, dim_size=6,n_action_steps=2, resource_size=2, dataflow="dla", is_discrete=True, true_continuous=False, chkpt_file="./chkpt.plt"):
        super(MaestroEnvironment,self).__init__()
        dst_path = "../../cost_model/maestro"

        maestro = dst_path
        self._executable = "{}".format(maestro)
        random.seed()
        random_file_name = random.randint(0, 2 ** 31)
        self.random_file_name = "{}".format(random_file_name)


        self.chkpt_file = chkpt_file
        self.is_gemm = False
        global action_space, action_bound, action_bottom
        action_space, action_bound, action_bottom = get_action_space()
        self.action_bottom = action_bound
        self.state = np.array([0.5]*8)
        self.last_runtime = 2 ** 64
        self.last_energy = 2**64
        self.last_throughput = 1
        self.observation = [0,0, 0,0,0,0]
        self.resource_state = [0, 0]
        self.consecutive_fail = 0
        self.max_fail = 0
        self.total_resources = action_bound
        self.action_bound = action_bound[:n_action_steps]
        self.total_step = len(model_defs)
        self.model_defs = model_defs
        self.model_defs_saved = copy.deepcopy(model_defs)
        model_bound = np.max(model_defs, axis=0, keepdims=True)
        self.model_defs_norm = model_defs/model_bound
        self.model_defs_norm_saved = copy.deepcopy(self.model_defs_norm)
        self.best_reward_whole = float("-inf")
        self.best_rewards = []
        self.best_sol = None
        self.reward = 0
        self.mae_reward = 0
        self.finish_reward = finish_reward
        self.mae_reward_decay = 0.
        self.worst_reward = None
        self.best_reward = None
        self.reward_scale = 0
        self.n_action_steps = n_action_steps
        self.emptyinfo =  [-1] * (len(self.observation) + len(self.resource_state))
        self.resource_size = resource_size
        self.reward_whole_eps = 0
        self.dim_size = dim_size
        self.reward_rec = []
        self.min_reward = None
        self.running_ave_reward = None
        self.worst_reward_list = [None for _ in range(self.total_step)]
        self.sig = 1
        self.mac_rec = []


        self.epoch = 0
        self.sol_record = []
        self.sol_reward_record = []
        self.dataflow = dataflow
        self.constraint_value = 2**63
        self.constraint = "area"
        self.prev_reward_whole_eps = 0
        self.exp_table = {}
        self.draw = np.arange(0,self.total_step )
        self.is_discrete = is_discrete
        self.true_contiunous = true_continuous and not is_discrete
        self.state_size = len(self.model_defs_norm[0]) + 3
        if self.is_discrete:
            self.N_DISCRETE_ACTIONS = len(action_space[0])
            self.action_space = spaces.MultiDiscrete([self.N_DISCRETE_ACTIONS, self.N_DISCRETE_ACTIONS])

        else:
            self.action_space = spaces.Box(low=0, high=1,
                                           shape=(self.n_action_steps,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.state_size,), dtype=np.float32)
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))
        self.resources_denorm = np.zeros((self.resource_size,))
    @property
    def get_state(self):
        """

        """
        return self.state

    def shuffle_model(self):
        draw = np.random.permutation(self.total_step)
        self.model_defs = self.model_defs_saved[draw]
        self.model_defs_norm = self.model_defs_norm_saved[draw]
        self.draw = draw
    def set_fitness(self, fitness="latency"):
        self.fitness = fitness

    def set_constraint_value(self, max_constraint, min_constraint, constraint_value):
        self.constraint_value = constraint_value
        self.constraint_info = {"value": constraint_value,
                                "max": max_constraint,
                                "min": min_constraint}
    def set_constraint(self, constraint="area"):
        self.constraint = constraint
    def reset(self):
        """

        """
        self.update_best_sol = False
        self.reward_whole_eps = 0
        self.mac_rec = []
        self.sig = 1
        self.fitness_record = []
        self.reward = 0
        self.sol = []
        self.mode = 0
        self.actions_step = 0
        self.total_used_constraints = 0
        dimensions = self.model_defs_norm[self.mode]
        self.action_idx = [0,0]
        self.action = np.array([action_space[idx][val] for idx, val in enumerate(self.action_idx)])
        self.state = np.concatenate((dimensions, np.array([  0,0, 0], dtype=np.float)))
        self.total_eps_rewards = 0
        infos = {}
        return self.state

    def norm_mac(self):
        mac_rec = np.array(self.mac_rec)
        impt =  mac_rec/ np.sum(mac_rec)
        return impt


    def update_reward_impt(self, done):
        # impt = np.ones(1, (self.mode) * self.n_action_steps + self.actions_step + 1)
        # if self.fitness == "thrpt_ave":
        #     impt_thrpt = self.norm_mac()
        #     impt[:len(impt_thrpt)] = impt_thrpt
        # return impt
        impt = None
        if self.fitness == "thrpt_ave":
            self.mac_rec.append(self.observation[-1])
            if done:
                impt = self.norm_mac()
        return impt

    def norm_state(self, T):
        T[:-1] = (T[:-1] - 0.5) * 2
        return T

    def update_mode_and_step(self):
        self.actions_step += 2
        if self.actions_step ==self.n_action_steps:
            self.mode+=1
            self.actions_step = 0

    def get_ref_constraint(self, bound=action_bound):
        sol = [bound[:self.n_action_steps] for i in range(len(self.model_defs))]
        _, total_constraint = self.exterior_search(sol)
        return total_constraint
    def update_best_reward_list(self, succeed):
        self.epoch += 1
        self.reward_whole_eps = self.prev_reward_whole_eps if not succeed else self.reward_whole_eps
        self.prev_reward_whole_eps = self.reward_whole_eps
        self.reward_rec.append(self.reward_whole_eps)
        self.best_rewards.append(self.best_reward_whole)



    def update_reward_whole(self):
        if self.reward_whole_eps > self.best_reward_whole:
            self.best_reward_whole = self.reward_whole_eps
            self.sol = self.retrive_sol_order(self.sol)
            self.update_best_sol =True
            print("Epoch {}: New best reward: {:9e}".format(self.epoch, self.best_reward_whole))
            self.best_sol = {"epoch": self.epoch,
                             "draw": self.draw,
                             "sol": self.sol,
                             "ctr": self.total_used_constraints}

    def retrive_sol_order(self, sol):
        ordered_sol = [None for _ in range(len(sol))]
        for i, d in enumerate(self.draw):
            ordered_sol[d] = copy.deepcopy(sol[i])
        return ordered_sol

    def get_reward(self, maestro_state):
        table_entry = tuple([self.mode] + self.action_idx)
        if table_entry in self.exp_table:
            reward, constraint = self.exp_table[table_entry]
        else:
            reward, constraint = self.oberserve_maestro(maestro_state)
            self.exp_table[table_entry] = (reward, constraint)
        if not reward:
            self.sig = -1
            return -1
        # self.worst_reward = min(reward, self.worst_reward) if self.worst_reward else reward
        if self.min_reward == None:
            self.min_reward = reward
        reward_saved = reward.copy()
        # self.reward_window.append(reward)
        self.min_reward = min(self.min_reward, reward_saved)
        reward -= self.min_reward

        # reward -= min(self.reward_window)
        self.total_eps_rewards += reward
        return reward, constraint, reward_saved



    def get_valid_action_range(self):
        valid_action_range = []
        for i, r in enumerate(self.left_resource):
            action_space_this = action_space[i]
            valid_action_range.append([1 if val==True else 0 for val in action_space_this<r])
        return np.vstack(valid_action_range).astype(float)


    def update_total_reward_constraint(self, constraint, reward_saved):
        self.total_used_constraints += constraint
        self.reward_whole_eps += reward_saved
    def is_cluster_step(self):
        if self.n_action_steps >2:
            if self.actions_step == self.n_action_steps -1:
                return True
            self.action[2] = min(self.action[2], self.action[0])
        return False

    def step(self, action):

        infos = {}
        infos["is_success"] = 0
        done = 0
        if self.is_discrete:
            self.action_idx[:] = action
        else:
            self.action_idx[:] =  np.clip(action * len(action_space[0]), a_min=0, a_max=11).astype(int)
        action_val = np.array([action_space[i][int(a)] for i, a in enumerate(self.action_idx)])
        self.action = action_val
        maestro_action =  self.action * self.action_bound
        if self.true_contiunous:
            self.action_idx[:] = np.clip(action *self.action_bound, a_min=action_bottom[:len(action)], a_max=action_bound[:len(action)]).astype(int)
            self.action = action
            maestro_action = self.action_idx
        maestro_state = np.concatenate((self.model_defs[self.mode], maestro_action)).copy()
        dimensions = self.model_defs_norm[self.mode]

        reward, constraint,reward_saved = self.get_reward(maestro_state)
        self.update_total_reward_constraint(constraint,reward_saved)
        self.sol.append((copy.deepcopy(self.action * self.action_bound)).clip(1))
        # self.action[:] = 0
        if self.total_used_constraints > self.constraint_value:
            # reward = (-self.total_eps_rewards + reward)
            reward = (-self.total_eps_rewards + reward) * (self.total_used_constraints - self.constraint_value)/self.constraint_value

            # reward =  -(self.total_used_constraints - self.constraint_value)/self.constraint_value
            done = 1
        if reward == -1:
            done = 1
        # self.state = self.norm_state(np.concatenate(
        #     (dimensions, np.array([ *[a/self.action_size for a in self.action_idx], self.actions_step], dtype=np.float))))
        self.state = self.norm_state(np.concatenate(
            (dimensions,
             np.array([*self.action, self.actions_step], dtype=np.float))))
        self.update_mode_and_step()

        if self.mode == self.total_step and not done:
            print("Done! Reward: {}".format(self.reward_whole_eps))
            infos["is_success"] = 1
            done = 1
            self.update_reward_whole()
            self.sol_record.append(self.sol)
            self.sol_reward_record.append(self.reward_whole_eps)
        done = bool(done == 1)
        infos["eps_rewards"] = self.reward_whole_eps
        infos["Best_eps_rewards"] = self.best_reward_whole
        infos["chkpt"] = self.get_chkpt()
        infos["reward_whole_eps"] = self.reward_whole_eps
        self.update_best_reward_list(infos["is_success"]) if done else None
        impt = self.update_reward_impt(done)
        if done:
            self.save_chkpt(self.chkpt_file)
        return self.state, reward, done, infos



    def judge(self):
        runtime, throughput, energy, area, l1_size, l2_size, mac, power = self.observation
        values = []
        for term in [self.fitness, self.constraint]:
            if term =="energy":
                reward =-energy
            elif term == "thrpt_ave":
                reward = throughput
            elif term == "LEP":
                reward = -energy * runtime
            elif term == "LAP":
                reward = -area * runtime
            elif term == "EAP":
                reward = -area * energy
            elif term == "thrpt" or term=="thrpt_naive":
                reward=throughput
            elif term == "thrpt_btnk":
                reward = throughput
            elif term == "latency":
                reward=-runtime
            elif term =="area":
                reward = -area
            elif term == "l1_size":
                reward = - l1_size
            elif term == "l2_size":
                reward = -l2_size
            elif term == "power":
                reward = -power
            else:
                raise NameError('Undefined fitness type')
            values.append(reward)
        return values[0], abs(values[1])


    def check_constraint(self, actions):
        used = np.sum(actions, axis=0)
        if any(used > self.total_resources[len(used)]):
            return False
        return True
    def ransom_search(self, max_epoch=1000, chpt_file="trial.plt"):
        self.chkpt_file = chpt_file
        n_layer = len(self.model_defs)
        best_reward = 0
        best_sol = None
        best_reward_record = []

        for epoch in range(max_epoch):
            self.epoch = epoch
            guess_action = []
            for _ in range(n_layer):
                pe = 2**random.randint(PE_RANGE_LOG[0], PE_RANGE_LOG[1])
                bw = 2**random.randint(BW_RANGE_LOG[0], BW_RANGE_LOG[1])
                action = [pe, bw]
                guess_action.append(action)
            if not self.check_constraint(guess_action):
                reward = 0
            else:
                reward = self.exterior_search(guess_action)

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_sol = guess_action
            print("Epoch {}: reward: {}".format(self.epoch, self.best_reward))
            if self.epoch %100==0:
                self.save_chkpt()

            self.best_rewards.append( self.best_reward)
        return self.best_rewards, self.best_sol

    def dfs(self, left_layers, guess_action):
        if left_layers == 0:
            self.epoch +=1
            if self.check_constraint(guess_action):
                reward = self.exterior_search(guess_action)
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_sol = guess_action
                print("Epoch {}: reward: {}".format(self.epoch, self.best_reward))
            self.best_rewards.append( self.best_reward)

            if self.epoch %100==0:
                self.save_chkpt()
            return
        for pe in [2**i for i in range(PE_RANGE_LOG[0], PE_RANGE_LOG[1]+1)]:
            for bw in [2**i for i in range(BW_RANGE_LOG[0], BW_RANGE_LOG[1]+1)]:
                action=[pe, bw]
                guess_action.append(action)
                self.dfs(left_layers-1, guess_action)
                guess_action.pop()
    def exhasustive_search(self, chkpt_file="trial.pgt"):
        n_layer = len(self.model_defs)
        self.best_reward = 0
        best_sol = None
        best_reward_record = []
        guess_action = []
        self.best_rewards = []
        self.epoch = 0
        self.best_sol = None
        self.chkpt_file = chkpt_file
        self.dfs(n_layer, guess_action)
        return self.best_rewards, self.best_sol

    def load_chkpt(self, chkpt):
        self.reward_rec = chkpt["reward_rec"]
        self.best_reward = chkpt["best_reward"]
        self.best_rewards= chkpt["best_rewards"]
        self.best_sol= chkpt["best_sol"]
        self.worst_reward = chkpt["worst_reward"]

    def get_chkpt(self):
        return {
            "reward_rec": self.reward_rec,
            "best_rewards": self.best_rewards,
            "best_sol": self.best_sol,
            "update_best_sol": self.update_best_sol,
            "best_reward": self.best_reward,
            "worst_reward": self.worst_reward,
            "sols": self.sol_record,
            "sol_reward_record": self.sol_reward_record,
            "ctrs_info": self.constraint_info
        }
    def save_chkpt(self, chkpt_file=None):
        if chkpt_file is None:
            chkpt_file = self.chkpt_file
        chkpt = self.get_chkpt()
        with open(chkpt_file, "wb") as fd:
            pickle.dump(chkpt, fd)
        # print(self.sol)


    def exterior_search_special(self, actions, action_size=2):
        total_reward = None
        mac_rec_list = []
        latency_list = []
        total_constraint = 0
        for i in range(len(actions)):
            action = actions[i]
            maestro_state = np.concatenate((self.model_defs[i], action))
            reward, constraint = self.oberserve_maestro(maestro_state)
            if reward == None:
                return None
            else:
                mac_rec_list.append(self.observation[-1])
                latency_list.append(self.observation[0])
            total_constraint += constraint
        total_reward = sum(mac_rec_list)/sum(latency_list)
        return total_reward, total_constraint

    def exterior_search(self, actions, action_size=2):
        if self.fitness == "thrpt_ave" or self.fitness=="thrpt_naive":
            return self.exterior_search_special(actions, action_size)
        total_reward = 0
        total_constraint = 0
        min_reward = float("Inf")
        for i in range(len(actions)):
            action = actions[i]
            maestro_state = np.concatenate((self.model_defs[i], action))
            reward, constraint = self.oberserve_maestro(maestro_state)
            if reward == None:
                return None
                # return float("-inf")
            if self.fitness == "thrpt_btnk":
                min_reward = min(min_reward, reward)
                total_reward = min_reward
            else:
                total_reward += reward
            total_constraint += constraint
        return total_reward, total_constraint

    def write_maestro(self, dimension, dataflow="dla", KTileSz=1, CTileSz=1, ClusterSz=4, m_file=None, layer_id=0):
        if len(dimension) > 6:
            m_type = m_type_dicts[int(dimension[-1])]
        else:
            m_type = "CONV"
        with open("../../data/dataflow/{}.m".format(dataflow), "r") as fd:
            with open("../../data/dataflow/dpt.m", "r") as fdpt:
                with open("{}.m".format(m_file), "w") as fo:
                    fo.write("Constant KTileSz {};\n".format(KTileSz))
                    fo.write("Constant CTileSz {};\n".format(CTileSz))
                    fo.write("Constant ClusterSz {};\n".format(ClusterSz))
                    fo.write("Network {} {{\n".format(layer_id))
                    fo.write("Layer {} {{\n".format(m_type))
                    fo.write("Type: {}\n".format(m_type))
                    fo.write(
                        "Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(
                            *dimension))
                    if m_type == "CONV":
                        fd.seek(0)
                        fo.write(fd.read())
                    else:
                        fdpt.seek(0)
                        fo.write(fdpt.read())
                    fo.write("}\n")
                    fo.write("}")

    def oberserve_maestro(self, state, firsttime=False):
        m_file = self.random_file_name
        dimension = state[:self.dim_size]

        actions =  state[-self.n_action_steps:]
        if self.n_action_steps==2:
            num_pe, KTileSz = actions.astype(int).squeeze()
            ClusterSz=num_pe
            self.resource_state = [num_pe, KTileSz]
            if self.is_gemm:
                self.write_maestro_gemm(dimension, dataflow=self.dataflow, KTileSz=KTileSz, ClusterSz=ClusterSz, m_file=m_file)
            else:
                self.write_maestro(dimension, dataflow=self.dataflow, KTileSz=KTileSz, ClusterSz=ClusterSz, m_file=m_file)
        else:
            num_pe, KTileSz, ClusterSz = actions.astype(int).squeeze()
            ClusterSz = max(1, ClusterSz)
            self.resource_state = [num_pe, KTileSz]
            if self.is_gemm:
                self.write_maestro_gemm(dimension, dataflow=self.dataflow, KTileSz=KTileSz, ClusterSz=ClusterSz, m_file=m_file)
            else:
                self.write_maestro(dimension, dataflow=self.dataflow, KTileSz=KTileSz, ClusterSz=ClusterSz, m_file=m_file)





        # print(num_pe, bw, l1_size)
        os.remove("./{}.csv".format(m_file)) if os.path.exists("./{}.csv".format(m_file)) else None
        command = [self._executable,
                 "--Mapping_file={}.m".format(m_file),
                   "--full_buffer=false", "--noc_bw_cstr=81920000",
                   "--noc_hops=1", "--noc_hop_latency=1",
                   "--noc_mc_support=true", "--num_pes={}".format(num_pe),
                   "--num_simd_lanes=1", "--l1_size_cstr=819200000",
                   "--l2_size_cstr=819200000", "--print_res=false", "--print_res_csv_file=true", "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]


        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        process.wait()




        try:

            df = pd.read_csv("./{}.csv".format(m_file))
            layer_name = df[" Layer Number"]
            runtime = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            throughput = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
            energy = np.array(df[" Activity count-based Energy (nJ)"]).reshape(-1, 1)
            area = np.array(df[" Area"]).reshape(-1, 1)
            power = np.array(df[" Power"]).reshape(-1, 1)
            l1_size = np.array(df[" L1 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l2_size = np.array(df["  L2 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            mac = np.array(df[" Num MACs"]).reshape(-1, 1)
            os.remove("./{}.csv".format(m_file))  if os.path.exists("./{}.csv".format(m_file)) else None
            os.remove("./log.txt") if os.path.exists("./log.txt") else None
            self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power]]
            # penalty = 1 / num_pe + 1 / bw
            # penalty= 1/penalty
            return self.judge()
        except:
            print("+"*20)
            print(num_pe, KTileSz, ClusterSz)
            print("+" * 20)
            return None
        # penalty = max(1, (num_pe-500))

        # return self.judge(firsttime)

    def write_maestro_gemm(self, dimension, dataflow="dla", KTileSz=1, CTileSz=1, ClusterSz=4, m_file=None, layer_id=0):
        m_type = "CONV"
        SzM, SzN, SzK = dimension
        dimension = [SzN, SzK, SzM, 1, 1, 1]
        with open("{}_f.m".format(dataflow), "r") as fd:
            with open("dpt_f.m", "r") as fdpt:
                with open("{}.m".format(m_file), "w") as fo:
                    fo.write("Constant KTileSz {};\n".format(KTileSz))
                    fo.write("Constant CTileSz {};\n".format(CTileSz))
                    fo.write("Constant ClusterSz {};\n".format(ClusterSz))
                    fo.write("Network {} {{\n".format(layer_id))
                    fo.write("Layer {} {{\n".format("CONV"))
                    fo.write("Type: {}\n".format(m_type))
                    fo.write(
                        "Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(
                            *dimension))
                    if m_type == "CONV":
                        fd.seek(0)
                        fo.write(fd.read())
                    else:
                        fdpt.seek(0)
                        fo.write(fdpt.read())
                    fo.write("}\n")
                    fo.write("}")



