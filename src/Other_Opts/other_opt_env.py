from subprocess import Popen, PIPE
import pandas as pd
import os,sys
import random
import pickle
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
from src.utils.get_action_space import *
import copy
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from scipy import optimize


action_space, action_bound, action_bottom = get_action_space()
action_space = [act * bound for act, bound in zip(action_space, action_bound)]
start_range, end_range = 0, len(action_space[0])
df_dict = {1:"dla", 2:"shi", 3:"eye"}
m_type_dicts = {1:"CONV", 2:"DSCONV"}


class MyBounds(object, ):
    def __init__(self,length):
        xmax = [11 for _ in range(length)]
        xmin = [0 for _ in range(length)]
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

def select_parents(pop, fitness, num_parents, num_layers):
    parents = np.empty((num_parents, num_layers, 2))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num] = pop[max_fitness_idx]
        fitness[max_fitness_idx] = float("-Inf")
    return parents

def crossover(parents, offspring_size, num_layers):
    offspring = np.empty((offspring_size, num_layers, 2))
    crossover_point = np.uint8(num_layers/2)

    for k in range(offspring_size):
        parent1_idx = k%parents.shape[0]
        parent2_idx = np.random.randint(0, parents.shape[0]) #(k+1)%parents.shape[0]
        offspring[k][0:crossover_point] = parents[parent1_idx][0:crossover_point]
        offspring[k][crossover_point:] = parents[parent2_idx][crossover_point:]
    return offspring

def mutation(offsprings, num_layers,rate=0.05):
    for idx in range(offsprings.shape[0]):
        for lay in range(offsprings.shape[1]):
            for p in range(offsprings.shape[2]):
                if random.random() < rate:
                    offsprings[idx][lay][p] = random.randint(0, 11)


    return offsprings

class MaestroEnvironment(object):


    def __init__(self, model_defs, finish_reward=100, dim_size=6,n_action_steps=2, resource_size=2, dataflow="dla", is_discrete=True):
        super(MaestroEnvironment,self).__init__()
        dst_path = "../../cost_model/maestro"

        maestro = dst_path
        self._executable = "{}".format(maestro)
        random.seed()
        random_file_name = random.randint(0, 2 ** 31)
        self.random_file_name = "{}".format(random_file_name)
        self.is_gemm = False

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
        self.count_invalid=0
        self.best_rewards_iteration = []
        self.epoch = 0
        self.sol_record = []
        self.exp_table = {}
        self.sol_reward_record = []
        self.dataflow = dataflow
        self.constraint_value = 2**63
        self.constraint = "area"
        self.prev_reward_whole_eps = 0
        self.exp_table = {}
        self.draw = np.arange(0,self.total_step )
        self.is_discrete = is_discrete
        self.state_size = len(self.model_defs_norm[0]) + 3
        self.update_best_sol = False


    def reset(self):

        self.update_best_sol = False
        self.mac_rec = []
        self.sig = 1
        self.reward_record = []
        self.reward = 0
        self.sol = []
        self.mode = 0
        self.actions_step = 0
        action_idx = [3, 3]
        self.action = np.array([action_space[idx][val] for idx, val in enumerate(action_idx)])
        self.left_resource = [1 for _ in range(self.resource_size)]
        dimensions = self.model_defs_norm[self.mode]
        self.state = np.zeros((self.state_size,), dtype=np.float32)
        self.total_eps_rewards = 0
        return self.state

    def get_ref_constraint(self, bound=action_bound):
        sol = [bound[:self.n_action_steps] for i in range(len(self.model_defs))]
        _, total_constraint = self.exterior_search(sol)
        return total_constraint
    def set_constraint_value(self, max_constraint, min_constraint, constraint_value):
        self.constraint_value = constraint_value
        self.constraint_info = {"value": constraint_value,
                                "max": max_constraint,
                                "min": min_constraint}
    def set_constraint(self, constraint="area"):
        self.constraint = constraint
    def set_fitness(self, fitness="energy"):
        self.fitness = fitness




    def resource_check(self):
        return not any(np.array(self.left_resource) < 0)

    def judge(self):
        runtime, throughput, energy, area, l1_size, l2_size, mac, power = self.observation
        values = []
        for term in [self.fitness, self.constraint]:
            if term == "energy":
                reward = -energy
            elif term == "thrpt_ave":
                reward = throughput
            elif term == "LEP":
                reward = -energy * runtime
            elif term == "LAP":
                reward = -area * runtime
            elif term == "EAP":
                reward = -area * energy
            elif term == "thrpt" or term == "thrpt_naive":
                reward = throughput
            elif term == "thrpt_btnk":
                reward = throughput
            elif term == "latency":
                reward = -runtime
            elif term == "area":
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
        if any(used > self.resource_bound):
            return False
        return True


    ##### Bayesian search
    def black_box_function(self, **args):

        sol = np.zeros((len(self.model_defs),2))
        for i in range(len(self.model_defs)):
            sol[i,:] = action_space[0][int(args["{}".format(2*i)])], action_space[1][int(args["{}".format(2*i + 1)])]

        reward, total_used_constraint = self.exterior_search(sol)
        if total_used_constraint >  self.constraint_value:
            reward = -2**63
        if(reward > self.best_reward):
            self.best_sol = sol
            self.best_reward = reward
            print("Epoch {}: new best award reward: {}".format(self.epoch, self.best_reward))
            self.fd.write("\nEpoch {}: new best award reward: {}".format(self.epoch, self.best_reward)) if self.fd else None
            self.best_rewards_iteration.append(self.epoch)
        self.best_rewards.append(self.best_reward)
        self.epoch += 1
        self.pbar.update(1)

        return reward

    def anneal_f(self, args):
        sol = np.zeros((len(self.model_defs), 2))
        for i in range(len(self.model_defs)):
            sol[i, :] = action_space[0][min(11,int(args[2*i]))], action_space[1][min(11,int(args[2*i+1]))]
        reward, total_used_constraint = self.exterior_search(sol)
        if total_used_constraint > self.constraint_value:
            reward = -2 ** 63
        return abs(reward)

    def print_fun(self,x, f,accepted):

        if (f < self.best_reward):
            self.best_sol = x
            self.best_reward = f
            print("Epoch {}: new best award reward: {}".format(self.epoch, self.best_reward))
            self.fd.write("\nEpoch {}: new best award reward: {}".format(self.epoch, self.best_reward)) if self.fd else None
            self.best_rewards_iteration.append(self.epoch)
            print(f)
        self.epoch += 1
        self.best_rewards.append(self.best_reward)
    def anealing_search(self, epochs, chkpt_file,fd=None):
        self.chkpt_file = chkpt_file
        self.start_range = start_range
        self.end_range = end_range - 1
        x0 =np.array([8 for _ in range(len(self.model_defs)*2)])
        self.fd = fd
        if self.best_reward is None:
            self.best_reward = float("Inf")

        mybounds = MyBounds(len(self.model_defs)*2)
        ret = optimize.basinhopping(self.anneal_f, x0,niter = epochs, accept_test = mybounds,stepsize=2, T=10,callback=self.print_fun)
        print("Best fitness: {}".format(ret.fun))
        print("sol:{}".format([int(a) for  a in ret.x]))
        self.fd.write("Best fitness: {}".format(ret.fun)) if self.fd else None
        self.fd.write("\nsol:{}".format([int(a) for  a in ret.x])) if self.fd else None
        self.save_chkpt()
    def bayesian_search(self, epochs, chkpt_file,fd=None):
        self.fd = fd
        self.chkpt_file = chkpt_file
        self.pbar = tqdm(total=epochs)
        self.start_range = start_range
        self.end_range = end_range-1
        self.epoch = 0
        self.count_invalid = 0

        self.num_init_points = 10
        if self.best_reward is None:
            self.best_reward = float("-Inf")


        pbounds = {"{}".format(i): (self.start_range, self.end_range) for i in range(len(self.model_defs) * 2)}
        optimizer = BayesianOptimization(
            f=self.black_box_function,
            pbounds=pbounds,
            verbose=1,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )
        optimizer.maximize(
            init_points=self.num_init_points,
            n_iter=epochs - self.num_init_points,
        )

        self.save_chkpt()


    ##### Random search
    def random_search(self, max_epoch, chkpt_file,fd=None):
        self.fd= fd
        self.chkpt_file = chkpt_file
        self.start_range =start_range
        self.end_range = end_range-1

        n_layer = len(self.model_defs)
        print("Num layers: {}".format(n_layer))
        assert(self.count_invalid == 0)
        if self.best_reward is None:
            self.best_reward = float("-Inf")
        for epoch in range(max_epoch):
            self.epoch = epoch
            guess_action = []
            for _ in range(n_layer):
                self.start_range = start_range
                self.end_range = end_range-1
                pe = action_space[0][random.randint(self.start_range, self.end_range)]
                bf = action_space[1][random.randint(self.start_range, self.end_range)]

                action = [pe, bf]
                guess_action.append(action)

            reward, total_used_constraint = self.exterior_search(guess_action)
            if total_used_constraint > self.constraint_value:
                reward = float("-Inf")
                self.count_invalid += 1
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_sol = guess_action
                self.best_sol_constraint = total_used_constraint
                print("Epoch {}: new best award reward: {}".format(self.epoch, self.best_reward))
                self.fd.write("\nEpoch {}: new best award reward: {}".format(self.epoch, self.best_reward)) if self.fd else None
                self.best_rewards_iteration.append(epoch)
            self.best_rewards.append(self.best_reward)
        self.save_chkpt()
        return self.best_rewards, self.best_sol

    def dfs(self, left_layers, guess_action, stride):
        if self.epochs > 0:
            if(self.epoch > self.epochs):
                return
            if self.epoch == self.epochs:
                self.save_chkpt()
                print("Number of cases covered in this algorithms: " + str(self.epochs))
                print("Number of invalid cases: " + str(self.count_invalid))
                self.epoch += 1
                return

        if left_layers == 0:
            reward, total_used_constraint = self.exterior_search(guess_action)
            if total_used_constraint> self.constraint_value:
                reward = float("-Inf")
                self.count_invalid += 1
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_sol = guess_action
                print("Epoch {}: new best award reward: {}".format(self.epoch, self.best_reward))
                self.fd.write("\nEpoch {}: new best award reward: {}".format(self.epoch, self.best_reward)) if self.fd else None
                self.best_rewards.append(self.best_reward)
                self.best_rewards_iteration.append(self.epoch)


            if self.epoch %100==0:
                self.save_chkpt()

            self.epoch +=1
            return


        for pe in [2**i for i in range(self.start_range, self.end_range+1, stride)]:
            for bf in [2**i for i in range(self.start_range, self.end_range+1, stride)]:
                action=[pe, bf]
                guess_action.append(action)
                self.dfs(left_layers-1, guess_action,stride)
                guess_action.pop()

    def exhaustive_search(self, epochs, chkpt_file, stride = 5,fd = None):
        self.epochs = epochs
        self.fd = fd
        self.count_invalid = 0
        self.chkpt_file = chkpt_file
        self.start_range = start_range
        self.end_range = end_range - 1

        if self.best_reward is None:
            self.best_reward = float("-Inf")

        n_layer = len(self.model_defs)

        num_candidates_pe = int((end_range -start_range) / stride) + 1
        count = 0

        for pe in [2**i for i in range(start_range, end_range+1,stride)]:
          count += 1

        num_candidates_pe = count

        print("Num candidates pe: "+ str(num_candidates_pe))
        print("Num layers of this model: " + str(n_layer))
        print("Total design space: " + str(num_candidates_pe**(2*n_layer)))

        guess_action = []
        self.best_rewards = []
        self.epoch = 0
        self.best_sol = None
        self.dfs(n_layer, guess_action, stride)

        return self.best_rewards, self.best_sol

    def genetic_search(self, epochs=100, chkpt_file="genetic_chkpt.plt",fd=None):

        num_pop = 50
        num_gen = epochs // num_pop
        num_parents = 10
        self.fd = fd
        self.chkpt_file = chkpt_file
        self.start_range = start_range
        self.end_range = end_range-1

        if self.best_reward is None:
            self.best_reward = float("-Inf")
        self.best_rewards = []
        self.epoch = 0
        self.best_sol = None
        num_layers = len(self.model_defs)


        self.num_generations = num_gen
        self.num_population = num_pop
        self.num_parents = num_parents

        print("Number of generations: " + str(num_gen))
        print("Number of population: " + str(num_pop))
        print("Number of parents: " + str(num_parents))

        new_population = np.empty((num_pop,num_layers,2),dtype=int)
        guess_action = np.empty((num_layers,2 ))
        count = 0
        while(True):
            for i in range(num_layers):
                pe = random.randint(self.start_range, self.end_range)
                bf = random.randint(self.start_range, self.end_range)

                action = np.array([pe, bf],dtype=int)
                guess_action[i] = action

            if 1:
                new_population[count] = guess_action
                guess_action = np.empty((num_layers,2 ))
                count += 1
                if(count == num_pop):
                    break
        print("[SYSTEM] Generated intial {} population".format(num_pop))
        fitness = np.empty(num_pop, float)
        for i in range(num_pop):
            action_id = new_population[i]
            action = np.zeros(action_id.shape)
            for p in range(len(action_id)):
                action[p,0] = action_space[0][action_id[p][0]]
                action[p,1] = action_space[1][action_id[p][1]]

            reward,total_used_constraint  = self.exterior_search(action)
            if reward is None:
                print("Error with reward")
                print(new_population[i])
                exit(-1)
            if total_used_constraint > self.constraint_value:
                reward = float("-Inf")
            fitness[i] = reward



        iteration = 0
        for generation in tqdm(range(num_gen)):
            best_gen_reward =None
            parents = select_parents(new_population, fitness,
                                            num_parents, num_layers)

            offspring_crossover = crossover(parents,
                                            num_pop-num_parents,
                                            num_layers)
            offspring_mutation = mutation(offspring_crossover, num_layers)

            new_population[0:parents.shape[0], :] = parents
            new_population[parents.shape[0]:, :] = offspring_mutation
            num_invalid_node = 0

            for i in range(num_pop):
                action_id = new_population[i]
                action = np.zeros(action_id.shape)
                for p in range(len(action_id)):
                    action[p, 0] = action_space[0][action_id[p][0]]
                    action[p, 1] = action_space[1][action_id[p][1]]

                reward, total_used_constraint = self.exterior_search(action)
                if reward is None:
                      print("Error with reward")
                      print(new_population[i])
                      exit(-1)
                if total_used_constraint > self.constraint_value:
                    reward = float("-Inf")
                    num_invalid_node += 1
                if reward > self.best_reward:
                    best_gen_reward = reward
                    self.best_reward = reward
                    self.best_sol = new_population[i]

                fitness[i] = reward
                iteration += 1
                self.best_rewards_iteration.append(iteration)
                self.best_rewards.append(self.best_reward)
            if best_gen_reward  is not None:
                self.fd.write("\nGeneration {}: new best award reward: {:9e}".format(generation+1, self.best_reward)) if self.fd else None
                print("\nGeneration {}: new best award reward: {:9e}".format(generation+1, self.best_reward))
            self.count_invalid += num_invalid_node
            self.save_chkpt()
        self.save_chkpt()






    def load_chkpt(self, chkpt):
        self.reward_rec = chkpt["reward_rec"]
        self.best_reward = chkpt["best_reward"]
        self.best_rewards= chkpt["best_rewards"]
        self.best_sol= chkpt["best_sol"]
        self.worst_reward = chkpt["worst_reward"]
    def get_chkpt(self):
        return  {
            "reward_rec":self.reward_rec,
            "best_rewards": self.best_rewards,
            "best_rewards_iteration": self.best_rewards_iteration,
            "best_sol": self.best_sol,
            "update_best_sol": self.update_best_sol,
            "best_reward": self.best_reward,
            "worst_reward": self.worst_reward,
            "count_invalid": self.count_invalid,
            "start_range": self.start_range,
            "end_range": self.end_range,
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
            table_entry = tuple(maestro_state)
            if table_entry in self.exp_table:
                reward, constraint = self.exp_table[table_entry]
            else:
                reward, constraint = self.oberserve_maestro(maestro_state)
                self.exp_table[table_entry] = (reward, constraint)
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

        actions = state[-self.n_action_steps:]
        if self.n_action_steps == 2:
            num_pe, KTileSz = actions.astype(int).squeeze()
            ClusterSz = num_pe
            self.resource_state = [num_pe, KTileSz]
            if self.is_gemm:
                self.write_maestro_gemm(dimension, dataflow=self.dataflow, KTileSz=KTileSz, ClusterSz=ClusterSz,
                                        m_file=m_file)
            else:
                self.write_maestro(dimension, dataflow=self.dataflow, KTileSz=KTileSz, ClusterSz=ClusterSz,
                                   m_file=m_file)
        else:
            num_pe, KTileSz, df_idx = actions.astype(int).squeeze()
            self.resource_state = [num_pe, KTileSz]
            ClusterSz = num_pe
            if self.is_gemm:
                self.write_maestro_gemm(dimension, dataflow=self.dataflow, KTileSz=KTileSz, ClusterSz=ClusterSz,
                                        m_file=m_file, df_idx=df_idx)
            else:
                self.write_maestro(dimension, dataflow=self.dataflow, KTileSz=KTileSz, ClusterSz=ClusterSz,
                                   m_file=m_file, df_idx=df_idx)





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

            return self.judge()
        except:
            print("+"*20)
            print(num_pe, KTileSz, ClusterSz)
            print("+" * 20)
            return None


    def write_maestro_gemm(self, dimension, dataflow="dla", KTileSz=1, CTileSz=1, ClusterSz=4, m_file=None, layer_id=0,df_idx=None):
        if df_idx is not None:
            dataflow = df_dict[df_idx]
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
