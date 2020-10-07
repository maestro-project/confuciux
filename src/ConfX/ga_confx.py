import numpy as np
import copy, random
# Not used
def get_population_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = np.sum(pop*equation_inputs, axis=1)
    return fitness

def select_parents(pop, fitness, num_parents, num_layers):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    idx = np.argsort(fitness[:,0])[::-1]
    pop[:] = pop[idx]
    fitness[:] = fitness[idx]
    parents = copy.deepcopy(pop[:num_parents])

    return parents




def self_crossover(pop, eps=0):
    for idx in range(1, pop.shape[0]):
        if random.random()< eps:
            picks = np.random.randint(0, pop.shape[1], 2)
            pick =  random.randint(0, 1)
            pop[idx][picks[0]][pick], pop[idx][picks[1]][pick] = pop[idx][picks[1]][pick], pop[idx][picks[0]][pick]

# def crossover(parents, pop, eps=0.1):
#     for idx in range(1, pop.shape[0], 2):
#         if random.random()<eps:
#             picks = np.random.randint(0, len(parents), 2)
#             parents_1 = parents[picks[0]]
#             parents_2 = parents[picks[1]]
#             pick_l = random.randint(0, len(parents[0]), 2)
#             pick = random.randint(0, 1, 2)
#             pop[idx]
#             pop[idx][pick_l[0]][pick] = parent[i][pick] + sampling

def crossover(parents, offspring_size, num_layers,eps=1):
    offspring = np.empty((offspring_size, num_layers, 2))
    crossover_point = np.uint8(num_layers/2)

    for k in range(offspring_size):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k][0:crossover_point] = parents[parent1_idx][0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k][crossover_point:] = parents[parent2_idx][crossover_point:]

    return offspring




def mutation_pe(parents, pop,  action_bound, action_bottom, env, fitness_list, pe_cstr = 8192, range_alpha=1):
    # Mutation changes a single gene in each offspring randomly.
    idx = 0
    count = 0
    max_count = pop.shape[0] * 2
    while idx <pop.shape[0]:
        max_count -= 1
        if max_count < 0:
            break
        # The random value to be added to the gene.
        parent = parents[idx % len(parents)]
        pop[idx] = copy.deepcopy(parent)
        reward, constraint = fitness_list[idx]
        constraint_budget = env.constraint_value - constraint
        num_pe = np.sum(p, 0)
        left_pe = pe_cstr - num_pe[0]
        succeed = 0
        for _ in range(3):
            p = pop[idx]

            pick = random.randint(0, 1)
            i = random.randint(0, len(pop[idx])-1)
            maestro_state = np.concatenate((env.model_defs[i], p[i]))
            reward_layer, constraint_layer = env.quick_observe(maestro_state)
            thr = max(1, action_bound[pick]//2)
            # thr = min(64, parent[i][pick]//2)
            sampling =  np.random.uniform(-range_alpha,range_alpha,1)
            # sampling = min(left_resource, int(sampling * thr))
            sampling = int(sampling * thr)
            saved_value = copy.deepcopy(p[i][pick])
            if pick == 0:
                adding_pe = min(left_pe,sampling)
                p[i][pick] = p[i][pick] + adding_pe
                left_pe = left_pe - adding_pe
            else:
                p[i][pick] = p[i][pick] + sampling

            p[i][pick] = min(max(action_bottom[pick],p[i][pick]), action_bound[pick])
            maestro_state = np.concatenate((env.model_defs[i], p[i]))
            reward_layer_new, constraint_layer_new = env.quick_observe(maestro_state)
            # if reward_layer_new < reward_layer:
            #     p[i][pick] = saved_value
            if constraint_layer_new - constraint_layer > constraint_budget:
                p[i][pick] = saved_value
            else:
                constraint_budget -= (constraint_layer_new - constraint_layer)
                succeed = 1
        if not succeed:
            count += 1
        else:
            idx += 1






def mutation_coarse(parents, pop,  action_bound, action_bottom, action_space,env, fitness_list, range_alpha=1):
    # Mutation changes a single gene in each offspring randomly.
    idx = 0
    count = 0
    max_count = pop.shape[0] * 2
    while idx <pop.shape[0]:
        max_count -= 1
        if max_count < 0:
            break
        # The random value to be added to the gene.
        parent = parents[idx % len(parents)]
        pop[idx] = copy.deepcopy(parent)
        reward, constraint = fitness_list[idx]
        constraint_budget = env.constraint_value - constraint
        succeed = 0
        for _ in range(1):
            p = pop[idx]

            pick = random.randint(0, 1)
            i = random.randint(0, len(pop[idx])-1)
            maestro_state = np.concatenate((env.model_defs[i], p[i]))
            reward_layer, constraint_layer = env.quick_observe(maestro_state)
            saved_value = copy.deepcopy(p[i][pick])

            choice = action_space[pick] * action_bound[pick]
            p[i][pick] = choice[random.randint(0,11)]
            # thr = max(1, action_bound[pick]//2)
            # # thr = min(64, parent[i][pick]//2)
            # sampling =  np.random.uniform(-range_alpha,range_alpha,1)
            # # sampling = min(left_resource, int(sampling * thr))
            # sampling = int(sampling * thr)
            # p[i][pick] = p[i][pick] + sampling
            # p[i][pick] = min(max(action_bottom[pick],p[i][pick]), action_bound[pick])
            # if pick==0:
            #     #=====method 1=============================
            #     choice = action_space[pick] * action_bound[pick]
            #     value = saved_value
            #     min_idx = np.argmin(np.abs(choice - value))
            #     p[i][pick] = choice[max(0, min_idx - 1)]
            #     if sampling <0:
            #         p[i][pick] = choice[max(0,min_idx-1)]
            #     else:
            #         p[i][pick] = choice[min(11, min_idx + 1)]

                #======method 2==============================
                # choice = action_space[pick]*action_bound[pick]
                # value = p[i][pick]
                # min_idx = np.argmin(np.abs(choice - value))
                # p[i][pick] = choice[min_idx]
            maestro_state = np.concatenate((env.model_defs[i], p[i]))
            reward_layer_new, constraint_layer_new = env.quick_observe(maestro_state)
            # if reward_layer_new < reward_layer:
            #     p[i][pick] = saved_value
            if constraint_layer_new - constraint_layer > constraint_budget:
                p[i][pick] = saved_value
            else:
                constraint_budget -= (constraint_layer_new - constraint_layer)
                succeed = 1
        if not succeed:
            count += 1
        else:
            idx += 1




def mutation(parents, pop,  action_bound, action_bottom, env, fitness_list, range_alpha=1):
    # Mutation changes a single gene in each offspring randomly.
    idx = 0
    count = 0
    max_count = pop.shape[0] * 2
    while idx <pop.shape[0]:
        max_count -= 1
        if max_count < 0:
            break
        # The random value to be added to the gene.
        parent = parents[idx % len(parents)]
        pop[idx] = copy.deepcopy(parent)
        reward, constraint = fitness_list[idx]
        constraint_budget = env.constraint_value - constraint
        succeed = 0
        for _ in range(3):
            p = pop[idx]

            pick = random.randint(0, 1)
            i = random.randint(0, len(pop[idx])-1)
            maestro_state = np.concatenate((env.model_defs[i], p[i]))
            reward_layer, constraint_layer = env.quick_observe(maestro_state)
            thr = max(1, action_bound[pick]//2)
            # thr = min(64, parent[i][pick]//2)
            sampling =  np.random.uniform(-range_alpha,range_alpha,1)
            # sampling = min(left_resource, int(sampling * thr))
            sampling = int(sampling * thr)
            saved_value = copy.deepcopy(p[i][pick])
            p[i][pick] = p[i][pick] + sampling
            p[i][pick] = min(max(action_bottom[pick],p[i][pick]), action_bound[pick])
            maestro_state = np.concatenate((env.model_defs[i], p[i]))
            reward_layer_new, constraint_layer_new = env.quick_observe(maestro_state)
            # if reward_layer_new < reward_layer:
            #     p[i][pick] = saved_value
            if constraint_layer_new - constraint_layer > constraint_budget:
                p[i][pick] = saved_value
            else:
                constraint_budget -= (constraint_layer_new - constraint_layer)
                succeed = 1
        if not succeed:
            count += 1
        else:
            idx += 1
    # print("Invalid num: {}".format(count))