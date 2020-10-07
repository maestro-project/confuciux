'''
Working version for both GEMM and nonGEMM
'''
import sys, os, glob
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
from other_rl_env import MaestroEnvironment

import argparse
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import pickle
from src.utils.get_action_space import *


import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
from stable_baselines import DDPG, TD3,SAC, PPO2, HER, DQN,ACKTR,A2C,ACER,TRPO
from datetime import datetime
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)


os.environ["OPENAI_LOG_FORMAT"] = 'stdout,log,csv'

best_mean_reward, n_steps = -np.inf, 0
time_steps = 50000


best_reward_global,num_episodes = float('-Inf'),0



class NormalizeActionWrapper(gym.Wrapper):
    def __init__(self, env):
        action_space = env.action_space
        # Retrieve the action space
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.low, self.high = action_space.low, action_space.high

        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)

        # Call the parent constructor, so we can access self.env later
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
      """
      Rescale the action from [-1, 1] to [low, high]
      (no need for symmetric action space)
      :param scaled_action: (np.ndarray)
      :return: (np.ndarray)
      """
      return self.low + (0.5 * (scaled_action + 1.0) * (self.high -  self.    low))

    def reset(self):
        """
        Reset the environment
        """
        # Reset the counter
        return self.env.reset()

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.rescale_action(action)
        obs, reward, done, info = self.env.step(rescaled_action)
        return obs, reward, done, info




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default="A2C", help='Algorithm: [PPO2, A2C, ACKTR, SAC, TD3, DDPG]',
                        choices=["PPO2", "A2C", "ACKTR", "SAC", "TD3", "DDPG"] )
    parser.add_argument('--ent', default=0.01, type=float, help='use pre-trained')
    parser.add_argument('--lam', default=0.95, type=float, help='use pre-trained')
    parser.add_argument('--discount', default=0.99, type=float, help='use pre-trained')
    parser.add_argument('--outdir', type=str, default="outdir", help='output directiory')
    parser.add_argument('--model', type=str, default="example", help='The experimenting model.')
    parser.add_argument('--fitness', type=str, default="latency", help='The objective.')
    parser.add_argument('--cstr', type=str, default="area", help='The constraint.')
    parser.add_argument('--mul', type=float, default=0.5, help='The resource ratio, the design is allowed to use.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--gpu', default=0, type=int, help='which gpu')
    parser.add_argument('--df', default="shi", type=str, help='The dataflow strategy.')
    opt = parser.parse_args()

    ratio = opt.mul
    now = datetime.now()
    now_date = "{}".format(now.date())
    now_time = "{}".format(now.time())
    is_discrete = True if opt.alg in ["A2C", "ACKTR", "PPO2"] else False
    n_acts = 2
    dis_or_cont = "D" if is_discrete else "C"
    alg = "REINFORCE"
    outdir = opt.outdir
    outdir = os.path.join("../../", outdir)
    exp_name = "{}_F-{}_C-{}-Mul-{}_DF-{}_{}_{}".format(opt.model, opt.fitness, opt.cstr, opt.mul, opt.df, opt.alg,dis_or_cont)

    outdir_exp = os.path.join(outdir, exp_name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_exp, exist_ok=True)
    chkpt_file_t = os.path.join(outdir_exp, "{}".format("result"))


    action_space, action_bound, action_bottom = get_action_space()
    m_file_path = "../../data/model/"
    m_file = os.path.join(m_file_path, opt.model + ".csv")
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()
    _, dim_size = model_defs.shape

    agent_file = chkpt_file_t + "_ag.plt"
    outfile = chkpt_file_t + "_o.plt"
    chkpt_file = chkpt_file_t + "_c.plt"
    img_file = chkpt_file_t + ".png"
    log_file = chkpt_file_t + ".csv"

   
    ent_coef = opt.ent
    lam = opt.lam
    discount = opt.discount
    # model_def = [256,256,256]
    model_def = [512,512,512]
    # model_def = [512,512,512,512,512]
    noise_par = 0.05


    time_steps = opt.epochs * len(model_defs)
    action_space, action_bound, action_bottom = get_action_space()
    env = MaestroEnvironment(model_defs=model_defs,dim_size=dim_size, resource_size=2,n_action_steps=2,is_discrete=is_discrete,dataflow=opt.df,chkpt_file=chkpt_file)
    # from my_env_dla_gym_v3 import MaestroEnvironment
    # env = MaestroEnvironment(model_defs=model_defs, dim_size=6, resource_size=2, n_action_steps=2, action_size=12)
    env.set_fitness(opt.fitness)
    env.set_constraint(opt.cstr)
    constraint_temp = [env.get_ref_constraint([action_bound[0], action_bound[1]]),
                       env.get_ref_constraint([action_bottom[0], action_bottom[1]]), env.get_ref_constraint([action_bound[0], action_bottom[1]]),
                       env.get_ref_constraint([action_bottom[0], action_bound[1]])]
    max_constraint, min_constraint = max(constraint_temp), min(constraint_temp)
    print("Max constraint: {}".format(max_constraint))
    print("Min constraint: {}".format(min_constraint))
    set_constraint = min_constraint + (max_constraint - min_constraint) * ratio
    env.set_constraint_value(max_constraint, min_constraint, set_constraint)
    print("Set constraint: {}".format(set_constraint))
    if is_discrete is False:
        env = NormalizeActionWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env,norm_obs=False)


    max_grad_norm = 50
    ent_coef=0.05
    lam = 0.9
    nminibatches = 4
    n_step = (len(model_defs) + 1) * nminibatches

    ##=====================
    try:
        if opt.alg == "DDPG":
            n_actions = env.action_space.shape[-1]
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(noise_par) * np.ones(n_actions))

            agent = DDPG('MlpPolicy', env, verbose=1, param_noise=None, action_noise=action_noise,
                         policy_kwargs=dict(layers=model_def), tensorboard_log=None)
        elif opt.alg == "SAC":
            agent = SAC('MlpPolicy', env, verbose=1, policy_kwargs=dict(layers=model_def),
                        tensorboard_log=None)
        elif opt.alg == "TD3":
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_par * np.ones(n_actions))
            agent = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, policy_kwargs=dict(layers=model_def),
                        tensorboard_log=None)
        elif opt.alg == "A2C":
            agent = A2C("MlpPolicy", env, verbose=1, tensorboard_log=None,policy_kwargs=dict(layers=model_def))
        elif opt.alg == "ACKTR":
            agent = ACKTR("MlpPolicy", env, verbose=1, tensorboard_log=None,policy_kwargs=dict(layers=model_def))
        elif opt.alg == "PPO2":
            agent = PPO2("MlpPolicy", env, verbose=1, tensorboard_log=None, policy_kwargs=dict(layers=model_def))
        else:
            print("Please choose from [PPO2, A2C, ACKTR, SAC, TD3, DDPG].")
        agent.learn(total_timesteps=time_steps,
                        tb_log_name=now_time)





        agent.save(agent_file)

        # ============================Open save file============================================================================================

        length = -1
        with open(chkpt_file, "rb") as fd:
            chkpt = pickle.load(fd)

        best_rewards = chkpt["best_rewards"][:length]
        reward_rec = chkpt["reward_rec"][:length]
        best_sol = chkpt["best_sol"]["sol"]
        best_sol_ctr = chkpt["best_sol"]["ctr"]
        sols = chkpt["sols"]
        sol_reward_record = chkpt["sol_reward_record"]
        set_constraint = chkpt["ctrs_info"]["value"]
        max_constraint = chkpt["ctrs_info"]["max"]
        min_constraint = chkpt["ctrs_info"]["min"]
        # ========================================================================================================================
        # ==========================Do plotting======================================================================================

        best_sol = np.vstack(best_sol).astype(int)
        # best_sol = [a  for A in best_sol for a in A]
        best_reward_point = abs(best_rewards[-1])
        default_min = float("-inf")
        import bisect

        idx = bisect.bisect_right(best_rewards, default_min)
        best_rewards[:idx] = [best_rewards[idx] - 1 for _ in range(idx)]
        print("Start to valid at {} epoch".format(idx))
        print("Best  fitness :{:9e}".format(best_reward_point))
        print("Sol:\n {}\n".format(best_sol))
        print("Used constraint: {}".format(best_sol_ctr))
        print("Set constraint: {} [Constraint range : ({}, {})]".format(set_constraint, min_constraint, max_constraint))
        with open(log_file, "w") as fd:
            fd.write("best rewards: {}\n".format(best_reward_point))
            fd.write("best sol:\n {}\n".format(best_sol))
            fd.write("Used constraint: {}\n".format(best_sol_ctr))
            fd.write(
                "Set constraint: {} [Constraint range : ({}, {})]\n".format(set_constraint, min_constraint, max_constraint))
            fd.write("Model: {}\n".format(opt.model))
            fd.write("{}".format(model_defs))

        #
        font = {
            'weight': 'bold',
            'size': 12}
        import matplotlib
        matplotlib.rc('font', **font)

        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(best_rewards)), np.abs(np.array(best_rewards)), label="{}".format(opt.alg), linewidth=5)
        # plt.plot(np.arange(len(reward_rec)), np.abs(np.array(reward_rec)), label="RL_track")
        plt.figtext(0, 0, "best_fitness: {}".format(best_reward_point))
        plt.figtext(0, 0.05, "Model: {}".format(opt.model))
        plt.yscale("log")
        # plt.xlim(right=200)
        plt.ylabel(opt.fitness)
        plt.legend()
        fig.tight_layout()
        plt.xlabel('Episode #')
        plt.savefig(img_file, dpi=300)
        plt.show()
    finally:
        for f in glob.glob("*.m"):
            os.remove(f)
        for f in glob.glob("*.csv"):
            os.remove(f)

