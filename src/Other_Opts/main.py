import pandas as pd
import os,sys
import argparse
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
from src.utils.get_action_space import *
from other_opt_env import MaestroEnvironment
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import bisect
import glob
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default="outdir", help='output directiory')
    parser.add_argument('--model', type=str, default="example", help='The experimenting model.')
    parser.add_argument('--fitness', type=str, default="latency", help='The objective.')
    parser.add_argument('--cstr', type=str, default="area", help='The constraint.')
    parser.add_argument('--mul', type=float, default=0.5, help='The resource ratio, the design is allowed to use.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--gpu', type=int, default=0,  help='which gpu')
    parser.add_argument('--df', type=str, default="shi",  help='The dataflow strategy.')
    parser.add_argument('--alg', type=str, default="random", help='Please choose from [genetic, random, bayesian, anneal, exhaustive]'
                        , choices=["genetic", "random", "bayesian", "anneal", "exhaustive"])
    parser.add_argument('--stride', type=int, default=1, help='Set stride for exhaustive')



    opt = parser.parse_args()
    ratio = opt.mul
    method = opt.alg

    now = datetime.now()
    now_date = "{}".format(now.date())
    now_time = "{}".format(now.time())
    is_discrete = True
    n_acts = 2
    dis_or_cont = "D" if is_discrete else "C"
    alg = "REINFORCE"
    outdir = opt.outdir
    outdir = os.path.join("../../", outdir)
    exp_name = "{}_F-{}_C-{}_Mul-{}_DF-{}_{}_{}".format(opt.model, opt.fitness, opt.cstr, opt.mul, opt.df, method,
                                                        dis_or_cont)
    outdir_exp = os.path.join(outdir, exp_name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_exp, exist_ok=True)
    chkpt_file_t = os.path.join(outdir_exp, "{}".format("result"))

    outfile = chkpt_file_t + "_o.plt"
    chkpt_file = chkpt_file_t + "_c.plt"
    img_file = chkpt_file_t + ".png"
    log_file = chkpt_file_t + ".csv"
    expLog_file = chkpt_file_t + ".log"
    m_file_path = "../../data/model/"
    m_file = os.path.join(m_file_path, opt.model + ".csv")
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()
    _, dim_size = model_defs.shape
    # fd = open(expLog_file, "w")

    action_space, action_bound, action_bottom = get_action_space()
    env = MaestroEnvironment(model_defs=model_defs,dim_size=dim_size, dataflow=opt.df)
    env.set_fitness(opt.fitness)
    env.set_constraint(opt.cstr)
    constraint_temp = [env.get_ref_constraint([action_bound[0], action_bound[1]]),
                       env.get_ref_constraint([action_bottom[0], action_bottom[1]]),
                       env.get_ref_constraint([action_bound[0], action_bottom[1]]),
                       env.get_ref_constraint([action_bottom[0], action_bound[1]])]
    max_constraint, min_constraint = max(constraint_temp), min(constraint_temp)
    print("Max constraint: {}".format(max_constraint))
    print("Min constraint: {}".format(min_constraint))
    set_constraint = min_constraint + (max_constraint - min_constraint) * ratio

    env.set_constraint_value(max_constraint, min_constraint, set_constraint)
    print("Set constraint: {}".format(set_constraint))

    print("[SYSTEM] The optimization target is: " + opt.fitness)
    print("[SYSTEM] The search type is: " + method)

    try:
        if method == "random":
            env.random_search(opt.epochs, chkpt_file)
        elif method == "exhaustive":
            env.exhaustive_search(opt.epochs, chkpt_file = chkpt_file, stride = opt.stride)
        elif method == "genetic":

            env.genetic_search(epochs = opt.epochs, chkpt_file=chkpt_file)

        elif method == "bayesian":
            env.bayesian_search(opt.epochs, chkpt_file=chkpt_file)

        elif method == "anneal":
            env.anealing_search(opt.epochs, chkpt_file=chkpt_file)

        else:
            print("Please choose from [genetic, random, bayesian, anneal, exhaustive]")
            exit(-1)
        # fd.close()



        #=======Do Plotting===========================================================
        with open(chkpt_file, "rb") as fd:
            chkpt = pickle.load(fd)
        best_rewards = chkpt["best_rewards"]
        reward_rec = chkpt["reward_rec"]
        best_sol = chkpt["best_sol"]
        best_sol = np.vstack(best_sol).astype(int)
        best_reward_point = abs(best_rewards[-1])
        length = -1
        default_min = float("-inf")
        if method in ["genetic", "anneal"]:
            if method == "anneal":
                best_sol = np.concatenate((best_sol[::2], best_sol[1::2]), axis=1)
            best_sol = (np.array([[action_space[0][p], action_space[1][b]] for p, b in zip(best_sol[:,0], best_sol[:,1]) ])* action_bound[:2]).astype(int)
        reward, best_sol_ctr = env.exterior_search(best_sol)

        print("Search type: {}".format(method))
        best_sol = np.vstack(best_sol).astype(int)
        best_reward_point = abs(best_rewards[-1])
        default_min = float("-inf")

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

        font = {
            'weight': 'bold',
            'size': 12}
        import matplotlib
        matplotlib.rc('font', **font)

        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(best_rewards)), np.abs(np.array(best_rewards)), label="{}".format(method), linewidth=5)
        plt.figtext(0, 0, "best_fitness: {}".format(best_reward_point))
        plt.figtext(0, 0.05, "Model: {}".format(opt.model))
        plt.yscale("log")
        plt.ylabel(opt.fitness)
        plt.legend()
        plt.xlabel('Episode #')
        plt.savefig(img_file, dpi=300)
        plt.show()
    finally:
        for f in glob.glob("*.m"):
            os.remove(f)
        for f in glob.glob("*.csv"):
            os.remove(f)
    # ===================================================================================================================


