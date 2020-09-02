# ConfuciuX #



### Setup ###
* Create virtural env
```
conda create --name confxEnv python=3.6
conda activate confxEnv
```
* Install requirement
```
pip install -r requirements.txt
```

* Download cost model and build symbolic link
```
python build.py
```



### Run ###
* Run ConfuciuX
```
./run_ConfX.sh
```
* Run other RL algorithms
```
./run_otherRLs.sh
```
* Run other optimization methods
```
./run_otherOpts.sh
```

#### Parameter ####
* fitness: The fitness objective (latency/ energy)
* cstr: Constraint (area/ power)
* df: The dataflow strategy
* platform: The targetting platform (Cloud/ IoT/ eIoT)
* outdir: The output result directory
* epochs: Number of generation for the optimization
* model_def: The model to run (available model in model_dir)
* alg: The algorithm to run
   * For RL, choose from [PPO2, A2C, ACKTR, SAC, TD3, DDPG]
   * For optimization methods, choose from [genetic, random, bayesian, anneal, exhaustive]
##### To find out all the options
```
python main.py --help
```

### Contributor ###
* Sheng-Chun (Felix) Kao
* Geonhwa Jeong
* Tushar Krishna

### Citation ###
```
@inproceedings{confuciux,
    author       = {Kao, Sheng-Chun and Jeong, Geonhwa and Krishna, Tushar},
    title        = {{ConfuciuX: Autonomous Hardware Resource Assignment for DNN Accelerators using Reinforcement Learning}},
    booktitle     = {Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year          = {2020}
}
```