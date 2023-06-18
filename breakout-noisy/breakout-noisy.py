import argparse
import datetime

start = datetime.datetime.now()

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env_id', type=str)
parser.add_argument('--exp_param_type', type=str) # "compress"
parser.add_argument('--exp_param_value', type=float)
parser.add_argument('--run_no', type=int)
parser.add_argument('--n_envs', type=int)
parser.add_argument('--run_seed', type=int)
parser.add_argument('--trainsteps', type=int)
parser.add_argument('--gpu_id', type=int)

args= parser.parse_args()

env_id = args.env_id
exp_param_type = args.exp_param_type # "compress"
exp_param_value = args.exp_param_value # compression ratio for each frame
compress_ratio = exp_param_value
run_no = args.run_no
gpu_id = args.gpu_id
NUM_ENVS = args.n_envs # The different number of processes that will be used
TRAIN_STEPS = args.trainsteps # TRAIN_STEPS = 3E7 should result in 12E7 timesteps due to VecStack=4
run_seed = args.run_seed

#################################

import torch
torch.cuda.set_device(gpu_id)

import os
import sys
import git
import pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

print(f"Project Root Directory: {PROJ_ROOT}")

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import A2C

from lib.folder_paths import get_exp_name_tag, deconstruct_exp_name, makeget_logging_dir
from lib.env_utils import make_atari_env_Compressed_VecFrameStack, SaveBestModelCallback
from lib.vizresults import plot_single_run_rewards

# Get names and tags of experiment
exp_name, exp_metaname, exp_tag = get_exp_name_tag(env_id, exp_param_type, exp_param_value)

print(f"Experiment: {exp_name}")
print(f"Using device: {torch.cuda.current_device()}")

# Get directories
models_dir, log_dir, gif_dir, image_dir = makeget_logging_dir(exp_name)

# ALgorithm to use
ALGO = A2C

# Start Experiment
print("-------")
print(f"RUN: {run_no}")

# Log directory for each run of the experiment
run_log_dir = f"{log_dir}/run_{run_no}"
best_model_path = f"{models_dir}/{exp_name}-run_{run_no}-best"

# Create the callback: check every 1000 steps
callback = SaveBestModelCallback(check_freq=1E4, 
                                log_dir=run_log_dir,
                                save_path=best_model_path,
                                verbose=0)

# Make vector environment
env = make_atari_env_Compressed_VecFrameStack(env_id,
                                                n_envs=NUM_ENVS,
                                                monitor_dir=run_log_dir,
                                                seed=run_seed,
                                                compress_ratio=compress_ratio)
# Create RL model
model = A2C("CnnPolicy", env, verbose=0)

# Train the agent
model.learn(total_timesteps=TRAIN_STEPS, 
            progress_bar=False, 
            callback=callback)

# Save the final agent
model.save(f"{models_dir}/{exp_name}-run_{run_no}")

# save training plot
from lib.vizresults import plot_single_run_rewards
window=100
plot_single_run_rewards(exp_name, run_no, window, savefig=True)


end = datetime.datetime.now()
runtime = end - start
print(f"Experiment Finished. Runtime: {runtime}")
print("*"*20)