import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy

from .folder_paths import makeget_logging_dir, deconstruct_exp_name
from .env_utils import AtariWrapper_NoisyFrame, AtariWrapper_Compressed, make_atari_env_Custom_VecFrameStack


# Helper Functions
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def smooth_ts2xy(run_log_dir, window):
    try:
        isdir = os.path.isdir(run_log_dir)
    except FileNotFoundError:
        print(f"Folder does not exist: {run_log_dir}")
    x, y = ts2xy(load_results(run_log_dir), "timesteps")
    # Smooth the values
    y = moving_average(y, window)
    # Truncate x
    x = x[len(x) - len(y) :] 

    return x,y
    


# Training statistics and visualization
def plot_single_run_rewards(exp_name, run_no, window, savefig=False):
    # Get directories
    models_dir, log_dir, gif_dir, image_dir = makeget_logging_dir(exp_name)

    # Load data from csv files
    run_log_dir = f"{log_dir}/run_{run_no}"

    # Consolidate to smoothed average timeseries data from vector environments
    x, y = smooth_ts2xy(run_log_dir, window)

    fig = plt.figure()
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(f"{exp_name}--run_{run_no} (Training) \n (moving avg window = {window})")
    plt.plot(x, y)
    if savefig:
        fig_filename = f"{exp_name}--run_{run_no}--train.png"
        fig_file = pathlib.Path(image_dir / fig_filename)
        plt.savefig(fig_file, bbox_inches='tight')
    plt.close()

def plot_all_run_rewards(exp_name, window, savefig=False):
    # Get directories
    models_dir, log_dir, gif_dir, image_dir = makeget_logging_dir(exp_name)
    
    # get all subfolders in log_dir
    folder_list = [folder for folder in os.listdir(log_dir)]
    
    # list of folders with prefix "run"
    runs = sorted([i for i in folder_list for j in ["run"] if j in i])
    run_nos = [run[-1] for run in runs]
    NO_OF_RUNS = len(runs)

    fig_width = 7
    fig_height = 5
    fig = plt.figure(figsize=[fig_width,fig_height])
    
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(f"{exp_name} (Training) \n Smoothing window = {window})")
    
    for run_no in run_nos:
        # Load data from csv files
        run_log_dir = f"{log_dir}/run_{run_no}"
        # Consolidate to smoothed average timeseries data from vector environments
        x, y = smooth_ts2xy(run_log_dir, window)
        plt.plot(x,y, label=f"run_{run_no}",linewidth=0.5)
    plt.legend()

    if savefig:
        fig_filename = f"{exp_name}--all_runs--train.png"
        fig_file = pathlib.Path(image_dir / fig_filename)
        plt.savefig(fig_file, bbox_inches='tight')
    
    plt.close()
    
def get_runs_in_exp_name(exp_name):
    # Get directories
    models_dir, log_dir, gif_dir, image_dir = makeget_logging_dir(exp_name)
    
    # get all subfolders in log_dir
    folder_list = [folder for folder in os.listdir(log_dir)]
    
    # list of folders with prefix "run"
    runs = sorted([i for i in folder_list for j in ["run"] if j in i])
    
    return runs

    

def get_statistics_of_all_run_rewards(exp_name, window):

    # Get directories
    models_dir, log_dir, gif_dir, image_dir = makeget_logging_dir(exp_name)
    
    # get all subfolders in log_dir
    folder_list = [folder for folder in os.listdir(log_dir)]
    
    # list of folders with prefix "run"
    runs = sorted([i for i in folder_list for j in ["run"] if j in i])
    run_nos = [run[-1] for run in runs]
    NO_OF_RUNS = len(runs)
    
    # Initialize data frame
    df = pd.DataFrame()
    for run_no in run_nos:
        
        # Load data from csv files
        run_log_dir = f"{log_dir}/run_{run_no}"
    
        # Consolidate to smoothed average timeseries data
        x, y = smooth_ts2xy(run_log_dir, window)
        
        new_df = pd.DataFrame({'x': x, f"run_{run_no}": y})
        df = pd.concat([df, new_df], ignore_index=True)

    # Sort by timesteps from all runs
    df = df.sort_values('x')

    # combine rows with same timesteps
    # if multiple runs have rewards outputs at the same timestep, take the max
    # agg_functions = {'run_0': 'max', 'run_1': 'max', 'run_2': 'max', 'run_3': 'max' }
    agg_functions = dict([(run, 'max') for run in runs])
    df = df.groupby(df['x']).aggregate(agg_functions)

    # Interpolate the data for visualization
    df = df.interpolate(method='linear', limit_direction='forward', axis=0)

    # Convert NaN to zeros
    df = df.fillna(0)

    # Get statistics
    df['avg'] = df[runs].mean(axis=1)
    df['std'] = df[runs].std(axis=1)
    df['min'] = df[runs].min(axis=1)
    df['max'] = df[runs].max(axis=1)

    # Convert to numpy values
    x = df.index.values
    y_avg = df['avg'].to_numpy()
    y_std = df['std'].to_numpy()
    y_min = df['min'].to_numpy()
    y_max = df['max'].to_numpy()


    return x, y_avg, y_std, y_min, y_max, NO_OF_RUNS

def plot_all_run_average_rewards(exp_name, window, err_type="std", color="red", savefig=False):
    # Get directories
    models_dir, log_dir, gif_dir, image_dir = makeget_logging_dir(exp_name)
    
    x, y_avg, y_std, y_min, y_max, NO_OF_RUNS = get_statistics_of_all_run_rewards(exp_name, window)

    # Plotting
    fig_width = 7
    fig_height = 5
    fig = plt.figure(figsize=[fig_width,fig_height])

    if err_type == "std":
        error_lims = [y_avg-y_std, y_avg+y_std]
    elif err_type == "range":
        error_lims = [ y_min, y_max]
    else:
        err_type = ""
        error_lims = [0,0]

    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(f"{exp_name} (Training)\n (Average over {NO_OF_RUNS} runs; Smoothing window = {window}; Err={err_type})")
        
    plt.plot(x, y_avg, 
             color=color,
             linestyle="-",
             label=f"Avg, smoothing={window} ",
            linewidth=0.5)  # Average line

    
    plt.fill_between(x,
                     error_lims[0], error_lims[1], 
                     color=color, 
                     alpha=0.2, 
                     label=err_type,
                    linewidth=0.05)  # Shaded standard deviation
    if savefig:
        fig_filename = f"{exp_name}--all_runs_avg_{err_type}--train.png"
        fig_file = pathlib.Path(image_dir / fig_filename)
        plt.savefig(fig_file, bbox_inches='tight')
    plt.close()

# Evaluation statistics and visualization

###############################################################
def evaluate_single_run(# model_params
                        exp_name, 
                        run_no, 
                        model_type,
                        algorithm,
                        # env_params
                        n_envs, 
                        seed, 
                        eval_param_value, 
                        # eval_params
                        NUM_EPISODES):
                            
    # Get names and tags of experiment
    env_id, exp_param_type, exp_param_value, exp_metaname = deconstruct_exp_name(exp_name)

    # Get directories
    models_dir, log_dir, gif_dir, image_dir = makeget_logging_dir(exp_name)
    # Monitor Directory for evaluation run
    eval_param_type = exp_param_type
    eval_run_log_dir = f"{log_dir}/eval_{eval_param_type}_{eval_param_value}/eval_{run_no}"
            
    # Make vector environment
    if exp_param_type == "compress":
        wrapper = AtariWrapper_Compressed
        wrapper_kwargs = {"compress_ratio":float(eval_param_value)}
    elif exp_param_type == "noisy":
        wrapper = AtariWrapper_NoisyFrame
        wrapper_kwargs = {"noise":float(eval_param_value)}
    else:
        wrapper = AtariWrapper
        wrapper_kwargs = None
        
    trial_env = make_atari_env_Custom_VecFrameStack(env_id=env_id,
                                            n_envs=n_envs,
                                            monitor_dir=eval_run_log_dir,
                                            seed=seed,
                                            wrapper_class=wrapper,
                                            wrapper_kwargs=wrapper_kwargs
                                             )

      
    # Load RL model
    if model_type == "best":
        model_file = f"{models_dir}/{exp_name}-run_{run_no}-best"
    else:
        model_file = f"{models_dir}/{exp_name}-run_{run_no}"
    model = algorithm.load(model_file)

    mean_reward, std_reward = evaluate_policy(model, 
                                    trial_env, 
                                    n_eval_episodes=NUM_EPISODES, 
                                    deterministic=True, 
                                    render=False, 
                                    callback=None, 
                                    reward_threshold=None, 
                                    return_episode_rewards=False, 
                                    warn=True)
    return mean_reward, std_reward

def evaluate_all_runs(# model_params
                    exp_name, 
                    model_type,
                    algorithm,
                    # env_params
                    n_envs, 
                    seed, 
                    eval_param_value, 
                    # eval_params
                    NUM_EPISODES):
    
    # Get names and tags of experiment
    env_id, exp_param_type, exp_param_value, exp_metaname = deconstruct_exp_name(exp_name)
    
    # Get directories
    models_dir, log_dir, gif_dir, image_dir = makeget_logging_dir(exp_name)
    
    # get all subfolders in log_dir
    folder_list = [folder for folder in os.listdir(log_dir)]
    
    # list of folders with prefix "run"
    runs = sorted([i for i in folder_list for j in ["run"] if j in i])
    run_nos = [run[-1] for run in runs]
    NO_OF_RUNS = len(runs)
    
    eval_results = {}
    for run_no in run_nos:
        eval_results[f"run_{run_no}"] = {}
    
        mean_reward, std_reward = evaluate_single_run(# model_params
                                                        exp_name, 
                                                        run_no, 
                                                        model_type,
                                                        algorithm,
                                                        # env_params
                                                        n_envs, 
                                                        seed, 
                                                        eval_param_value, 
                                                        # eval_params
                                                        NUM_EPISODES)
        
        eval_results[f"run_{run_no}"]["avg"] = mean_reward
        eval_results[f"run_{run_no}"]["std"] = std_reward
    
    # If the keys of the passed dict should be the columns of the resulting DataFrame, 
    # pass ‘columns’ (default) Otherwise, if the keys should be rows, pass ‘index’. 
    df = pd.DataFrame.from_dict(eval_results, orient='index')
            
    # Save csv
    eval_param_type = exp_param_type                    
    csv_filename = f"{exp_name}--eval_{model_type}-{eval_param_type}_{eval_param_value}.csv"
    csv_file = pathlib.Path(log_dir / csv_filename)
    df.to_csv(csv_file, sep='\t', encoding='utf-8', header='true')
        
    return df

def barplot_evaluate_all_runs(exp_name, 
                                model_type,
                                eval_param_value, 
                                savefig=False):
    
    # Get names and tags of experiment
    env_id, exp_param_type, exp_param_value, exp_metaname = deconstruct_exp_name(exp_name)                   
    eval_param_type = exp_param_type

    # Get directories
    models_dir, log_dir, gif_dir, image_dir = makeget_logging_dir(exp_name)
                            
    csv_filename = f"{exp_name}--eval_{model_type}-{eval_param_type}_{eval_param_value}.csv"
    csv_file = pathlib.Path(log_dir / csv_filename)
    df = pd.read_csv(csv_file, sep='\t', encoding='utf-8')
                    
    global_avg = np.mean(df["avg"])
    global_std = np.std(df["avg"])

    # Plotting
    fig_width = 7
    fig_height = 5
    fig = plt.figure(figsize=[fig_width,fig_height])
                        
    plt.bar(x=df.index,height=df["avg"], yerr=df["std"], capsize=5)

    plt.xlabel("Run")
    plt.ylabel("Average Reward")
    plt.title(f"{exp_name} ({exp_param_type} during evaluation: {eval_param_value}) \nAverage over all runs: {global_avg:0.2f} \u00B1 {global_std:0.2f} (Using {model_type} model)")
                                
    
    if savefig:
        fig_filename = f"{exp_name}--eval_all_runs-{model_type}-{eval_param_type}_{eval_param_value}.png"
        fig_file = pathlib.Path(image_dir / fig_filename)
        plt.savefig(fig_file, bbox_inches='tight')
        
    plt.close()
    print("---")
                                
###############################################################

# def eval_single_run(# model_params
#                     exp_name, 
#                     run_no, 
#                     model_type,
#                     algorithm,
#                     # env_params
#                     n_envs, 
#                     seed, 
#                     sparsity, 
#                     # eval_params
#                     NUM_EPISODES):

#     # Get env_id and sparsity
#     env_id, exp_tag = exp_name.split("--")
        
#     # Create environment
#     # when training xxx-sparse_xx agents, the frame was flipped upsidedown by mistake in MyAtariWrapper
    
#     # When agent is vanilla, don't flip frame
#     if exp_tag == "vanilla":
#         # flip the image to correct orientation
#         trial_env = make_trial_env_nonflipped(env_id, n_envs, seed, sparsity)
#     else:
#         trial_env = make_trial_env(env_id, n_envs, seed, sparsity)
    
#     # Get directories
#     models_dir, log_dir, gif_dir, image_dir = makeget_logging_dir(exp_name)
       
#     # Load RL model
#     if model_type == "best":
#          model_file = f"{log_dir}/run_{run_no}/best_model.zip"
#     else:
#         model_file = f"{models_dir}/{exp_name}-run_{run_no}"
#     model = algorithm.load(model_file)

#     mean_reward, std_reward = evaluate_policy(model, 
#                                     trial_env, 
#                                     n_eval_episodes=NUM_EPISODES, 
#                                     deterministic=True, 
#                                     render=False, 
#                                     callback=None, 
#                                     reward_threshold=None, 
#                                     return_episode_rewards=False, 
#                                     warn=True)
#     return mean_reward, std_reward




# def eval_all_runs(# model_params
#                 exp_name, 
#                 model_type,
#                 algorithm,
#                 # env_params
#                 n_envs, 
#                 seed, 
#                 sparsity, 
#                 # eval_params
#                 NUM_EPISODES):
    
#     # Get directories
#     models_dir, log_dir, gif_dir, image_dir = makeget_logging_dir(exp_name)
    
#     # get all subfolders in log_dir
#     folder_list = [folder for folder in os.listdir(log_dir)]
    
#     # list of folders with prefix "run"
#     runs = sorted([i for i in folder_list for j in ["run"] if j in i])
#     run_nos = [run[-1] for run in runs]
#     NO_OF_RUNS = len(runs)

#     eval_results = {}
#     for run_no in run_nos:
#         eval_results[f"run_{run_no}"] = {}

#         mean_reward, std_reward = eval_single_run(# model_params
#                                                     exp_name, 
#                                                     run_no, 
#                                                     model_type,
#                                                     algorithm,
#                                                     # env_params
#                                                     n_envs, 
#                                                     seed, 
#                                                     sparsity, 
#                                                     # eval_params
#                                                     NUM_EPISODES)
        
#         eval_results[f"run_{run_no}"]["avg"] = mean_reward
#         eval_results[f"run_{run_no}"]["std"] = std_reward

#     # If the keys of the passed dict should be the columns of the resulting DataFrame, 
#     # pass ‘columns’ (default) Otherwise, if the keys should be rows, pass ‘index’. 
#     df = pd.DataFrame.from_dict(eval_results, orient='index')
        
#     return df

def plot_eval_all_runs(# model_params
                        exp_name, 
                        model_type,
                        algorithm,
                        # env_params
                        n_envs, 
                        seed, 
                        sparsity, 
                        # eval_params
                        NUM_EPISODES, 
                        # fig params
                        savefig=False):

    # Get directories
    models_dir, log_dir, gif_dir, image_dir = makeget_logging_dir(exp_name)
                            
    df = eval_all_runs(# model_params
                exp_name, 
                model_type,
                algorithm,
                # env_params
                n_envs, 
                seed, 
                sparsity, 
                # eval_params
                NUM_EPISODES)

    csv_filename = f"{exp_name}--eval_{model_type}-SR_{sparsity}.csv"
    csv_file = pathlib.Path(image_dir / csv_filename)
    df.to_csv(csv_file, sep='\t', encoding='utf-8', header='true')
                    
    global_avg = np.mean(df["avg"])
    global_std = np.std(df["avg"])

    # Plotting
    fig_width = 7
    fig_height = 5
    fig = plt.figure(figsize=[fig_width,fig_height])
                        
    plt.bar(x=df.index,height=df["avg"], yerr=df["std"], capsize=5)

    plt.xlabel("Run")
    plt.ylabel("Average")
    plt.title(f"{exp_name} (Evaluation on env with SR={sparsity}) \nAverage over all runs: {global_avg:0.2f} \u00B1 {global_std:0.2f} (Using {model_type} model)")
    if savefig:
        fig_filename = f"{exp_name}--eval_{model_type}-SR_{sparsity}.png"
        fig_file = pathlib.Path(image_dir / fig_filename)
        plt.savefig(fig_file, bbox_inches='tight')
        
    plt.close()
    print(fig_filename)
    print("---")


def generate_frames(trial_env, model, duration=10, fps=120):
    no_of_frames = duration*fps*2 #only every other frame is used for animation
    
    images = []
    obss = []
    obs = trial_env.reset()
    img = trial_env.render(mode="rgb_array")
    for i in range(no_of_frames):
        images.append(img)
        obss.append(obs[0,:,:,-1])
        action, _ = model.predict(obs)
        obs, reward, done, info = trial_env.step(action)
        img = trial_env.render(mode="rgb_array")
    return obss, images
    
    
def generate_gif_single_run(# model_params
                                exp_name,
                                run_no,
                                model_type, 
                                algorithm,
                            # env params
                                seed,  
                                sparsity,
                            # anim params
                                duration=5, 
                                fps=120):
    
    # Get env_id and sparsity
    env_id, exp_tag = exp_name.split("--")
                                                                     
    # Get directories
    models_dir, log_dir, gif_dir, image_dir = makeget_logging_dir(exp_name)

    # Create environment
    # when training xxx-sparse_xx agents, the frame was flipped upsidedown by mistake in MyAtariWrapper
    if exp_tag == "vanilla":
        # flip the image to correct orientation
        trial_env = make_trial_env_nonflipped(env_id=env_id,
                                               n_envs=1,
                                               seed=seed,
                                               sparsity=sparsity)
    else:
        trial_env = make_trial_env(env_id=env_id,
                           n_envs=1,
                           seed=seed,
                           sparsity=sparsity)


        
       
    # Load RL model
    if model_type == "best":
         model_file = f"{log_dir}/run_{run_no}/best_model.zip"
    else:
        model_file = f"{models_dir}/{exp_name}-run_{run_no}"
    model = algorithm.load(model_file)

    # Generate frames    
    obss, images = generate_frames(trial_env, model, duration, fps)
        
    # Convert frames to animation
    img_gif_file = f"{gif_dir}/{exp_name}-run_{run_no}--eval_{model_type}-SR_{sparsity}--img.gif"
    imageio.mimsave(img_gif_file, 
                    [np.array(img) for i, img in enumerate(images) if i%2 == 0], duration=duration)

    # Convert obss to animation
    obs_gif_file = f"{gif_dir}/{exp_name}-run_{run_no}--eval_{model_type}-SR_{sparsity}--obs.gif"
    imageio.mimsave(obs_gif_file, 
                    [np.array(obs) for i, obs in enumerate(obss) if i%2 == 0], duration=duration)

    return obs_gif_file, img_gif_file


    