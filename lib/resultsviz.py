import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common.results_plotter import load_results, ts2xy

from .folder_paths import get_logging_dir


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
    x, y = ts2xy(load_results(run_log_dir), "timesteps")
    # Smooth the values
    y = moving_average(y, window)
    # Truncate x
    x = x[len(x) - len(y) :] 

    return x,y
    

def plot_single_run_rewards(exp_name, run_no, window):
    # Get directories
    models_dir, log_dir, gif_dir, image_dir = get_logging_dir(exp_name)

    # Load data from csv files
    run_log_dir = f"{log_dir}/run_{run_no}"

    # Consolidate to smoothed average timeseries data from vector environments
    x, y = smooth_ts2xy(run_log_dir, window)

    fig = plt.figure()
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(f"{exp_name}--run_{run_no} \n (moving avg window = {window})")
    plt.plot(x, y)

def plot_all_run_rewards(exp_name, window):
    # Get directories
    models_dir, log_dir, gif_dir, image_dir = get_logging_dir(exp_name)
    
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
    plt.title(f"{exp_name} \n Smoothing window = {window})")
    
    for run_no in run_nos:
        # Load data from csv files
        run_log_dir = f"{log_dir}/run_{run_no}"
        # Consolidate to smoothed average timeseries data from vector environments
        x, y = smooth_ts2xy(run_log_dir, window)
        plt.plot(x,y, label=f"run_{run_no}",linewidth=0.5)
    plt.legend()
    

    
    

def get_statistics_of_all_run_rewards(exp_name, window):

    # Get directories
    models_dir, log_dir, gif_dir, image_dir = get_logging_dir(exp_name)
    
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
    agg_functions = dict([(run_no, 'max') for run_no in runs])
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

def plot_all_run_average_rewards(exp_name, window, err_type="std"):
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
    plt.title(f"{exp_name} \n (Average over {NO_OF_RUNS} runs; Smoothing window = {window})")
        
    plt.plot(x, y_avg, 
             'r-', 
             label=f"Avg, smoothing={window} ",
            linewidth=0.5)  # Average line

    
    plt.fill_between(x,
                     error_lims[0], error_lims[1], 
                     color='gray', 
                     alpha=0.3, 
                     label=err_type,
                    linewidth=0.05)  # Shaded standard deviation
    
    # plt.legend()
    plt.show()
    


def gif_anim(exp_name, run_no, 
             model_type="best", # "last"
             anim_type="render", # # "wrapped_obs"
            ):
    