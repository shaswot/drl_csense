import os
import sys
import git
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

# print(f"Project Root Directory: {PROJ_ROOT}")

def get_exp_name_tag(env_id, exp_param_type, exp_param_value):
    
    exp_metaname = f"{env_id}--{exp_param_type}"
    exp_tag = f"{exp_param_type}_{exp_param_value}"
    exp_name = f"{env_id}--{exp_tag}"
    return exp_name, exp_metaname, exp_tag

def deconstruct_exp_name(exp_name):
    env_id, exp_tag = exp_name.split("--")
    exp_param_type, exp_param_value = exp_tag.split("_")
    exp_metaname = f"{env_id}--{exp_param_type}"
    return env_id, exp_param_type, exp_param_value, exp_metaname
    
# Root logging directory
logdir_root = pathlib.Path(PROJ_ROOT_PATH / "logging")
os.makedirs(logdir_root, exist_ok=True)

modeldir_root = pathlib.Path(PROJ_ROOT_PATH / "models")
os.makedirs(logdir_root, exist_ok=True)

def makeget_logging_dir(exp_name: str = None):

    try:
        exp_name is not None
    except ValueError:
        print("Experiment name has not been specified")

    env_id, exp_param_type, exp_param_value, exp_metaname = deconstruct_exp_name(exp_name)

    # Directory to save models
    models_dir = pathlib.Path(modeldir_root / exp_metaname / exp_name)
    os.makedirs(models_dir, exist_ok=True)
    # Directory to save all training statistics
    log_dir = pathlib.Path(logdir_root / exp_metaname / exp_name)
    os.makedirs(log_dir, exist_ok=True)
    # Directory to save gif animations
    gif_dir = pathlib.Path(log_dir / "gifs")
    os.makedirs(gif_dir, exist_ok=True)
    # Directory to save images
    image_dir = pathlib.Path(log_dir / "images" )
    os.makedirs(image_dir, exist_ok=True)
            

    return models_dir, log_dir, gif_dir, image_dir

def get_logging_dir(exp_name: str = None):

    try:
        exp_name is not None
    except ValueError:
        print("Experiment name has not been specified")

    # Directory to save models
    models_dir = pathlib.Path(modeldir_root / exp_name)
    os.makedirs(models_dir, exist_ok=True)
    # Directory to save all training statistics
    log_dir = pathlib.Path(logdir_root / exp_name)
    os.makedirs(log_dir, exist_ok=True)
    # Directory to save gif animations
    gif_dir = pathlib.Path(log_dir / "gifs")
    os.makedirs(gif_dir, exist_ok=True)
    # Directory to save images
    image_dir = pathlib.Path(log_dir / "images" )
    os.makedirs(image_dir, exist_ok=True)
            

    return models_dir, log_dir, gif_dir, image_dir






