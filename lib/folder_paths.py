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

# Root logging directory
logdir_root = pathlib.Path(PROJ_ROOT_PATH / "logging")
os.makedirs(logdir_root, exist_ok=True)

modeldir_root = pathlib.Path(PROJ_ROOT_PATH / "models")
os.makedirs(logdir_root, exist_ok=True)


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






