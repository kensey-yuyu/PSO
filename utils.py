import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime


def check_log_dir():
    """ Check whether log directory exits.

    Returns:
        bool: whether log directory exits.
    """

    # Check directory.
    if not os.path.isdir("./log"):
        print("No log directory. Create log directory...")
        try:
            os.mkdir("./log")
        except OSError:
            exit("Error: Could not make log directory.")
    return


def log(args, global_best, global_history, particles_history):
    """ Output log.

    Args:
        args (): _description_
        global_best (_type_): _description_
        history (_type_): _description_
    """

    # Try to create directory.
    time = datetime.now()
    path = f"./log/{time}"
    try:
        os.mkdir(path)
    except OSError:
        exit("Error: Could not make log directory.")

    # Write log.
    with open(f"{path}/log.txt", "w", encoding="utf-8") as file:
        for key, value in args.items():
            file.write(str(f"{key}: {value}\n"))

    # Save history.
    history = {
        "global": global_history,
        "particles": particles_history
    }
    with open(f"{path}/history.pickle", mode="wb") as file:
        pickle.dump(history, file)

    # Plot graph of global best.
    plot_global_best(global_best, global_history, path)
    return


def plot_global_best(global_best, history, path):
    """ Plot global best.

    Args:
        global_best (float): Global best.
        history (dict): History of global best and global best position.
        path (str): Save path
    """

    # Plot graph of global best.
    plt.figure()
    plt.title(f"Iteration and Global best\n Global best: {global_best}")
    plt.xlabel("Iteration")
    plt.ylabel("Global best")
    plt.plot(history["global_best"])
    plt.savefig(f"{path}/result.png")
    return
