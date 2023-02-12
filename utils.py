import shutil
import os
import torch
import random
import numpy as np

import config
from model import Model


def compress_folder(output_filename, input_folder_name):
    shutil.make_archive(output_filename, 'zip', input_folder_name)


def save_model(model, optimizer, loss, saved_dir):
    os.makedirs(saved_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, f"{saved_dir}/checkpoint_{model.iter}.pt")


def load_model(PATH):
    model = Model()
    optimizer = torch.optim.SGD(
        model.parameters(), config.LEARNING_RATE, config.MOMENTUM)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


if __name__ == "__main__":
    compress_folder("training_data", "./training_data/raw")
