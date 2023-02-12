from torch.utils.data import Dataset
import torch
import pandas as pd
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


class DrivingDataset(Dataset):
    def __init__(self, csv="training_data.csv", transforms=T.ToTensor()):
        self.data = pd.read_csv(csv)
        self.length = len(self.data)
        self.transforms = transforms

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        target = self.data[idx]

        X = target["X"]
        X = Image.open(X).convert("RGB")
        width, height = X.size
        t_width = .6
        t_height = .4
        left = width/2-width*t_width/2
        right = width/2+width*t_width/2
        top = height - height*t_height
        bottom = height - 100
        roi_region = (left, top, right, bottom)
        X = X.crop(roi_region)
        X = self.transforms(X)

        Y = int(target["Y"])
        # Y = torch.eye(3, dtype=torch.float32)[int(Y)]

        return X, Y


if __name__ == "__main__":
    transforms = T.Compose([
        T.Resize(224, 0),
        T.ToTensor()
    ])
    dataset = DrivingDataset("training_data.csv", )

    x, y = dataset[0]

    plt.imshow(x)
    print(y)
    plt.show()
