import torch
import torchvision
import torch.nn as nn
from torchsummary import summary
import os
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet = torchvision.models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 3)

        self.register_buffer("iter", torch.tensor(0))

    def forward(self, x):
        self.iter += 1

        out = self.resnet(x)

        return out


def search_conv_layers(model_children):
    model_weights = []
    conv_layers = []

    for child in model_children:
        if type(child) == nn.Conv2d:
            model_weights.append(child.weight)
            conv_layers.append(child)

        elif type(child) == nn.Sequential:
            for chd in child:
                for c in chd.children():
                    if type(c) == nn.Conv2d:
                        model_weights.append(c.weight)
                        conv_layers.append(c)

    print(conv_layers)
    return model_weights, conv_layers


def visualize_filter(model, path="./filter/initial"):
    if not os.path.exists(path):
        os.makedirs(path)

    # t_layer = [i for i, (name, layer) in enumerate(
    #     model.named_modules()) if isinstance(layer, nn.Conv2d)]
    model_children = list(model.children())[0].children()
    model_weights, conv_layers = search_conv_layers(model_children)

    for layer_num, w in enumerate(model_weights):
        w = w.data.cpu()
        print(w.shape, w[0])
        if w.shape[1] not in [1, 3]:
            w_ = torch.FloatTensor([w[i][0].unsqueeze(0).numpy()
                                    for i in range(len(w))])

            w = w_

        # weight renormalization
        min_w = torch.min(w)
        w1 = (-1 / (2*min_w)) * w + 0.5

        grid_size = len(w1)
        x_grid = [w1[i] for i in range(grid_size)]
        x_grid = torchvision.utils.save_image(
            x_grid, f"{path}/{layer_num}th_layer.png", nrow=8, padding=1)


if __name__ == "__main__":
    model = Model()

    visualize_filter(model)

    summary(model, input_size=(3, 224, 224))
