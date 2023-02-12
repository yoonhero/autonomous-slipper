from utils import set_seed
from dataset import DrivingDataset
from model import Model
import config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./runs/resnet18")


set_seed(19283)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = config.LEARNING_RATE
MOMENTUM = config.MOMENTUM
NB_EPOCH = config.EPOCHS
transforms = config.transforms
TRAIN_BATCH = config.TRAIN_BATCH
VALIDATION_BATCH = config.VALIDATION_BATCH
TEST_BATCH = config.TEST_BATCH

dataset = DrivingDataset("training_data.csv", transforms)
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
validation_size = int(dataset_size * 0.1)
test_size = dataset_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(
    dataset, [train_size, validation_size, test_size])

print(f"Training Data Size: {train_size}")
print(f"Validation Data Size: {validation_size}")
print(f"Testing Data Size: {test_size}")

train_dataloader = DataLoader(
    train_dataset, batch_size=TRAIN_BATCH, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=VALIDATION_BATCH, shuffle=True, drop_last=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=TEST_BATCH, shuffle=True, drop_last=True)

model = Model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


def train_loop(model, optimizer, criterion, dataloader, epoch):
    model.train()

    cost = 0.0
    train_bar = tqdm(dataloader)
    train_bar.set_description(f"EPOCH: {epoch}")
    for idx, (x, y) in enumerate(train_bar):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = criterion(y, pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss.item()

        train_bar.set_postfix(loss=cost/(idx+1))

    cost = cost / len(dataloader)
    return cost


def validation_loop(model, criterion, dataloader):
    cost = 0.0
    acc = 0.0
    with torch.no_grad():
        model.eval()
        print("Validation Start!")

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(y, pred)

            cost += loss.item()

            acc += (torch.argmax(pred, dim=1).unsqueeze(1) == y).float().sum()

    accuracy = 100 * acc / len(validation_dataset)
    cost = cost / len(dataloader)

    return cost, accuracy


for epoch in range(NB_EPOCH):
    train_loss = train_loop(model, optimizer, criterion,
                            train_dataloader, epoch)
    validation_loss, acc = validation_loop(model, validation_dataloader)

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/validation", validation_loss, epoch)
    writer.add_scaler("Accuracy/validation", acc, epoch)

    print(
        f"EPOCH: {epoch} | Train Loss: {train_loss} | Validation Loss: {validation_loss} | Accuracy: {acc}")


writer.close()
