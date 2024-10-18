import argparse
import random
import wandb
import time
import tqdm
import torch
import numpy as np
from torch_geometric.data import DataLoader
from data_part2 import CustomGraphDataset
from torch_geometric.data import InMemoryDataset
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
import os
from models import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_layer', type=int, default=5)
parser.add_argument('--agg_hidden', type=int, default=32)
parser.add_argument('--fc_hidden', type=int, default=64)
parser.add_argument('--agg_method', type=str, default="avg")
parser.add_argument('--wandb', type=int, default=0)  # WandB flag (0 or 1)
args = parser.parse_args()


# Inits & Costants
class CustomGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        # Store the list of Data objects (graphs)
        self.data_list = data_list
        super(CustomGraphDataset, self).__init__(root=None)

    def len(self):
        # Return the number of graphs in the dataset
        return len(self.data_list)

    def get(self, idx):
        # Return the graph at index `idx`
        return self.data_list[idx]
    
if args.wandb:
    wandb.init(project='GDP_Project', config=args)

DATA_PATH = "data_part2"

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_data = torch.load(f'{DATA_PATH}/train.pt')
val_data = torch.load(f'{DATA_PATH}/val.pt')
test_data = torch.load(f'{DATA_PATH}/test.pt')

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GraphTransformer(n_layer=args.n_layer, agg_hidden=args.agg_hidden, fc_hidden=args.fc_hidden).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train_step(model, train_loader, optimizer, device):
    model.train()
    epoch_loss, correct = 0, 0
    num_train_examples = len(train_loader)

    total_samples = 0

    for graph in iter(train_loader):
        graph = graph.to(device)

        optimizer.zero_grad()
        prediction = model(graph)

        weight = torch.tensor([2, 1], dtype=torch.float32).to(device)
        loss = F.cross_entropy(prediction, graph.y, weight=weight)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predicted_class = torch.argmax(prediction)
        correct += predicted_class.eq(graph.y).sum().item()
        total_samples += len(graph.y)

    epoch_loss = epoch_loss / num_train_examples
    epoch_accuracy = correct / total_samples

    return epoch_loss, epoch_accuracy

def validation_step(model, train_loader, optimizer, device):
    epoch_loss, correct = 0, 0
    num_train_examples = len(train_loader)

    total_samples = 0

    for graph in iter(train_loader):
        graph = graph.to(device)

        with torch.no_grad():
            prediction = model(graph)

        weight = torch.tensor([2, 1], dtype=torch.float32).to(device)
        loss = F.cross_entropy(prediction, graph.y, weight=weight)

        epoch_loss += loss.item()
        predicted_class = torch.argmax(prediction)
        correct += predicted_class.eq(graph.y).sum().item()
        total_samples += len(graph.y)

    epoch_loss = epoch_loss / num_train_examples
    epoch_accuracy = correct / total_samples

    return epoch_loss, epoch_accuracy

epoch_progress = tqdm.trange(args.epochs, desc="Epochs")

train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in epoch_progress:
    train_loss, train_accuracy = train_step(model, train_loader, optimizer, device)
    val_loss, val_accuracy = validation_step(model, val_loader, optimizer, device)

    train_losses.append(train_loss)
    train_accs.append(train_accuracy)
    val_losses.append(val_loss)
    val_accs.append(val_accuracy)

    LOG = {
            'Epoch': epoch + 1,
            'Train/Epoch_Loss': train_loss,
            'Train/Epoch_Accuracy': train_accuracy,
            'Val/Epoch_Loss': val_loss,
            'Val/Epoch_Accuracy': val_accuracy
        }

    # Log metrics to WandB
    if args.wandb:
        wandb.log(LOG)

    epoch_progress.set_postfix(LOG)

if not args.wandb:
    if not os.path.exists('./results/Q2/graphs'):
        os.makedirs('./results/Q2/graphs')

    print("Saving the graphs")

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./results/Q2/graphs/loss.png')

    # Plotting the training and validation accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./results/Q2/graphs/accuracy.png')

    # # Making predictions on the test set
    # print("Making predictions on the test set")
    # test_predictions(model=model, test_loader=test_loader, device=device, file_name='results/Q2/predications.csv')
else:
    wandb.log({
        'Val/Min_Loss': min(val_losses),
        'Val/Avg_Accuracy': np.mean(val_accs)
    })
    wandb.finish()