#must be run at offical folder https://github.com/state-spaces/mamba
# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn as nn
import pandas as pd
import datasets
import numpy as np
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba

import torch.optim.lr_scheduler as lr_scheduler

def load_and_split_dataset(context_len, horizon, load=True):
    # Load the dataset from the CSV file or the Hugging Face repository
    if load:
        dataset = datasets.load_dataset('Ammok/apple_stock_price_from_1980-2021')
        dataset = dataset['train']
        dataset = dataset.remove_columns('Date')
        context_windows = dataset.select(range(context_len)).to_pandas()
        tar_windows = dataset.select(range(context_len, context_len + horizon)).to_pandas()
    else:
        dataset = pd.read_csv("apple/apple_stock_price.csv")
        context_windows = dataset.iloc[:context_len, 1:]
        tar_windows = dataset.iloc[context_len:context_len + horizon, 1:]

    # Convert the context windows and target windows to tensors
    context_windows = context_windows.to_numpy(dtype=np.float32)
    tar_windows = tar_windows.to_numpy(dtype=np.float32)

    context_windows = torch.from_numpy(context_windows).float().cuda()
    tar_windows = torch.from_numpy(tar_windows).float().cuda()

    return context_windows, tar_windows


class Head(nn.Module):
    def __init__(self,dim, horizon, positive):
        super(Head, self).__init__()
        self.positive= positive
        self.linear = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.head = nn.Linear(dim, horizon, bias=False)

        self.head.apply(self.init_weights)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(F.silu(x))
        if self.positive ==True:
            x = F.sigmoid(x)
        x = self.head(x)
        return x
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)  # Initialize the weights using Xavier initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)


horizon= 20
d_model=256 # Model dimensions
positive = True

model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=d_model,
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).cuda()
torch.manual_seed(42)
head = Head(d_model, horizon, positive=positive).cuda()


optim_groups = [
    {'params': model.parameters()},
    {'params': head.parameters(), 'lr': 01e-07}  # Set a different learning rate for the head
]

optimizer = torch.optim.AdamW(optim_groups, weight_decay=0.1, lr=1e-07,  betas =(0.90, 0.95), eps=0.000010)

criterion = nn.CrossEntropyLoss()

context_window, target = load_and_split_dataset(256, horizon)

context_window = context_window.unsqueeze(0)
context_window = context_window.permute(0,2,1)

# Train the model
def train(model, head,  criterion, optimizer, x_batch, y_batch, num_epochs=1000):
    # Set the model to training mode
    model.train()
    
    train_losses = []  # List to store the training losses
    avg_loss = 0.0
    num_batches = 0

        # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        # Zero the gradients of the optimizer
        optimizer.zero_grad()
        model.zero_grad()
        output_tensor = model(x_batch)
        prediction = head(output_tensor)
        prediction = prediction.permute(0,2,1).squeeze(0)
        loss = criterion(prediction, y_batch)
        train_losses.append(loss.item())  # Append the loss for the current batch
        num_batches += 1
        if (epoch + 1) % 100 == 0:  # Check if 100 epochs have passed
            avg_loss = sum(train_losses) / num_batches  # Calculate the average loss
            with open('losses.txt', 'a') as file:  # Append to the text file
                file.write(f'Average Loss after {epoch + 1} epochs: {avg_loss}\n')
            train_losses = []  # Reset the list of training losses
            num_batches = 0  # Reset the number of batches

        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        torch.nn.utils.clip_grad_value_(head.parameters(), clip_value=1.0)

        optimizer.step()  # Update the model's weights



train(model,head,  criterion, optimizer, context_window, target)

