#must be run at offical folder https://github.com/state-spaces/mamba
# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn as nn
import pandas as pd
import datasets
import numpy as np

from mamba_ssm.modules.mamba_simple import Mamba


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
    #print(dataset)
    # Convert the context windows and target windows to NumPy arrays
    context_windows = context_windows.to_numpy(dtype=np.float32)
    tar_windows = tar_windows.to_numpy(dtype=np.float32)

    context_windows = torch.from_numpy(context_windows).float().cuda()
    tar_windows = torch.from_numpy(tar_windows).float().cuda()

    return context_windows, tar_windows



model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=256 , # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).cuda()
torch.manual_seed(42)

context_window, target = load_and_split_dataset(256, 1)


context_window = context_window.unsqueeze(0)
context_window = context_window.permute(0,2,1)

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1, lr=0.0000012,  betas =(0.95, 0.999), eps=0.0001)
criterion = nn.CrossEntropyLoss()

# Train the model
def train(model, criterion, optimizer, x_batch, y_batch, num_epochs=1000):
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
        output_tensor= output_tensor[:,:, -1]

        loss = criterion(output_tensor, y_batch)
        #print(loss)
        train_losses.append(loss.item())  # Append the loss for the current batch
        num_batches += 1

        if (epoch + 1) % 100 == 0:  # Check if 100 epochs have passed
            avg_loss = sum(train_losses) / num_batches  # Calculate the average loss
            with open('losses.txt', 'a') as file:  # Append to the text file
                file.write(f'Average Loss after {epoch + 1} epochs: {avg_loss}\n')
            train_losses = []  # Reset the list of training losses
            num_batches = 0  # Reset the number of batches
        loss.backward()
        # Update the weights
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=4.0)
        optimizer.step()



train(model, criterion, optimizer, context_window, target)
