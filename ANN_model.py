import pandas as pd
# from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from torch.utils.data import TensorDataset, DataLoader, random_split

# Load data
df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
print(y)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
# print(X_tensor)
y_tensor = torch.LongTensor(y)
# print(y_tensor)
# Create a TensorDataset
dataset = TensorDataset(X_tensor, y_tensor)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
print(train_size)
test_size = len(dataset) - train_size
print(test_size)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Architecture of the ANN model which contains the two fully connected layers as it is linear in nature from initial stage and we do not want to use the flattening layer for ANN's.
class ANN_Model(nn.Module):
    def __init__(self, input_features=8, hidden1=20, hidden2=20, out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_features)

    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x

# Initialize model, loss function, and optimizer
torch.manual_seed(20)
model = ANN_Model()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 500
final_losses = []

# Ensure proper device usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the profiler with TensorBoard trace handler
profiler_path = r'C:\\Users\\Radhika\\Downloads\\myPytorch\\profiler'
with profile(
    activities=[ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=tensorboard_trace_handler(profiler_path),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for epoch in range(epochs):
        epoch += 1
        model.train()
        for batch in train_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            with record_function("model_training"):
                y_pred = model.forward(X_batch)
                loss = loss_fn(y_pred, y_batch)

                final_losses.append(loss.item())

                if epoch % 10 == 1:
                   print("Epoch number: {} and loss: {}".format(epoch, loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        prof.step()

# Save the model
torch.save(model.state_dict(), 'diabetes.pt')  # Save only state_dict for best practice

# To print the profiler summary
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
