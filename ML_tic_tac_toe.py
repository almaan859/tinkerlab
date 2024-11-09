import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
data = pd.read_csv("tic_tac_toe_dataset.csv")

# Convert the board_state (string representation) to an array
data["board_state"] = data["board_state"].apply(lambda x: np.fromstring(x[1:-1], sep=','))
X = np.array(data["board_state"].tolist())  # Board states
y = data["move"].values                     # Optimal moves

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the neural network model
class TicTacToeModel(nn.Module):
    def __init__(self):
        super(TicTacToeModel, self).__init__()
        self.fc1 = nn.Linear(9, 128)   # Input layer
        self.fc2 = nn.Linear(128, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, 9)    # Output layer (predicts best move for 9 positions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = TicTacToeModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')

# Save the model
torch.save(model.state_dict(), "tic_tac_toe_model.pth")

# Function to predict the best move
def get_best_move(board_state):
    with torch.no_grad():
        board_tensor = torch.tensor(board_state, dtype=torch.float32).unsqueeze(0)
        predictions = model(board_tensor)
        best_move = predictions.argmax().item()
    return best_move

# Test function in gameplay
sample_board = np.array([1, 0, -1, 1, -1, 1, 0, -1, 0])  # Example board state
best_move = get_best_move(sample_board)
print(f"The AI suggests move position: {best_move}")
