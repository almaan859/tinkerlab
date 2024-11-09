import torch
import torch.nn as nn
import numpy as np
import random  # For testing without the model

# Load the trained model (Commented for debugging purposes)
class TicTacToeModel(nn.Module):
    def __init__(self):
        super(TicTacToeModel, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Uncomment these lines if you want to use the model
# model = TicTacToeModel()
# model.load_state_dict(torch.load("tic_tac_toe_model.pth"))
# model.eval()

# Function to get the AI's best move (Using random move for debugging)
def get_best_move(board_state):
    # For debugging, replace model inference with random choice
    # board_tensor = torch.tensor(board_state, dtype=torch.float32).unsqueeze(0)
    # with torch.no_grad():
    #     predictions = model(board_tensor)
    # best_move = predictions.argmax().item()
    # return best_move
    available_moves = [i for i, v in enumerate(board_state) if v == 0]
    return random.choice(available_moves)

# Helper functions for game logic
def display_board(board):
    for i in range(3):
        row = "|".join(["X" if board[i*3 + j] == 1 else "O" if board[i*3 + j] == -1 else str(i*3 + j) for j in range(3)])
        print(row)
    print("\n")

def check_winner(board):
    winning_combinations = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    for (x, y, z) in winning_combinations:
        if board[x] == board[y] == board[z] != 0:
            return board[x]
    return 0

def check_draw(board):
    return all([spot != 0 for spot in board])

# Main game loop
def play_game():
    board = [0] * 9  # Initialize empty board (0 = empty, 1 = player, -1 = computer)
    
    print("Welcome to Tic-Tac-Toe! You are 'X' and the computer is 'O'.")
    display_board(board)
    
    while True:
        # Player's turn
        try:
            move = int(input("Enter the position (0-8) you want to play in: "))
            print(f"Player chose position {move}")
            while board[move] != 0:
                print("Invalid move! Position already taken.")
                move = int(input("Enter the position (0-8) you want to play in: "))
            board[move] = 1
            print("Board after player move:")
            display_board(board)
        except ValueError:
            print("Please enter a valid number between 0 and 8.")
            continue  # Restart the loop if invalid input is provided

        # Check for winner or draw
        if check_winner(board) == 1:
            print("Congratulations! You win!")
            break
        if check_draw(board):
            print("It's a draw!")
            break

        # Computer's turn
        board_state = np.array(board)
        print("Computer is making a move...")
        computer_move = get_best_move(board_state)
        while board[computer_move] != 0:
            computer_move = get_best_move(board_state)  # Re-check if position taken (just in case)
        board[computer_move] = -1
        print(f"Computer plays at position {computer_move}.")
        
        # Display board and check for winner or draw
        display_board(board)
        if check_winner(board) == -1:
            print("Computer wins! Better luck next time.")
            break
        if check_draw(board):
            print("It's a draw!")
            break

play_game()
