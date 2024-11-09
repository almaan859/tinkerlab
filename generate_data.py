import random
import numpy as np
import pandas as pd

def check_winner(board):
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                      (0, 3, 6), (1, 4, 7), (2, 5, 8),
                      (0, 4, 8), (2, 4, 6)]
    for a, b, c in win_conditions:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0 if 0 in board else "draw"

def generate_game_data():
    data = []
    for _ in range(10000):  # Number of games to simulate
        board = [0] * 9
        moves = []
        for turn in range(9):  # Max 9 moves per game
            player = 1 if turn % 2 == 0 else -1
            available_positions = [i for i, x in enumerate(board) if x == 0]
            move = random.choice(available_positions)
            board[move] = player
            moves.append((board[:], move, player))  # Record the move and board

            winner = check_winner(board)
            if winner:
                for state, move_pos, p in moves:
                    result = 1 if winner == p else (-1 if winner != "draw" else 0)
                    data.append((state, move_pos, result))
                break
    return pd.DataFrame(data, columns=["board_state", "move", "outcome"])

# Generate the data and save as CSV
df = generate_game_data()
df.to_csv("tic_tac_toe_dataset.csv", index=False)
