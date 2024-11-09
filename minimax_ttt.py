import math

# Initialize the board with empty spaces
board = [" " for _ in range(9)]

# Function to print the board in a 3x3 format
def print_board(board):
    for row in range(3):
        print(" | ".join(board[row * 3:(row + 1) * 3]))
        if row < 2:
            print("---------")

# Check if there are any empty cells left
def is_draw(board):
    return " " not in board

# Function to check if a player has won
def evaluate(board):
    # Winning combinations
    win_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]

    for pos in win_positions:
        if board[pos[0]] == board[pos[1]] == board[pos[2]] != " ":
            return 1 if board[pos[0]] == "X" else -1

    return 0  # No winner

# Minimax algorithm with recursion
def minimax(board, depth, is_maximizing):
    score = evaluate(board)

    # Base cases for terminal states
    if score == 1:  # AI wins
        return score
    if score == -1:  # Player wins
        return score
    if is_draw(board):  # Draw
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == " ":
                board[i] = "X"  # AI plays
                best_score = max(best_score, minimax(board, depth + 1, False))
                board[i] = " "  # Undo move
        return best_score

    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == " ":
                board[i] = "O"  # Player plays
                best_score = min(best_score, minimax(board, depth + 1, True))
                board[i] = " "  # Undo move
        return best_score

# Use minimax to find the best move for AI
def find_best_move(board):
    best_val = -math.inf
    best_move = None
    for i in range(9):
        if board[i] == " ":
            board[i] = "X"
            move_val = minimax(board, 0, False)
            board[i] = " "
            if move_val > best_val:
                best_move = i
                best_val = move_val
    return best_move

# Main game loop
def main():
    print("Welcome to Tic-Tac-Toe!")
    print_board(board)

    while True:
        # Player move
        player_move = int(input("Enter your move (0-8): "))
        if board[player_move] != " ":
            print("Invalid move. Try again.")
            continue

        board[player_move] = "O"
        print_board(board)

        # Check if player wins
        if evaluate(board) == -1:
            print("Congratulations! You win!")
            break
        elif is_draw(board):
            print("It's a draw!")
            break

        # AI move
        ai_move = find_best_move(board)
        board[ai_move] = "X"
        print("AI plays:")
        print_board(board)

        # Check if AI wins
        if evaluate(board) == 1:
            print("AI wins! Better luck next time.")
            break
        elif is_draw(board):
            print("It's a draw!")
            break

if __name__ == "__main__":
    main()
