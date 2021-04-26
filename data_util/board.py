import numpy as np

import chess.pgn
import chess

chess_dict = {
    'p':  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'n':  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'b':  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'r':  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'q':  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'k':  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'P':  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'N':  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'B':  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'R':  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'Q':  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'K':  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    None: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}


def board_to_matrix(board: chess.Board) -> np.array:
    matrix = [
        [
          None for _ in range(8)
        ] for _ in range(8)
    ]
    for i, square in enumerate(chess.SQUARES):
        rank = i // 8
        file = i % 8
        symbol = board.piece_at(square).symbol() if board.piece_at(square) else None
        matrix[rank][file] = chess_dict[symbol]
    return np.array([
        np.array([
            np.array(file, dtype=np.short) for file in rank
        ]) for rank in matrix
    ])
