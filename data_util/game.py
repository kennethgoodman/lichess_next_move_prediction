import queue
from typing import List, Tuple, Union, Generator
import random
import copy

import chess.pgn
import chess
import numpy as np

from lichess_downloader_api.python_chess_utils.header_utils import get_white_elo, get_black_elo
from lichess_downloader_api.models.games import Games

from data_util.board import board_to_matrix


def seq_data_to_input_data(board_position: chess.Board, game: chess.pgn.Game) -> np.array:
    matrix = board_to_matrix(board_position)
    reshaped: np.array = np.array(matrix).reshape(8 * 8 * 12)
    white_elo = get_white_elo(game)
    black_elo = get_black_elo(game)
    clock_white = 0  # TODO: add white clock, chess.pgn is ignoring it
    clock_black = 0  # TODO: add block clock, chess.pgn is ignoring it
    elo = white_elo if board_position.turn == chess.WHITE else black_elo
    elo /= 3300  # divide by top possible elo to get range 0 -> 1
    return np.append(
        reshaped,
        np.array([
            elo, clock_white, clock_black
        ], dtype=np.short)
    )


def sequence_gen(seq_len: int, game: chess.pgn.Game) -> Tuple[np.array, chess.Board]:
    seq = queue.Queue(maxsize=seq_len)
    if game.is_end() or len(game.variations) == 0:
        return
    cur_board = game.variations[0]
    while True:  # TODO: rework if-break into while statement
        if seq.full():  # only yield full sequences
            seq.get()  # ignore output, just dump it
            seq.put(seq_data_to_input_data(cur_board.board(), game))
            yield np.array(seq.queue), cur_board.board()
        else:
            seq.put(seq_data_to_input_data(cur_board.board(), game))
        if game.is_end() or len(cur_board.variations) == 0:
            break
        cur_board = cur_board.variations[0]


def get_n_examples(games: Union[Games, Generator], num_positive_examples_needed: int, seq_len: int, num_random_moves: int,
                   positive_example_score: float) -> Tuple[int, np.array, List[float], List[chess.Board]]:
    # TODO, clean up this function, some refactoring to clean it up can be done
    sum_negative_example_scores = 1 - positive_example_score
    num_positive_examples = 0
    for (game_id, game) in games:
        for example_board in sequence_gen(seq_len, game):
            example: np.array = example_board[0]
            board: chess.Board = copy.deepcopy(example_board[1])
            # Positive Example
            yield game_id, copy.deepcopy(example), positive_example_score, copy.deepcopy(example_board[1])
            num_positive_examples += 1

            # Negative Examples
            last_move = board.pop()  # undo the last move so we can play a fake one
            # get all possible moves, except the one last played
            legal_moves = [move for move in board.legal_moves if move != last_move]
            if len(legal_moves) == 0:  # if only one legal move (the one played), this is a bad example
                continue  # skip this example
            random_moves_to_choose = num_random_moves
            if num_random_moves == -1:
                random_moves_to_choose = len(legal_moves) - 1  # all but the actual moved played
            random.shuffle(legal_moves)  # shuffle randomly then just pick the first elements
            label = sum_negative_example_scores / len(legal_moves)
            for i in range(random_moves_to_choose):
                random_move = legal_moves[i]  # because shuffled, we can just pick the elements by index
                board.push(random_move)  # play the fake move
                example[-1] = seq_data_to_input_data(board, game)  # set the last move to be the random move
                yield game_id, copy.deepcopy(example), label, copy.deepcopy(board)
                board.pop()  # undo that move
            if num_positive_examples == num_positive_examples_needed:
                return
    return
