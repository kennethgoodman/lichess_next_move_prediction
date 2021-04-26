from typing import Tuple, List, Callable, Union, Generator
import random
import logging

import numpy as np
import chess

from lichess_downloader_api.data_manager.manager import Manager
from lichess_downloader_api.game_fetcher.get_with_filter import get_n_games_with_filter_gen
from lichess_downloader_api.models.games import Games

from data_util.game import get_n_examples

logger = logging.getLogger(__name__)


def get_training_examples(games: Union[Games, Generator], num_positive_examples_needed: int, seq_len: int,
                          num_random_moves: int,
                          positive_example_score: float) -> Tuple[np.array, List[chess.Board]]:
    # TODO, should games be segregated for training/testing?
    # TODO, should we convert this to a generator as well? The main problem is with
    #           randomizing positive and negative examples that come from the same game id
    #           completely random probably isn't necessary, one option is to randomly store some in memory and
    #           offload those random ones later
    examples, labels, boards = [], [], []
    for (game_id, example, label, board) in get_n_examples(games=games,
                                                           num_positive_examples_needed=num_positive_examples_needed,
                                                           seq_len=seq_len, num_random_moves=num_random_moves,
                                                           positive_example_score=positive_example_score):
        examples.append(example)
        labels.append(label)
        boards.append(board)
    examples = np.array(examples, dtype=np.short)  # TODO, is this needed?
    training = list(zip(examples, labels))
    random.seed(1)
    random.shuffle(training)
    random.seed(1)
    random.shuffle(boards)
    return training, boards


def get_training_test_data(year: int, month: int, percent_training: float, num_positive_examples_needed: int,
                           seq_len: int,
                           num_random_moves: int, positive_example_score: float, filter_f: Callable):
    # TODO, convert to generator to keep memory usage low (and possible parallelism high)
    #       biggest problem is randomization of training examples
    data_manager = Manager(year=year, month=month)
    # TODO, can probably figure out num_games_needed in a better way,
    #       currently average of 2500 games is 37 moves, so 37 - seq_len + 1
    #       when num_positive_examples_needed is -1 we get all possible games
    logger.info("getting games")
    games = get_n_games_with_filter_gen(data_manager, num_games_needed=num_positive_examples_needed // (37 - seq_len + 1),
                                        filter_f=filter_f)
    examples, boards = get_training_examples(games, num_positive_examples_needed=num_positive_examples_needed,
                                             seq_len=seq_len, num_random_moves=num_random_moves,
                                             positive_example_score=positive_example_score)
    num_training = int(len(examples) * percent_training)
    training_examples = examples[:num_training]
    training_boards = boards[:num_training]
    test_examples = examples[num_training:]
    test_boards = boards[num_training:]
    return training_examples, training_boards, test_examples, test_boards
