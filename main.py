import logging
import logging.config
import os
import sys
import time

sys.path.append(os.path.join('./', 'lichess_downloader_api'))

import torch
import numpy as np

import lichess_downloader_api.filters.filter_utils as f
import lichess_downloader_api.filters.game_filters as gf

from data_util.board import board_to_matrix
from model.model import LSTM, training
from data_util.train_test import get_training_test_data
from data_util.pickle_manager import pickle_dump, pickle_read


def init_logger():
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default_handler': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': os.path.join('logs', 'application.log'),
                'encoding': 'utf8'
            },
            'default_stdout': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',  # Default is stderr
            },
        },
        'loggers': {
            '': {
                'handlers': ['default_handler', 'default_stdout'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
    logging.config.dictConfig(logging_config)


def get_saved_data(num_examples: int):
    # TODO, clean this function up
    def redo_scores(var):
        for i in range(len(var)):
            # TODO, assuming that highest negative example got 0.1, maybe store previous POSITIVE_EXAMPLE_SCORE
            if var[i][1] > .1:
                var[i][1] = POSITIVE_EXAMPLE_SCORE
            else:
                var[i][1] = 1 - POSITIVE_EXAMPLE_SCORE
        return var

    # num examples * 2 * seq len * (8 * 8 * 12 + 3)
    training_examples = redo_scores(pickle_read('training_examples'))
    training_boards = pickle_read('training_boards')
    test_examples = redo_scores(pickle_read('test_examples'))
    test_boards = pickle_read('test_boards')
    total_examples = len(training_examples) + len(test_examples)
    if num_examples == -1:
        num_examples = float('inf')
    total_percent_wanted = 1.0 * num_examples / total_examples
    if total_percent_wanted != float('inf') and total_percent_wanted > 1:
        logger.warning(f"wanted {num_examples}, but there are only {total_examples}")
    total_percent_wanted = min(total_percent_wanted, 1.0)  # if we want more than we have
    total_training = int(total_percent_wanted * len(training_examples))
    total_test = int(total_percent_wanted * len(test_examples))
    training_examples = training_examples[:total_training]
    training_boards = training_boards[:total_training]
    test_examples = test_examples[:total_test]
    test_boards = test_boards[:total_test]
    return training_examples, training_boards, test_examples, test_boards


def get_data():
    if SHOULD_RECREATE_DATA:
        training_examples, training_boards, test_examples, test_boards = get_training_test_data(
            year=YEAR,
            month=MONTH,
            percent_training=PERCENT_TRAINING,
            num_positive_examples_needed=NUM_EXAMPLES,
            seq_len=SEQ_LEN,  # 10 half-moves per sequence
            num_random_moves=NUM_RANDOM_MOVES,
            positive_example_score=POSITIVE_EXAMPLE_SCORE,
            filter_f=FILTER_F,
        )
        logger.info("pickling data")
        pickle_dump('training_examples', training_examples)
        pickle_dump('training_boards', training_boards)
        pickle_dump('test_examples', test_examples)
        pickle_dump('test_boards', test_boards)
        logger.info("pickled data")
    else:
        training_examples, training_boards, test_examples, test_boards = get_saved_data(NUM_EXAMPLES)
    return training_examples, training_boards, test_examples, test_boards


def main_with_saving():
    training_examples, training_boards, test_examples, test_boards = get_data()
    model = LSTM(seq_len=10)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    training(
        model,
        EPOCHS,
        training_examples,
        optimizer,
        loss_function
    )
    fn = f'{int(time.time())}__te_{len(training_examples)}__sl_{SEQ_LEN}__nrm_{NUM_RANDOM_MOVES}' \
         f'__epochs_{EPOCHS}__'
    logger.info(f"saving model to {fn}")
    torch.save(model.state_dict(), os.path.join('model_state_dict', fn), _use_new_zipfile_serialization=False)


def main_with_saved():
    model = LSTM(seq_len=10)
    fn = max(map(lambda x: int(x.split("__")[0]), os.listdir('model_state_dict')))
    model.load_state_dict(torch.load(os.path.join('model_state_dict', f'{fn}')))
    model.eval()
    torch.save(model.state_dict(), os.path.join('model_state_dict', f'{time.time()}'),
               _use_new_zipfile_serialization=False)
    training_examples, training_boards, test_examples, test_boards = get_data()
    for (test_example, test_board) in zip(test_examples, test_boards):
        example, label = test_example
        if label == 0:
            continue
        current_move = test_board.pop()
        print(test_board)
        print(f"correct move: {current_move}")
        for move in test_board.legal_moves:
            test_board.push(move)
            m = board_to_matrix(test_board)
            test_board.pop()  # undo the move
            example[-1][:8 * 8 * 12] = np.array(m).reshape(8 * 8 * 12)
            print(move, model(example)[-1])
        break


if __name__ == '__main__':
    init_logger()
    logger = logging.getLogger(__name__)
    SHOULD_RECREATE_DATA = True
    PERCENT_TRAINING = 0.95
    NUM_EXAMPLES = -1
    POSITIVE_EXAMPLE_SCORE = 1
    SEQ_LEN = 10
    NUM_RANDOM_MOVES = -1
    YEAR = 2017
    MONTH = 4
    EPOCHS = 4
    FILTER_F = f.OR(
        f.AND(
            # 1950+ with any time control
            gf.get_filter_by_avg_rating(min_rating=1950),
        ),
    )
    main_with_saving()
