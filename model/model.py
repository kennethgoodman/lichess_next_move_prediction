import logging

import torch
import numpy as np

INPUT_SIZE = 8 * 8 * 12 + 3  # 8 ranks, 8 files, 12 possible pieces + elo of current player + clocks
logger = logging.getLogger(__name__)


class LSTM(torch.nn.Module):
    def __init__(self, seq_len: int, input_size=INPUT_SIZE, hidden_layers_size=100, output_size=1):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_layers_size = hidden_layers_size
        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_layers_size
        )
        self.linear = torch.nn.Linear(hidden_layers_size, output_size)
        self.hidden_cell: tuple = (None, None)
        self.set_hidden_cell()

    def set_hidden_cell(self):
        self.hidden_cell = (
            torch.zeros(1, self.seq_len, self.hidden_layers_size),
            torch.zeros(1, self.seq_len, self.hidden_layers_size),
        )

    def forward(self, input_seq):
        input_seq = torch.from_numpy(input_seq)
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(
                1, len(input_seq), -1
            ).float(),
            self.hidden_cell
        )
        prediction = self.linear(
            lstm_out.view(
                len(input_seq), -1
            ).float()
        )
        return prediction[-1]


def training(model, epochs, train_input_seq, optimizer, loss_function):
    logger.info(f"starting training: epochs={epochs}, len(train_input_seq)={len(train_input_seq)}")
    for i in range(epochs):
        for seqs, labels in train_input_seq:
            optimizer.zero_grad()
            model.set_hidden_cell()  # should this be done in each epoch or each seq?
            y_pred = model(np.array(seqs))
            single_loss = loss_function(y_pred, torch.Tensor([labels]))
            single_loss.backward()
            optimizer.step()
        if i % 1 == 0 and i != 0:
            logger.info(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    logger.info(f'Done! epoch: {epochs:3} loss: {single_loss.item():10.8f}')
