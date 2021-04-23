from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

from src.base.base_model import BaseModel


class RnnModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Assign parameters
        self.filter_base = config.network.filter_base
        self.kernel_size = config.network.kernel_size
        self.max_pooling = config.network.max_pooling
        self.num_blocks = config.network.num_blocks
        self.num_channels = len(config.data_loader.modalities) + 1
        self.num_classes = config.data_loader.num_classes
        self.rnn_bidirectional = config.network.rnn_bidirectional
        self.rnn_num_layers = config.network.rnn_num_layers
        self.rnn_num_units = config.network.rnn_num_units if config.network.rnn_num_units is not None else 4 * \
            self.filter_base * (2 ** (self.num_blocks - 1))

        # Create network
        if self.num_channels != 1:
            self.mixing_block = nn.Sequential(OrderedDict([
                ('mix_conv', nn.Conv2d(1, self.num_channels, (self.num_channels, 1))),
                ('mix_batchnorm', nn.BatchNorm2d(self.num_channels)),
                ('mix_relu', nn.ReLU())
            ]))

        # Define shortcut
        self.shortcuts = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('shortcut_conv_{}'.format(k), nn.Conv2d(
                    in_channels=self.num_channels if k == 0 else 4 *
                    self.filter_base * (2 ** (k - 1)),
                    out_channels=4 * self.filter_base * (2 ** k),
                    kernel_size=(1, 1)))
            ])) for k in range(self.num_blocks)
        ])

        # Define basic block structure
        self.blocks = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv_{}_1".format(k), nn.Conv2d(
                    in_channels=self.num_channels if k == 0 else 4 * self.filter_base *
                    (2 ** (k - 1)),
                    out_channels=self.filter_base * (2 ** k),
                    kernel_size=(1, 1))),
                # ("padding_{}".format(k), nn.ConstantPad2d([1, 1, 0, 0], 0)),
                ("batchnorm_{}_1".format(k), nn.BatchNorm2d(
                    self.filter_base * (2 ** k))),
                ("relu_{}_1".format(k), nn.ReLU()),
                ("conv_{}_2".format(k), nn.Conv2d(
                    in_channels=self.filter_base * (2 ** k),
                    out_channels=self.filter_base * (2 ** k),
                    kernel_size=(1, self.kernel_size),
                    padding=(0, self.kernel_size // 2))),
                ("batchnorm_{}_2".format(k), nn.BatchNorm2d(
                    self.filter_base * (2 ** k))),
                ("relu_{}_2".format(k), nn.ReLU()),
                ("conv_{}_3".format(k), nn.Conv2d(
                    in_channels=self.filter_base * (2 ** k),
                    out_channels=4 * self.filter_base * (2 ** k),
                    kernel_size=(1, 1))),
                ("batchnorm_{}_3".format(k), nn.BatchNorm2d(
                    4 * self.filter_base * (2 ** k)))
            ])) for k in range(self.num_blocks)
        ])
        self.maxpool = nn.MaxPool2d(kernel_size=(1, self.max_pooling))
        self.relu = nn.ReLU()

        if self.rnn_num_units == 0:

            # Classification (outputs only logits)
            self.classification = nn.Conv1d(
                in_channels=4 * self.filter_base * (2 ** (self.num_blocks - 1)),
                out_channels=self.num_classes,
                kernel_size=1)
        else:

            # Temporal processing
            self.temporal_block = nn.GRU(
                input_size=4 * self.filter_base * (2 ** (self.num_blocks - 1)),
                hidden_size=self.rnn_num_units, num_layers=self.rnn_num_layers,
                batch_first=True, dropout=0, bidirectional=self.rnn_bidirectional)
            self.temporal_block.flatten_parameters()

            # Classification (outputs only logits)
            self.classification = nn.Conv1d(
                in_channels=(1 + self.rnn_bidirectional) * self.rnn_num_units,
                out_channels=self.num_classes,
                kernel_size=1)
        # self.classification = nn.Sequential(OrderedDict([
        #     ('cls_conv', nn.Conv1d(
        #         in_channels=(1 + self.rnn_bidirectional) *
        #         self.rnn_num_units,
        #         out_channels=self.num_classes,
        #         kernel_size=1)),
        #     ('softmax', nn.Softmax(dim=1))
        # ]))

    def forward(self, x):

        # if self.temporal_block:
        #     self.temporal_block.flatten_parameters()

        z = self.mixing_block(x)
        for block, shortcut in zip(self.blocks, self.shortcuts):
            y = shortcut(z)
            z = block(z)
            z += y
            z = self.relu(z)
            z = self.maxpool(z)
        # print(z.shape)
        # RNN part
        # print(self.rnn_num_units)
        if self.rnn_num_units == 0:
            z = self.classification(z.squeeze(2))
            # print('Hej! ' + str(z.shape))
        else:
            z = self.temporal_block(z.squeeze(2).transpose(1, 2))
            # print(z[0].shape)
            z = self.classification(z[0].transpose(1, 2))

        return z


if __name__ == '__main__':

    import numpy as np
    import torch
    from torch.autograd import Variable

    from src.utils.config import process_config

    config = process_config('./src/configs/test-rnn_model.yaml')
    print(config.network.rnn_num_units)
#     config.network.rnn_num_units = 0
    model = RnnModel(config)
    n_channels = model.num_channels
    length = config.data_loader.segment_length * 128
    x = np.random.randn(config.data_loader.batch_size.train, 1,
                        n_channels, length)
    x = Variable(torch.from_numpy(x).float())
    print(model)
    z = model(x)
    print(z.shape)
