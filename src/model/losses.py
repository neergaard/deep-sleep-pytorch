import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


def nll_loss(output, target):
    return F.nll_loss(output, target.long())


def temporal_crossentropy_loss(output, target):
    return F.cross_entropy(output, target, reduction='mean')


if __name__ == '__main__':

    import numpy as np
    import torch
    from torch.autograd import Variable

    outputs = np.random.randint(0, 4, size=(32, 5, 120))
    outputs = Variable(torch.from_numpy(outputs).float().cuda())

    from src.utils.config import process_config

    config = process_config('./src/configs/test-rnn_model.yaml')
    model = RnnModel(config).cuda()
    n_channels = model.num_channels
    length = config.data_loader.segment_length * 128
    x = np.random.randn(config.data_loader.batch_size.train, 1,
                        n_channels, length)
    x = Variable(torch.from_numpy(x).float()).cuda()
    print(model)
    z = model(x)
    print(z.shape)
