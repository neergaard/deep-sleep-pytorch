import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
import math


from src.base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics,
                                      optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler:
            self.lr_scheduler = self.lr_scheduler(self.optimizer, **self.config.lr_scheduler.args)
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            # self.writer.add_scalar('{}'.format(
            #     metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """

        ## Find LR
        # def find_lr(data_loader, model, optimizer, loss, init_value = 1e-8, final_value=10., beta = 0.98):
        #     batch_loader = data_loader
        #     pbar = tqdm(enumerate(batch_loader), total=len(batch_loader))
        #     num = len(batch_loader)-1
        #     mult = (final_value / init_value) ** (1/num)
        #     lr = init_value
        #     optimizer.param_groups[0]['lr'] = lr
        #     avg_loss = 0.
        #     best_loss = 0.
        #     losses = []
        #     log_lrs = []
        #     model.train()

        #     for batch_num, (data, target) in pbar:
        #         #As before, get the loss for this mini-batch of inputs/outputs
        #         data, target = data.to(self.device), target.to(self.device)
        #         optimizer.zero_grad()
        #         outputs = model(data)
        #         loss_val = loss(outputs, target)
        #         #Compute the smoothed loss
        #         avg_loss = beta * avg_loss + (1-beta) *loss_val.item()
        #         smoothed_loss = avg_loss / (1 - beta**(batch_num+1))
        #         #Stop if the loss is exploding
        #         if (batch_num+1) > 1 and smoothed_loss > 4 * best_loss:
        #             return log_lrs, losses
        #         #Record the best loss
        #         if smoothed_loss < best_loss or (batch_num+1)==1:
        #             best_loss = smoothed_loss
        #         #Store the values
        #         losses.append(smoothed_loss)
        #         log_lrs.append(math.log10(lr))
        #         #Do the SGD step
        #         loss_val.backward()
        #         optimizer.step()

        #         #Update the lr for the next step
        #         lr *= mult
        #         ptvsd.break_into_debugger()
        #         optimizer.param_groups[0]['lr'] = lr
        #     ptvsd.break_into_debugger()
        #     plt.plot(log_lrs[10:-5], losses[10:-5])
        #     plt.savefig('./test_bs256_ngpu4.png')
        # find_lr(self.data_loader, self.model, self.optimizer, self.loss)


        self.model.train()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        bar_train = tqdm(self.data_loader, total=len(self.data_loader))
        bar_train.set_description('[ TRAIN ] | Loss: inf')
        for batch_idx, out in enumerate(bar_train):
            data, target = out['data'].to(self.device), out['target'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            # self.writer.set_step(
            #     (epoch - 1) * len(self.data_loader) + batch_idx)
            # self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)
            bar_train.set_description('[ TRAIN ] | Loss: {:.4f}'.format(loss.item()))
            # if self.verbosity >= 2 and batch_idx % self.log_step == 0:
            #     self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
            #         epoch,
            #         batch_idx * self.data_loader.batch_size,
            #         len(self.data_loader),
            #         100.0 * batch_idx / len(self.data_loader),
            #         loss.item()))
            #     self.writer.add_image('input', make_grid(
            #         data.cpu(), nrow=8, normalize=True))
        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_log['val_loss'])

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            bar_eval = tqdm(self.valid_data_loader, total=len(self.valid_data_loader))
            bar_eval.set_description('[ EVAL ] | Loss: inf')
            for batch_idx, out in enumerate(bar_eval):
                data, target = out['data'].to(self.device), out['target'].to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                # self.writer.set_step(
                #     (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                bar_eval.set_description('[ EVAL ] | Loss: {:.4f}'.format(loss.item()))
                # self.writer.add_image('input', make_grid(
                #     data.cpu(), nrow=8, normalize=True))

        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
