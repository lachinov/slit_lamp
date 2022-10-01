import os
import shutil
import pickle
import torch
from torch.optim import *
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import time
from . import loader_helper
from . import metrics
import gc


class TrainingState(object):
    def __init__(self):
        self.epoch = 0
        self.train_metric = dict()
        self.val_metric = dict()
        # number of processed batches
        self.global_step = 0
        self.best_val = 0
        self.optimizer_state = None
        self.cuda = True


class Trainer(object):
    def __init__(self, name, models_root, model=None, rewrite=False, connect_tb=True):

        self.model = model

        assert (isinstance(self.model, (list, tuple, torch.nn.Module)) or self.model is None)

        self.name = name
        self.models_root = models_root
        self.model_path = os.path.join(models_root, self.name)
        self.logs_path = os.path.join(self.model_path, 'logs')

        self.state = TrainingState()
        self.resume_training = False

        if os.path.exists(self.model_path):
            if rewrite or not os.path.exists(os.path.join(self.model_path,self.name + 'last_model.pth')):
                shutil.rmtree(self.model_path)
            else:
                self.resume_training = True

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
            os.mkdir(self.logs_path)

        if connect_tb:
            self.tb_writer = SummaryWriter(logdir=self.logs_path)

    def cuda(self):
        if self.model is not None:
            self.model.cuda()
        self.state.cuda = True

    def train(self, criterion:dict, optimizer, optimizer_params, scheduler, scheduler_params, training_data_loader,
              evaluation_data_loader, pretrained_weights, train_metrics, val_metrics,
              epoches, comparator, virtual_batch):

        # TODO: custom initializer here

        # load weights if any
        if self.resume_training:
            # load training and continue
            self.load_latest()
        elif pretrained_weights is not None:
            # load dictionary only
            self.model.load_state_dict(pretrained_weights)
            self.state.best_val = comparator.get()
        else:
            self.state.best_val = comparator.get()

        if isinstance(optimizer, type):
            optimizer = optimizer(params=self.model.parameters(), **optimizer_params)

        if scheduler is not None:
            if isinstance(scheduler, type):
                scheduler = scheduler(optimizer=optimizer, **scheduler_params)

        assert (isinstance(optimizer, torch.optim.Optimizer))
        assert (isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler) or scheduler is None)

        if self.state.optimizer_state is not None:
            optimizer.load_state_dict(self.state.optimizer_state)
            print('Loaded optimizer state')

        # prepare dicts for metrics
        if not self.state.train_metric:
            for m in train_metrics:
                self.state.train_metric[m] = []

            for m in val_metrics:
                self.state.val_metric[m] = []


        # training loop
        start_epoch = self.state.epoch

        print('Evaluation Run')

        self._evaluate_and_save(evaluation_data_loader, val_metrics, self.state.val_metric, -1, comparator)
        print('End Evaluation Run')

        for i in range(start_epoch, epoches):
            gc.collect()
            tic = time.time()

            # gpu debug
            #for obj in gc.get_objects():
            #    try:
            #        if torch.is_tensor(obj) and obj.is_cuda or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #            print(type(obj), obj.size())
            #    except:
            #        pass

            self.state.global_step = self._train_one_epoch(criterion, optimizer, training_data_loader, train_metrics,
                                                           self.state.train_metric, i, self.state.global_step,
                                                           scheduler, virtual_batch)

            self._evaluate_and_save(evaluation_data_loader, val_metrics, self.state.val_metric, i, comparator)

            torch.cuda.empty_cache()

            tac = time.time()
            print('Epoch %d, time %s \n' % (i, tac - tic))

            self._save(suffix='_epoch_' + str(self.state.epoch))
            self._save(suffix='last_model')
            self.state.epoch = self.state.epoch + 1

    def predict(self, batch, **kwargs):
        self.model.eval()

        torch.cuda.empty_cache()
        gc.collect()

        if self.state.cuda:
            self.model.cuda()

        with torch.no_grad():
            assert (isinstance(batch, dict))

            if self.state.cuda:
                batch = self._array_to_cuda(batch)

            output = self.model(batch, **kwargs)
        return output

    def _array_to_cuda(self,array):
        if isinstance(array, torch.Tensor):
            array = array.cuda()
        elif isinstance(array, dict):
            for key in array:
                array[key] = self._array_to_cuda(array[key])

        return array

    def _train_one_epoch(self, criterion:dict, optimizer, training_data_loader, train_metrics:dict, train_metrics_results, epoch,
                         global_step, scheduler, virtual_batch):

        for m in train_metrics:
            train_metrics[m].reset()

        if self.state.cuda:
            self.model.cuda()

        self.model.train()

        optimizer.zero_grad()
        for idx, batch in enumerate(training_data_loader):
            torch.cuda.empty_cache()

            assert (isinstance(batch, dict))

            if self.state.cuda:
                batch = self._array_to_cuda(batch)

            output = self.model(batch)

            loss_list = [(criterion[c](batch, output), c) for c in criterion]
            loss = sum([l for (l, skip), name in loss_list if skip is False]) / (len(loss_list))

            if isinstance(loss, torch.Tensor):
                loss.backward()

            if (idx + 1) % virtual_batch == 0:

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.tb_writer.add_scalar('misc/grad-abs-{}'.format(name),
                                                  torch.mean(torch.abs(param.grad)).cpu().numpy(), global_step)

                #for param in self.model.parameters():
                #    param.grad.data = torch.clamp(param.grad.data, min=-1.0,max=1.0)

                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
            del loss

            with torch.no_grad():
                for m in train_metrics:
                    train_metrics[m].update(batch, output)

            for idx, ((l, skip), name) in enumerate(loss_list):
                if not skip:
                    self.tb_writer.add_scalar('loss/loss-{}'.format(name), l.item(), global_step)

            for idx, param_group in enumerate(optimizer.param_groups):
                self.tb_writer.add_scalar('misc/lr-{}'.format(idx), param_group['lr'], global_step)

            global_step = global_step + 1

        for m in train_metrics:
            train_metrics_results[m].append(train_metrics[m].get())
            metrics.print_metrics(self.tb_writer, m, train_metrics[m], 'train/', epoch)

        self.state.optimizer_state = optimizer.state_dict()
        return global_step

    def _evaluate_and_save(self, evaluation_data_loader, val_metrics:dict, val_metrics_results, epoch, comparator):

        for m in val_metrics:
            val_metrics[m].reset()

        if self.state.cuda:
            self.model.cuda()

        self.model.eval()

        for batch in evaluation_data_loader:
            assert (isinstance(batch, dict))

            with torch.no_grad():
                if self.state.cuda:
                    batch = self._array_to_cuda(batch)

                output = self.model(batch)

                for m in val_metrics:
                    val_metrics[m].update(batch, output)

        for m in val_metrics:
            val_metrics_results[m].append(val_metrics[m].get())
            metrics.print_metrics(self.tb_writer, m, val_metrics[m], 'val/', epoch)

        if comparator(val_metrics, self.state.best_val):
            self.state.best_val = comparator.get()
            self._save(suffix='best_model')
            print('model saved')

    def _save(self, suffix):
        s = {'state': self.state,
             'model': self.model}

        torch.save(s, os.path.join(self.model_path, self.name + suffix + '.pth'))

    def _load(self, suffix):
        print('loading model %s' % suffix)
        s = torch.load(os.path.join(self.model_path, self.name + suffix + '.pth'), map_location=torch.device('cpu'))
        self.state = s['state']
        if self.model is None:
            self.model = s['model']
        else:
            self.model.load_state_dict(s['model'].state_dict())

    def load_latest(self):
        self._load('last_model')

    def load_best(self):
        self._load('best_model')