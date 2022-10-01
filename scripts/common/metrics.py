import torch
import numpy as np
import math
from scipy import ndimage

class Metrics(object):
    def __init__(self):
        self.accumulator = []

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        return 0

    def update(self, ground, predict):
        result = self.calculate_batch(ground,predict)
        self.accumulator.extend(result.tolist())

    def get(self):
        return np.nanmean(self.accumulator)

    def reset(self):
        self.accumulator = []

class RMSE(Metrics):
    def __init__(self, output_key=0, target_key=0):
        super(RMSE, self).__init__()
        self.output_key=output_key
        self.target_key=target_key

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        assert (gr.shape == pred.shape)

        N = gr.shape[0]

        pred = pred.view(N,-1)
        gr = gr.view(N,-1)

        result = (pred - gr) ** 2

        return result.mean(dim=1).cpu().numpy()

    def get(self):
        return np.sqrt(np.nanmean(self.accumulator))


class MONAIWrapper(Metrics):
    def __init__(self, metric, output_key=0, target_key=0):
        super(MONAIWrapper, self).__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.metric = metric

    @torch.no_grad()
    def update(self, ground, predict):

        pred = predict[self.output_key].detach() > 0.5
        gr = ground[self.target_key].detach() > 0.5

        result = self.metric(pred,gr)

        return result

    def get(self):
        agg = self.metric.aggregate()
        return agg[0].item() if isinstance(agg,list) else agg.item()

    def reset(self):
        self.metric.reset()



class MAE(Metrics):
    def __init__(self, output_key=0, target_key=0):
        super(MAE, self).__init__()
        self.output_key=output_key
        self.target_key=target_key

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        assert (gr.shape == pred.shape)

        N = gr.shape[0]

        pred = pred.view(N,-1)
        gr = gr.view(N,-1)

        result = torch.abs(pred - gr)

        return result.mean(dim=1).cpu().numpy()

def print_metrics(writer, name, metric, prefix, epoch):
    if isinstance(metric.get(), np.ndarray):
        for i in range(metric.get().shape[0]):
            writer.add_scalar(prefix + name+str(i), metric.get()[i], epoch)
    else:
        writer.add_scalar(prefix + name, metric.get(), epoch)

    print('Epoch %d, %s %s %s' % (epoch, prefix, name, metric.get()))