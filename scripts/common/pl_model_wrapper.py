import torch
import pytorch_lightning as pl
from . import loader_helper
import gc
import numpy as np
import monai
from monai.transforms import *
import os

import matplotlib.pyplot as plt

class Model(pl.LightningModule):

    def __init__(self,model, losses, training_metrics, metrics, metametrics, optim, post_transforms, save_dir):
        super().__init__()
        self.model = model
        self.loss = losses
        self.metrics = metrics
        self.metametrics = metametrics
        self.optim = optim
        self.training_metrics = training_metrics
        self.post_transforms = post_transforms
        self.save_dir = save_dir

        if self.save_dir is not None:
            os.makedirs(os.path.join(self.save_dir,'viz_train'),exist_ok=True)


    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def training_step(self, batch, batch_idx):
        #gc.collect()

        self.model = self.model.train()
        res = self.model(batch)
        loss, values = self.loss(batch, res)

        for k in values:
            self.log('Training/'+str(k), values[k].item(), on_step=True,on_epoch=False)

        with torch.no_grad():
            for k in self.training_metrics:
                self.training_metrics[k].update(batch,res)

        return loss

    def training_epoch_end(self, outputs) -> None:

        metric_results = {k:self.training_metrics[k].get() for k in self.training_metrics}

        if self.training_metrics is not None:
            for k in self.training_metrics:
                self.log('Training/' + str(k), metric_results[k], on_epoch=True)
                self.training_metrics[k].reset()

        del metric_results
        gc.collect()
        #torch.cuda.empty_cache()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        gc.collect()
        #torch.cuda.empty_cache()
        self.model = self.model.eval()

        res = self.model(batch)

        for k in self.metrics:
            self.metrics[k].update(batch,res)
        '''
        batch.update(res)

        for b in monai.data.utils.decollate_batch(batch,detach=True,pad=False):

            b = self.post_transforms(b)

            gt = monai.visualize.blend_images(b['image'].cpu(),b['mask_orig'].cpu()).numpy().transpose((1,2,0))
            pred = monai.visualize.blend_images(b['image'].cpu(), b['proba'].cpu()).numpy().transpose((1,2,0))
            pred_t = monai.visualize.blend_images(b['image'].cpu(), b['proba'].cpu()>0.5).numpy().transpose((1,2,0))



            fig, ax = plt.subplots(1, 3)
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            ax[0].imshow((np.clip(gt,0,1)*255).astype(np.uint8))
            ax[1].imshow((np.clip(pred,0,1)*255).astype(np.uint8))
            ax[2].imshow((np.clip(pred_t,0,1)*255).astype(np.uint8))
            fig.savefig(os.path.join(os.path.join(self.save_dir,'viz_train'),os.path.basename(b['path'])),dpi=300)
            plt.close('all')
        '''

        del res

    def validation_epoch_end(self, validation_step_outputs):

        metric_results = {k:self.metrics[k].get() for k in self.metrics}

        for k in self.metrics:
            self.log('Validation/'+str(k), metric_results[k], on_epoch=True)
            self.metrics[k].reset()

        if self.metametrics is not None:
            for k in self.metametrics:
                self.log(str(k), self.metametrics[k].get(metric_results), on_epoch=True)

        del metric_results
        gc.collect()
        #torch.cuda.empty_cache()

    def configure_optimizers(self):
        return self.optim