import numpy as np
import torch
import argparse
from matplotlib import pyplot as plt
import math
import re
import pandas as pd
import pickle
import os
from scipy import ndimage
import nibabel as nii
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats
from torch.utils.data.dataloader import DataLoader
import copy
import seaborn as sns
import gc
import cv2
import monai
import json

from common import loader_helper
from common import trainer
import dataloader
from common import metrics
from common import pl_model_wrapper
import importlib

parser = argparse.ArgumentParser(description="PyTorch Validate")
parser.add_argument("--config", default="/models", type=str, help="path to models folder")
parser.add_argument("--validation_path", default="", type=str, help="path to train data")



params = {'losses':None, 'metrics':None, 'metametrics':None, 'optim':None, 'training_metrics':None, 'post_transforms':None, 'save_dir':None}

if __name__ == '__main__':
    opt = parser.parse_args()
    print(torch.__version__)
    print(opt)

    with open(opt.config) as f:
        config = json.load(f)

    torch.backends.cudnn.benchmark = True

    metrics_val = {
        'Dice': metrics.MONAIWrapper(metric=monai.metrics.DiceMetric(include_background=True), output_key='proba',
                                     target_key='mask_orig'),
        'Hausdorff': metrics.MONAIWrapper(metric=monai.metrics.HausdorffDistanceMetric(include_background=True),
                                          output_key='proba', target_key='mask_orig'),
    }


    metrics_thresholds = {'Dice': [],
                          'Hausdorff':[]}

    split_folder = config['split_path']

    split_list = [f for f in os.listdir(split_folder) if ('split' in f) and f.endswith('.json')]

    if not config['split'] is None:
        split_list = config['split']

    #split_list = ['0.json']

    model = importlib.import_module(config['code_path']+'.model')

    results = []
    results_plots = []

    #torch.inference_mode(True)

    for n, split_filename in enumerate(split_list):
        split_opt = copy.deepcopy(config)

        split_opt['name'] = config['name'] + '_' + split_filename
        model_path = os.path.join(split_opt['models_path'], config['name'], split_opt['name'])
        split_opt['models_path'] = model_path


        with open(os.path.join(config['split_path'], split_filename),'r') as f:
            split_dict = json.load(f)

        train_ids = split_dict['training']
        val_ids = split_dict['val']
        print('running {} out of {}, validation samples {}'.format(n, len(split_list), len(val_ids)))

        val_data = dataloader.EyeDataset(data_folder=opt.validation_path, images=val_ids, provide_orig_mask=True)
        val_data_dataset = monai.data.Dataset(data=val_data, transform=model.data_transform_val)

        evaluation_data_loader = monai.data.DataLoader(dataset=val_data_dataset, num_workers=4, batch_size=1,
                                                       shuffle=False, drop_last=False)

        print("===> Building model")
        layers = [1, 1, 2, 4]
        number_of_channels = [int(32 * 1 * 2 ** i) for i in range(0, len(layers))]  # [16,16,16,32,32,64,64,64,64,16]

        arch = model.SMP_Model(**config['model_kwargs'])

        compiled_model = pl_model_wrapper.Model(model=arch, **params)
        path_weights = loader_helper.get_checkpoint_path(split_opt['models_path'], 'Dice', format=':.6f', start_epoch=0, n_from_best=0)
        checkpoint = torch.load(os.path.join(split_opt['models_path'], path_weights))
        compiled_model.load_state_dict(checkpoint['state_dict'], strict=True)
        compiled_model = compiled_model.eval().cuda()

        with torch.no_grad():
            for batch in evaluation_data_loader:
                gc.collect()
                torch.cuda.empty_cache()

                batch['image'] = loader_helper._array_to_cuda(batch['image'])
                batch['mask'] = loader_helper._array_to_cuda(batch['mask'])

                metrics_row = {}

                output = compiled_model(batch)
                batch.update(output)

                batch = monai.data.utils.default_collate([model.data_transform_post(b) for b in monai.data.utils.decollate_batch(batch)])

                copy_list = ['path']

                for c in copy_list:
                    if isinstance(batch[c], torch.Tensor):
                        metrics_row[c] = batch[c].item()
                    else:
                        metrics_row[c] = batch[c][0]

                for m, v in metrics_val.items():
                    metrics_thresholds[m].append([])

                proba = batch['proba'].clone()
                #for t in np.linspace(0,1,20):
                #    batch['proba'] = proba > t
                #    for m, v in metrics_val.items():
                #        c = v.update(batch, batch)
                #        #metrics_row[m] = c if c is None else c.item()
                #        metrics_thresholds[m][-1].append(c if c is None else c.item())

                for m, v in metrics_val.items():
                    batch['proba'] = proba #> 0.5263
                    c = v.update(batch, batch)
                    metrics_row[m] = c if c is None else c.item()

                results.append(metrics_row)


                del output, batch

                print(metrics_row)


    df_results = pd.DataFrame(results)
    df_results.to_csv('cross_val_output.csv')

    df_results = pd.read_csv('cross_val_output.csv',index_col=0)


    print(df_results.describe())
    '''
    for m, v in metrics_val.items():
        x = np.linspace(0,1,20)
        mean_metrics = np.array(metrics_thresholds[m]).mean(axis=0)
        plt.plot(x, mean_metrics)
        plt.title(m)
        plt.show()
        plt.close()

        print(x[np.argmax(mean_metrics)], x[np.argmin(mean_metrics)])
    '''