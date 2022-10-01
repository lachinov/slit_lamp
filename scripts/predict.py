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
parser.add_argument("--test_path", default="", type=str, help="path to train data")
parser.add_argument("--output_path", default="", type=str, help="path to train data")




def average_outputs(outputs, dtype):
    if isinstance(outputs,list) and dtype==dict:
        keys = outputs[0].keys()
        return {key: average_outputs([d[key] for d in outputs],dtype=type(outputs[0][key])) for key in keys}

    elif isinstance(outputs,list) and dtype==str:
        return outputs[0]
    elif isinstance(outputs, list) and (dtype == torch.Tensor or dtype == monai.data.meta_tensor.MetaTensor):
        return sum(outputs)/len(outputs)
    else:
        print(dtype)
        assert()

params = {'losses':None, 'metrics':None, 'metametrics':None, 'optim':None, 'training_metrics':None, 'post_transforms':None, 'save_dir':None}

with torch.no_grad():
    if __name__ == '__main__':
        opt = parser.parse_args()
        print(torch.__version__)
        print(opt)

        with open(opt.config) as f:
            config = json.load(f)

        torch.backends.cudnn.benchmark = True


        model_path = os.path.join(config['models_path'],config['name'])
        models_list = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path,d))]
        models = []

        model = importlib.import_module(config['code_path'] + '.model')


        for n, f in enumerate(models_list):
            arch = model.SMP_Model(**config['model_kwargs'])

            compiled_model = pl_model_wrapper.Model(model=arch, **params)
            path_weights = loader_helper.get_checkpoint_path(os.path.join(model_path, f), 'Dice', format=':.6f', start_epoch=0,
                                                             n_from_best=0)
            checkpoint = torch.load(os.path.join(os.path.join(model_path, f), path_weights))
            compiled_model.load_state_dict(checkpoint['state_dict'], strict=True)
            compiled_model = compiled_model.eval().cuda()

            models.append(compiled_model)

        val_data = dataloader.EyeDatasetInfer(data_folder=opt.test_path)
        val_data_dataset = monai.data.Dataset(data=val_data, transform=model.data_transform_val)

        evaluation_data_loader = monai.data.DataLoader(dataset=val_data_dataset, num_workers=4, batch_size=1,
                                                       shuffle=False, drop_last=False)

        #torch.inference_mode(True)

        with monai.transforms.utils.allow_missing_keys_mode(model.data_transform_val):
            for batch in evaluation_data_loader:

                batch['image'] = loader_helper._array_to_cuda(batch['image'])


                outputs = []
                #batch = loader_helper._array_to_cuda(batch)
                for m in models:
                    output = m(batch)
                    outputs.append(output)

                output = average_outputs(outputs, dict)
                batch.update(output)

                batch = monai.data.utils.default_collate([model.data_transform_post(b) for b in monai.data.utils.decollate_batch(batch)])

                copy_list = ['path']

                proba = (batch['proba'] > 0.50).cpu().numpy()[0,0]

                cv2.imwrite(os.path.join(opt.output_path,os.path.basename(batch['path'][0])),(255*proba).astype(np.uint8))
                print(batch['path'][0])

                del output, batch

