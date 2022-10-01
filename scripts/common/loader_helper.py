import nibabel as nib
import numpy as np
import os
import torch
import pydicom
import re
import pandas as pd

from scipy import ndimage

class DuplicateError(Exception):
    def __init__(self, key, message):
        self.key = key
        self.message = message

def get_key(parent_key, key):
    if parent_key:
        key = parent_key + '.' + key
    return key

# lookup table for naming rules
naming_rule = {
    'Visit.Scan.ScanProperties.ScanProperty': 'name',
    'PatientProperties.PatientProperty': 'name',
    'Visit.VisitProperties.VisitProperty': 'name'
}




def average_outputs(outputs, dtype):
    if isinstance(outputs,list) and dtype==dict:
        keys = outputs[0].keys()
        return {key: average_outputs([d[key] for d in outputs],dtype=type(outputs[0][key])) for key in keys}

    elif isinstance(outputs,list) and dtype==str:
        return outputs[0]
    elif isinstance(outputs, list) and dtype == torch.Tensor:
        return torch.mean(torch.stack(outputs,dim=0),dim=0)
    else:
        assert()

def resize_image(image, size, order = 1):
    return ndimage.zoom(image, tuple([t/f for t,f in zip(size, image.shape)]), order=order)

def closest_to_k(n,k=8):
    if n % k == 0:
        return n
    else:
        return ((n // k) + 1)*k

def bbox3(img):
    """
    compute bounding box of the nonzero image pixels
    :param img: input image
    :return: bbox with shape (2,3) and contents [min,max]
    """
    rows = np.any(img, axis=1)
    rows = np.any(rows, axis=1)

    cols = np.any(img, axis=0)
    cols = np.any(cols, axis=1)

    slices = np.any(img, axis=0)
    slices = np.any(slices, axis=0)

    rows = np.where(rows)
    cols = np.where(cols)
    slices = np.where(slices)
    if (rows[0].shape[0] > 0):
        rmin, rmax = rows[0][[0, -1]]
        cmin, cmax = cols[0][[0, -1]]
        smin, smax = slices[0][[0, -1]]

        return np.array([[rmin, cmin, smin], [rmax, cmax, smax]])
    return np.array([[-1,-1,-1],[0,0,0]])

def bbox2(img):
    """
    compute bounding box of the nonzero image pixels
    :param img: input image
    :return: bbox with shape (2,2) and contents [min,max]
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)

    rows = np.where(rows)
    cols = np.where(cols)
    if (rows[0].shape[0] > 0):
        rmin, rmax = rows[0][[0, -1]]
        cmin, cmax = cols[0][[0, -1]]

        return np.array([[rmin, cmin], [rmax, cmax]])
    return np.array([[-1,-1,-1],[0,0,0]])

def get_checkpoint_path(path, metric_name, format=':.4f', start_epoch=0, n_from_best=0):
    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    rx = re.compile('epoch=(\d+)-'+metric_name+'=(\d+\.\d+)(?:-v(?:\d+))?\.ckpt')
    results = []

    for f in filenames:
        g = rx.search(f)
        if g is None:
            continue
        r = {'epoch':int(g.group(1)),
             'val':float(g.group(2))}
        results.append(r)

    results = pd.DataFrame(results)

    results = results[results['epoch']>=start_epoch].sort_values(by='val',ascending=False)

    return ('epoch={:d}-{}={'+format+'}.ckpt').format(int(results.iloc[n_from_best,:]['epoch']),metric_name,
                                                       results.iloc[n_from_best,:]['val'])



def _array_to_cuda(array):
    if isinstance(array, torch.Tensor):
        array = array.cuda()
    elif isinstance(array, dict):
        for key in array:
            array[key] = _array_to_cuda(array[key])
    elif isinstance(array, list):
        array = [_array_to_cuda(a) for a in array]

    return array


def _array_to_cpu(array):
    if isinstance(array, torch.Tensor):
        array = array.to('cpu')
    elif isinstance(array, dict):
        for key in array:
            array[key] = _array_to_cpu(array[key])
    elif isinstance(array, list):
        array = [_array_to_cpu(a) for a in array]

    return array

def _freeze(params):
    for p in params:
        p.requires_grad = False

def _unfreeze(params):
    for p in params:
        p.requires_grad = True