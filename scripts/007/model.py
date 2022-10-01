import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import torch.nn.functional as F

import monai
from monai.transforms import *
from monai.networks import nets
import monai.networks.blocks

from common import loss



class CacheStop(RandomizableTransform, MapTransform, InvertibleTransform):
    def __init__(self,allow_missing_keys=True) -> None:
        RandomizableTransform.__init__(self, 1.0)
        self.allow_missing_keys= allow_missing_keys

    def __call__(self, data):
        d = dict(data)
        return d

    def inverse(self, data):
        d = dict(data)
        return d

data_transform = Compose([
    #ToTensord(keys=["image", "mask"]),
    NormalizeIntensityd(keys=['image'], channel_wise=True, subtrahend=[0.18805979, 0.16423294, 0.14528641],
                        divisor=[0.22065658, 0.21452696, 0.19555074]),
    # Lambdad(keys=['image'], func = lambda x: (x - x.mean()) / x.std()),
    RandCropByPosNegLabeld(keys=["image", "mask"], label_key='mask', pos=0.5, neg=0.5, spatial_size=[192, 192],
                           num_samples=4),
    #Zoomd(keys=['image', 'mask'], zoom=2, mode=['bilinear', 'nearest'], keep_size=False),

    RandFlipd(keys=["image", "mask"], spatial_axis=(0, 1), prob=0.3),

    RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.8),
    RandShiftIntensityd(keys=['image'], offsets=0.5, prob=0.8),
    # RandCoarseDropoutd(keys=['image'],holes=2,max_holes=5,spatial_size=(700,10),max_spatial_size=(700,20),fill_value=(-1,1),prob=0.4),

])

data_transform_val = Compose([
    #ToTensord(keys=["image", "mask_orig"]),
    NormalizeIntensityd(keys=['image'], channel_wise=True, subtrahend=[0.18805979, 0.16423294, 0.14528641],
                        divisor=[0.22065658, 0.21452696, 0.19555074]),
    CacheStop(),
    #Zoomd(keys=['image', 'mask'], zoom=2, mode=['bilinear', 'area'], keep_size=False),
    # Lambdad(keys=['image'], func = lambda x: (x - x.mean()) / x.std()),
    DivisiblePadd(keys=["image", "mask_orig"], k=32, mode='reflect'),
    # RandCropByPosNegLabeld(keys=["image", "mask"], label_key='mask',pos=1,neg=0, spatial_size=[256, 256], num_samples=1),

    # RandFlipd(keys=["image", "mask"], spatial_axis=(1,2), prob=0.3),

    # RandScaleIntensityd(keys=['image'],factors=0.1,prob=0.8),
    # RandShiftIntensityd(keys=['image'],offsets=0.5,prob=0.8),
    # RandCoarseDropoutd(keys=['image'],holes=2,max_holes=5,spatial_size=(700,10),max_spatial_size=(700,20),fill_value=(-1,1),prob=0.4),

])

data_transform_post = Compose([
    Invertd(
        keys=['image', 'proba', 'mask_orig'],  # invert the `pred` data field, also support multiple fields
        transform=data_transform_val,
        orig_keys=['image', 'mask_orig', 'mask_orig'],
        nearest_interp=False,
        to_tensor=True,
        allow_missing_keys=True
    )
])




class Model(nn.Module):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super(Model, self).__init__()

        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )



    def forward(self, x):

        out = self.model(x)

        return out

class SMP_Model(nn.Module):
    def __init__(self, **kwargs):
        super(SMP_Model, self).__init__()

        self.model = Model(**kwargs)

    def forward(self, x):

        if self.training == False:
            inf = monai.inferers.SlidingWindowInferer(roi_size=512,sw_batch_size=1,overlap=0.25,mode='gaussian')
            out = inf(x['image'],network=self.model)

            #out = -F.max_pool2d(input=-out,kernel_size=3,stride=1,padding=1,return_indices=False)
        else:
            out = self.model(x['image'])

        return {
            'prediction': out,
            'proba':torch.sigmoid(out)
        }



criterion = loss.Mix(losses={
        'Dice Loss ': loss.LossWrapper(loss=monai.losses.Dice(include_background=True,to_onehot_y=False,squared_pred=True,reduction='mean',batch=True),output_key='proba', target_key='mask'),
        'Focal Loss ': loss.LossWrapper(loss=monai.losses.FocalLoss(include_background=True,to_onehot_y=False,gamma=2.0,reduction='mean'),output_key='proba', target_key='mask'),
        #'Recall Loss': loss.RecallLoss(output_key='proba', target_key='mask')
    })