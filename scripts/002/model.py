import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from monai.transforms import *


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
    ToTensord(keys=["image", "mask","mask_orig"],allow_missing_keys=True),
    NormalizeIntensityd(keys=['image'], channel_wise=True, subtrahend=[0.18805979, 0.16423294, 0.14528641],
                        divisor=[0.22065658, 0.21452696, 0.19555074]),
    # Lambdad(keys=['image'], func = lambda x: (x - x.mean()) / x.std()),
    RandCropByPosNegLabeld(keys=["image", "mask"], label_key='mask', pos=0.1, neg=0.9, spatial_size=[128, 128],
                           num_samples=4),

    RandFlipd(keys=["image", "mask"], spatial_axis=(0, 1), prob=0.3),

    RandScaleIntensityd(keys=['image'], factors=0.5, prob=0.8),
    RandShiftIntensityd(keys=['image'], offsets=2.5, prob=0.8),
    # RandCoarseDropoutd(keys=['image'],holes=2,max_holes=5,spatial_size=(700,10),max_spatial_size=(700,20),fill_value=(-1,1),prob=0.4),

])

data_transform_val = Compose([
    ToTensord(keys=["image", "mask"]),
    NormalizeIntensityd(keys=['image'], channel_wise=True, subtrahend=[0.18805979, 0.16423294, 0.14528641],
                        divisor=[0.22065658, 0.21452696, 0.19555074]),
    CacheStop(),
    # Lambdad(keys=['image'], func = lambda x: (x - x.mean()) / x.std()),
    DivisiblePadd(keys=["image", "mask"], k=32, mode='reflect'),
    # RandCropByPosNegLabeld(keys=["image", "mask"], label_key='mask',pos=1,neg=0, spatial_size=[256, 256], num_samples=1),

    # RandFlipd(keys=["image", "mask"], spatial_axis=(1,2), prob=0.3),

    # RandScaleIntensityd(keys=['image'],factors=0.1,prob=0.8),
    # RandShiftIntensityd(keys=['image'],offsets=0.5,prob=0.8),
    # RandCoarseDropoutd(keys=['image'],holes=2,max_holes=5,spatial_size=(700,10),max_spatial_size=(700,20),fill_value=(-1,1),prob=0.4),

])

data_transform_post = Compose([
    Invertd(
        keys=['image', 'mask', 'proba'],  # invert the `pred` data field, also support multiple fields
        transform=data_transform_val,
        orig_keys=['image', 'mask', 'mask'],
        nearest_interp=False,
        to_tensor=True,
        allow_missing_keys=True
    ),
])



class SMP_Model(nn.Module):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super(SMP_Model, self).__init__()

        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )


    def forward(self, x):
        out = self.model(x['image'])

        return {
            'prediction': out,
            'proba':torch.sigmoid(out)
        }




