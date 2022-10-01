import sys

print(sys.path)

import numpy as np
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import os
import copy
import json
import gc
import monai
import monai.data

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor



from common import loss
from common import metrics
import dataloader
from common import pl_model_wrapper

import importlib
import segmentation_models_pytorch as smp


parser = argparse.ArgumentParser(description="PyTorch (:")
parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
parser.add_argument("--virtualBatchSize", type=int, default=1, help="virual training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--train_path", default="", type=str, help="path to train data")
parser.add_argument("--split_path", default="", type=str, help="path to the splits")
parser.add_argument("--split", nargs='+', default=None, type=str, help="splits to train")
parser.add_argument("--name", default="test", type=str, help="experiment name")
parser.add_argument("--models_path", default=None, type=str, help="path to models folder")
parser.add_argument("--gpus", default=1, type=int, help="number of gpus")



def worker_init_fn(worker_id):
    seed = torch.initial_seed() + worker_id
    np.random.seed([int(seed % 0x80000000), int(seed // 0x80000000)])
    torch.manual_seed(seed)
    random.seed(seed)
    #print('worker id {} seed {}'.format(worker_id, seed))


def parse():
    return parser.parse_args()

def lr_function(step):
    step_list = [20000, 30000, 40000]
    lr_list = [1,1e-1,1e-1,1e-2]#[1e-2,1,1e-1,1e-2]

    for idx, s in enumerate(step_list):
        if step > s:
            return lr_list[idx+1]

    return lr_list[0]

def train_main(opt:dict=None, training_file_list=None, validation_file_list=None, split_name=None):
    print(opt)
    print(torch.__version__)
    print(monai.__version__)
    print(smp.__version__)
    print(pl.__version__)
    print(torch.backends.cudnn.version())
    print(*torch.__config__.show().split("\n"), sep="\n")
    print('cuda devices', torch.cuda.device_count())

    model = importlib.import_module(opt['code_path']+'.model')

    opt['seed'] = 1337
    pl.seed_everything(1234)


    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    if training_file_list is None or validation_file_list is None:
        print('The training or validation list is empty')

    #training_set = loader_helper.get_patient_dict(opt.train_path, training_file_list)  # splits.train
    #validation_set = loader_helper.get_patient_dict(opt.train_path, validation_file_list)  # splits.train_val

    print("===> Building model")

    arch = model.SMP_Model(**opt['model_kwargs'])

    #arch.apply(weight_init.weight_init)

    print("===> Loading datasets")
    print('Train data:', opt['train_path'])

    print('Train {}'.format(training_file_list))
    print('Val {}'.format(validation_file_list))


    train_data = dataloader.EyeDataset(data_folder=opt['train_path'],  images=training_file_list, provide_orig_mask=False, epoch_size=opt['epoch_size'] if 'epoch_size' in opt else None)
    val_data = dataloader.EyeDataset(data_folder=opt['train_path'],  images=validation_file_list, provide_orig_mask=True)

    training_data_dataset = monai.data.Dataset(data = train_data,transform=model.data_transform)
    val_data_dataset = monai.data.CacheDataset(data=val_data, transform=model.data_transform_val)

    training_data_loader = monai.data.DataLoader(dataset=training_data_dataset,num_workers=opt['threads'], batch_size=opt['batchSize'],
                                      shuffle=True, drop_last=True, worker_init_fn=worker_init_fn,
                                      pin_memory=True)

    evaluation_data_loader = monai.data.DataLoader(dataset=val_data_dataset, num_workers=4, batch_size=1, shuffle=False, drop_last=False)

    if hasattr(model,'criterion'):
        criterion = model.criterion
    else:
        criterion =  loss.Mix(losses={
            'Dice Loss ': loss.LossWrapper(loss=monai.losses.Dice(include_background=True,to_onehot_y=False,squared_pred=True,reduction='mean',batch=True),output_key='proba', target_key='mask'),
            'Focal Loss ': loss.LossWrapper(loss=monai.losses.FocalLoss(include_background=True,to_onehot_y=False,gamma=2.0,reduction='mean'),output_key='proba', target_key='mask'),
        })

    print("===> Training")

    metrics_train = {
        'Dice': metrics.MONAIWrapper(metric=monai.metrics.DiceMetric(include_background=True), output_key='proba', target_key='mask'),
    }

    metrics_val = {
        'Dice': metrics.MONAIWrapper(metric=monai.metrics.DiceMetric(include_background=True), output_key='proba', target_key='mask_orig'),
        'Precision': metrics.MONAIWrapper(metric=monai.metrics.ConfusionMatrixMetric(include_background=True,metric_name='precision',reduction='mean'), output_key='proba', target_key='mask_orig'),
        'Recall': metrics.MONAIWrapper(metric=monai.metrics.ConfusionMatrixMetric(include_background=True,metric_name='recall',reduction='mean'), output_key='proba', target_key='mask_orig'),
        'Hausdorff': metrics.MONAIWrapper(metric=monai.metrics.HausdorffDistanceMetric(include_background=True), output_key='proba', target_key='mask_orig'),
    }

    class mmetric:
        def get(sefl, m: dict):
            return m['Dice']

    meta_metric_val = {'Dice': mmetric()}

    logger = TensorBoardLogger(
        save_dir=os.path.join(opt['models_path'], opt['name']),
        version=None,
        name='logs',
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(opt['models_path'], opt['name']),
        filename='{epoch}-{Dice:.6f}',
        save_top_k=10,
        # verbose=True,
        monitor='Dice',
        mode='max',
        # prefix=''
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    optimizers = [torch.optim.Adam(arch.parameters(), lr=1e-3),]

    # torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizers[0],
    #                                     milestones=[800000, ], gamma=0.5),

    schedulers = [{'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer=optimizers[0],lr_lambda=lr_function),
                   'name': 'step_scheduler',
                   'interval': 'step',
                   }, ]

    compiled_model = pl_model_wrapper.Model(model=arch,
                                            losses=criterion,
                                            training_metrics=metrics_train,
                                            metrics=metrics_val,
                                            metametrics=meta_metric_val,
                                            optim=(optimizers, schedulers),
                                            post_transforms=model.data_transform_post,
                                            save_dir=os.path.join(opt['models_path'], opt['name']))


    trainer = pl.Trainer(logger=logger,
                         callbacks=[lr_monitor, checkpoint_callback],
                         log_every_n_steps=2,
                         precision=32,
                         devices=opt['gpus'],
                         num_sanity_val_steps=2,
                         check_val_every_n_epoch=5,
                         accumulate_grad_batches=opt['virtualBatchSize'],
                         max_epochs=opt['nEpochs'],
                         sync_batchnorm=False,
                         enable_model_summary=True, enable_progress_bar=False, benchmark=True,accelerator='gpu')

    trainer.fit(compiled_model, train_dataloaders=training_data_loader, val_dataloaders=evaluation_data_loader)


def main(**kwargs):
    split_folder = kwargs['split_path']

    split_list = [f for f in os.listdir(split_folder) if ('split' in f) and f.endswith('.json')]

    if not kwargs['split'] is None:
        split_list = kwargs['split']

    for n, split_filename in enumerate(split_list):
        gc.collect()
        torch.cuda.empty_cache()
        split_opt = copy.deepcopy(kwargs)

        model_path = os.path.join(kwargs['models_path'], kwargs['name'])
        os.makedirs(model_path, exist_ok=True)

        with open(os.path.join(model_path,'config.json'),'w') as f_conf:
            json.dump(kwargs,f_conf)

        split_opt['name'] = kwargs['name'] + '_' + split_filename
        split_opt['models_path'] = model_path

        with open(os.path.join(kwargs['split_path'], split_filename), 'r') as f:
            split_dict = json.load(f)

        train_ids = split_dict['training']
        val_ids = split_dict['val']

        print('running {} out of {}, training samples {}'.format(n + 1, len(split_list), len(train_ids)))
        train_main(split_opt, train_ids, val_ids, split_filename)


if __name__ == "__main__":
    #try:
    #    torch.multiprocessing.set_start_method('spawn')
    #except RuntimeError:
    #    pass
    opt = vars(parse())

    main(**opt)