import numpy as np
from flask import Flask, render_template, request, send_file
import importlib
import json
import os

import torch
import torch.utils.data
import hashlib
from PIL import Image

from common import pl_model_wrapper
from common import loader_helper
import dataloader
import monai

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
savefolder = './static/img/'


config_path = '../models/test_003_u_effb3_adam.json/config.json'
with open(config_path) as f:
    config = json.load(f)

params = {'losses':None, 'metrics':None, 'metametrics':None, 'optim':None, 'training_metrics':None, 'post_transforms':None, 'save_dir':None}

model = importlib.import_module(config['code_path'] + '.model')
arch = model.SMP_Model(**config['model_kwargs'])
compiled_model = pl_model_wrapper.Model(model=arch, **params)

model_path = os.path.join(config['models_path'], config['name'])
models_list = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path,d))]
f = models_list[0]
path_weights = loader_helper.get_checkpoint_path(os.path.join(model_path, f), 'Dice', format=':.6f', start_epoch=0,
                                                             n_from_best=0)
checkpoint = torch.load(os.path.join(os.path.join(model_path, f), path_weights))
compiled_model.load_state_dict(checkpoint['state_dict'], strict=True)
compiled_model = compiled_model.eval().cuda()

data_transform_post = model.data_transform_post

def transform_image(img):
    transforms = model.data_transform_val

    batch = {'image': img.transpose((2,0,1)),
             'mask_orig': np.zeros_like(img).transpose((2,0,1))}
    batch = transforms(batch)
    return batch

@torch.no_grad()
def get_prediction(img):
    batch = transform_image(img)
    batch = monai.data.list_data_collate([batch])

    batch['image'] = loader_helper._array_to_cuda(batch['image'])


    outputs = compiled_model(batch)
    batch.update(outputs)

    batch = monai.data.list_data_collate(
        [data_transform_post(b) for b in monai.data.decollate_batch(batch,detach=True,pad=False)])

    proba = (batch['proba'] > 0.5263).cpu().numpy()[0, 0]

    return proba

@app.route('/infer', methods=['POST'])
def success():
    global savefolder
    if request.method == 'POST':
        f = request.files['file']
        saveLocation = f.filename
        saveLocation = savefolder + saveLocation
        f.save(saveLocation)

        img = dataloader.EyeDataset.read_image(saveLocation)

        image_nparray = get_prediction(img)

        output_hash = hashlib.sha1(saveLocation.encode("UTF-8")).hexdigest()[:20]
        output_image = savefolder+output_hash+".jpeg"
        result_image = Image.fromarray(image_nparray)
        result_image.resize((img.shape[1],img.shape[0]))
        result_image.save(output_image)
        return send_file(output_image, mimetype='image/jpg')

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 8001))
    app.run(host='0.0.0.0', port=port, debug=True)
