import numpy as np
import dataloader
import glob
import os
from sklearn.model_selection import KFold

import json

if __name__ == '__main__':
    training_folder = '../eye_train'
    annotation_files = glob.glob(f"{training_folder}/*.geojson")
    images_files = glob.glob(f"{training_folder}/*.png")

    ids_annotations = [os.path.basename(f).split('.')[0] for f in annotation_files]
    ids_images = [os.path.basename(f).split('.')[0] for f in images_files]
    ids_annotations.remove('673')

    ids = np.array([i for i in ids_annotations if i in ids_images])


    skf = KFold(n_splits=5, shuffle=True, random_state=42)

    for idx, (training_index, val_index) in enumerate(skf.split(ids)):
        val_fold = ids[val_index].tolist()
        train_folds = ids[training_index].tolist()

        with open('../splits/{}.json'.format(idx),'w') as f:
            json.dump({'training':train_folds, 'val':val_fold},f)