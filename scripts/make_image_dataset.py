import numpy as np
import monai
import cv2
import os
import glob

import dataloader


if __name__ == '__main__':
    training_folder = '../eye_train'


    annotation_files = glob.glob(f"{training_folder}/*.geojson")
    images_files = glob.glob(f"{training_folder}/*.png")
    ids_annotations = [os.path.basename(f).split('.')[0] for f in annotation_files]
    ids_images = [os.path.basename(f).split('.')[0] for f in images_files]
    ids_annotations.remove('673')
    ids = [i for i in ids_annotations if i in ids_images]


    #ids=['162']

    dataset = dataloader.EyeDataset(data_folder=training_folder,images=ids)

    mean = np.zeros(3)
    std = np.zeros(3)

    for i_dict in dataset:
        mask = i_dict['mask'].transpose((1,2,0))

        print(i_dict['path'], mask.shape , mask.max())

        cv2.imwrite(os.path.join(training_folder+'/masks/',os.path.basename( i_dict['path']))+'.mask.png',(mask*255).astype(np.uint8))