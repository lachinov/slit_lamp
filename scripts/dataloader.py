import numpy as np
from torch.utils.data import Dataset
import cv2
import glob
import json
import os

from skimage import morphology

class EyeDataset(Dataset):
    """
    Класс датасета, организующий загрузку и получение изображений и соответствующих разметок
    """

    def __init__(self, data_folder: str, images: list = None, provide_orig_mask=False, epoch_size=None):
        self.class_ids = {"vessel": 1}

        self.data_folder = data_folder
        self.provide_orig_mask = provide_orig_mask

        if images is not None:
            self._image_files = [f"{data_folder}/{f}.png" for f in images]
        else:
            self._image_files = glob.glob(f"{data_folder}/*.png")

        self.epoch_size = epoch_size if (epoch_size is not None) else len(self._image_files)


    @staticmethod
    def read_image(path: str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image / 255, dtype=np.float32)
        return image

    @staticmethod
    def parse_polygon(coordinates: dict, image_size: tuple) -> np.ndarray:
        mask = np.zeros(image_size, dtype=np.float32)
        if len(coordinates) == 1:
            points = [np.int32(coordinates)]
            cv2.fillPoly(mask, points, 1,lineType=cv2.LINE_8)
        else:
            cv2.fillPoly(mask, [np.int32(c) for c in coordinates], 1,lineType=cv2.LINE_8)
        return mask

    @staticmethod
    def parse_mask(shape: dict, image_size: tuple) -> np.ndarray:
        """
        Метод для парсинга фигур из geojson файла
        """
        mask = np.zeros(image_size, dtype=np.float32)
        coordinates = shape['coordinates']
        if shape['type'] == 'MultiPolygon':
            for polygon in coordinates:
                mask += EyeDataset.parse_polygon(polygon, image_size)
        else:
            mask += EyeDataset.parse_polygon(coordinates, image_size)

        return mask

    def read_layout(self, path: str, image_size: tuple) -> np.ndarray:
        """
        Метод для чтения geojson разметки и перевода в numpy маску
        """
        with open(path, 'r', encoding='cp1251') as f:  # some files contain cyrillic letters, thus cp1251
            json_contents = json.load(f)

        num_channels = 1 + max(self.class_ids.values())
        mask_channels = [np.zeros(image_size, dtype=np.float32) for _ in range(num_channels)]
        mask = np.zeros(image_size, dtype=np.float32)

        if type(json_contents) == dict and json_contents['type'] == 'FeatureCollection':
            features = json_contents['features']
        elif type(json_contents) == list:
            features = json_contents
        else:
            features = [json_contents]

        for shape in features:
            channel_id = self.class_ids["vessel"]
            mask = self.parse_mask(shape['geometry'], image_size)
            mask_channels[channel_id] = np.maximum(mask_channels[channel_id], mask)


        #mask_channels[0] = 1 - np.max(mask_channels[1:], axis=0)


        return (np.expand_dims(mask_channels[1],axis=-1) > 0).astype(np.float32)#np.stack(mask_channels, axis=-1)

    def __getitem__(self, idx: int) -> dict:
        # Достаём имя файла по индексу
        idx = idx%len(self._image_files)

        image_path = self._image_files[idx]

        # Получаем соответствующий файл разметки
        json_path = image_path.replace("png", "geojson")

        image = self.read_image(image_path)

        mask = cv2.imread(os.path.dirname(image_path)+'/masks/'+os.path.basename(image_path) +'.mask.png')[:,:,0]/255

        #mask_dil = morphology.dilation(mask,selem=morphology.square(3))[:,:,None]
        mask = mask[:,:,None]

        #mask = self.read_layout(json_path, image.shape[:2])

        sample = {'image': image.transpose((2,0,1)),
                  'mask': mask.transpose((2,0,1)),
                  'path':image_path}

        if self.provide_orig_mask:
            sample['mask_orig'] = mask.transpose((2,0,1)).copy()


        return sample

    def __len__(self):
        return self.epoch_size


class EyeDatasetInfer(Dataset):
    """
    Класс датасета, организующий загрузку и получение изображений и соответствующих разметок
    """

    def __init__(self, data_folder: str, images: list = None):
        self.class_ids = {"vessel": 1}

        self.data_folder = data_folder

        if images is not None:
            self._image_files = [f"{data_folder}/{f}.png" for f in images]
        else:
            self._image_files = glob.glob(f"{data_folder}/*.png")


    @staticmethod
    def read_image(path: str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image / 255, dtype=np.float32)
        return image

    def __getitem__(self, idx: int) -> dict:
        # Достаём имя файла по индексу
        image_path = self._image_files[idx]

        image = self.read_image(image_path)


        sample = {'image': image.transpose((2,0,1)),
                  'path':image_path,
                  'mask_orig':np.zeros((1,image.shape[0],image.shape[1]))}

        return sample

    def __len__(self):
        return len(self._image_files)