import os
import io
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer



class T2IGANDataset(Dataset):
    def __init__(self, dataset_file, split="train", emb_type='default'):
        self.dataset_file = dataset_file
        self.split = split
        self.dataset = h5py.File(self.dataset_file, mode='r')[self.split]
        self.img_names, self.dataset_len = self.get_img_names()

        self.emb_type = emb_type
        if emb_type == 'default':
            self.embed_dim = 1024
        elif emb_type == 'all-mpnet-base-v2':
            self.model = SentenceTransformer(emb_type)
            self.embed_dim = 768
        elif emb_type =='all-distilroberta-v1':
            self.model = SentenceTransformer(emb_type)
            self.embed_dim = 768
        elif emb_type == 'all-MiniLM-L12-v2':
            self.model = SentenceTransformer(emb_type)
            self.embed_dim = 384

       


    def tokenizer(self, x):
        if self.emb_type == 'default':
            return torch.FloatTensor(np.array(x['embeddings'], dtype=float))

        txt = str(np.array(x['txt']).astype(str))
        return self.model.encode(txt, convert_to_tensor=True)
            

    def get_img_names(self):
        img_names = [str(k) for k in self.dataset.keys()]
        length = len(self.dataset)
        return img_names, length

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        example_name = self.img_names[item]
        example = self.dataset[example_name]
        text = np.array(example['txt']).astype(str)

        right_image = bytes(np.array(example['img']))
       
        right_embed = self.tokenizer(example)
        inter_embed = self.tokenizer(self.__find_inter_embed())
        

        wrong_image = bytes(np.array(self.__find_wrong_image(example['class'])))


        right_image = self.__validate_image(Image.open(io.BytesIO(right_image)).resize((64, 64)))
        wrong_image = self.__validate_image(Image.open(io.BytesIO(wrong_image)).resize((64, 64)))
        

        return {
            'right_images': torch.FloatTensor(right_image),
            'right_embed': right_embed,
            'wrong_images': torch.FloatTensor(wrong_image),
            'inter_embed': inter_embed,
            'txt': str(text),
        }

    def __find_wrong_image(self, category):
        _c = category
        while _c == category:
            item = np.random.randint(len(self.img_names))
            example_name = self.img_names[item]
            example = self.dataset[example_name]
            _c = example['class']
        return example['img']

    def __find_inter_embed(self):
        idx = np.random.randint(len(self.img_names))
        example_name = self.img_names[idx]
        example = self.dataset[example_name]
        return example

    def __validate_image(self, img):
        img = np.array(img, dtype=float)
        if np.max(img) > 1:
            img = img / 255
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = T2IGANDataset(dataset_file="data/flowers.hdf5", split="train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for a in dataloader:
        print(a['right_embed'].shape)
        break

