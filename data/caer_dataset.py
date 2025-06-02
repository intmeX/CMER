import ast
import numpy as np
import cv2
import os
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets


class CAERDataset(Dataset):
    def __init__(self, root, input_file, context_transform, face_transform, context_norm, face_norm):
        self.root = root
        self.context_transform = context_transform
        self.face_transform = face_transform
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.face_norm = transforms.Normalize(face_norm[0], face_norm[1])
        self.contexts = []
        self.faces = []
        self.labels = []
        self._preprocess(input_file)

    def _preprocess(self, file):
        logger = logging.getLogger('Experiment')
        data = [line.rstrip('\n') for line in open(file)]
        mode = file.split('\\')[-1].split('/')[-1].split('.')[0]
        context_path = os.path.join(self.root, 'context_{}.npy'.format(mode))
        face_path = os.path.join(self.root, 'face_{}.npy'.format(mode))
        label_path = os.path.join(self.root, 'label_{}.npy'.format(mode))
        if not os.path.exists(context_path):
            for i in range(len(data)):
                sample = data[i].split(',')
                path, label, x1, y1, x2, y2 = os.path.join(self.root, sample[0]), int(sample[1]), int(sample[2]), int(
                    sample[3]), int(sample[4]), int(sample[5])
                try:
                    context = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                    x1 = max(0, x1)
                    x2 = max(0, x2)
                    y1 = max(0, y1)
                    y2 = max(0, y2)
                    if y1 > y2:
                        y1, y2 = y2, y1
                    if x1 > x2:
                        x1, x2 = x2, x1
                    face = context[y1: y2, x1: x2].copy()
                    context = cv2.resize(context, (224, 224))
                    face = cv2.resize(face, (48, 48))
                    # one_hot = np.zeros(7, dtype=float)
                    # one_hot[label] = 1.0
                    self.contexts.append(context)
                    self.faces.append(face)
                    self.labels.append(label)
                except Exception as e:
                    logger.info(path)
                    raise e
                if (i + 1) % 100 == 0:
                    logger.info('[caer process] {}/{} images processd'.format(i + 1, len(data)))
            self.contexts = np.array(self.contexts)
            self.faces = np.array(self.faces)
            self.labels = np.array(self.labels)
            np.save(context_path, self.contexts)
            np.save(face_path, self.faces)
            np.save(label_path, self.labels)
        else:
            self.contexts = np.load(context_path)
            self.faces = np.load(face_path)
            self.labels = np.load(label_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if idx >= len(self.labels):
            raise IndexError('Index out of bound')
        context = self.contexts[idx]
        face = self.faces[idx]
        label = self.labels[idx]
        return (
            self.context_norm(self.context_transform(context)),
            self.face_norm(self.face_transform(face)),
            label,
        )
