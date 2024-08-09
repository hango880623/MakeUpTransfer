from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
from sklearn.model_selection import train_test_split

import dlib
import cv2
import numpy as np

import csv
from tqdm import tqdm

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class MT(data.Dataset):
    """Dataset class for the Makeup Transfer dataset."""

    def __init__(self, image_dir, attr_path,train_label, test_label, transform, mode):
        """Initialize and preprocess the Makeup Transfer dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.transform = transform
        self.train_label = train_label
        self.test_label = test_label
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.predictor = dlib.shape_predictor("./pretrained_model/shape_predictor_68_face_landmarks.dat")
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def get_lips_color(self, image_path):
        # Load the pre-trained face detector
        detector = dlib.get_frontal_face_detector()
        image = cv2.imread(image_path)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        for face in faces:
            # Predict facial landmarks
            landmarks = self.predictor(gray, face)
            lips = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 61)]
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.fillPoly(mask, [np.array(lips)], (255, 255, 255))
            
            image_lab = cv2.cvtColor(np.float32(image)/ 255., cv2.COLOR_BGR2LAB)
            # Extract pixel values within the mask
            pixel_values = image_lab[np.where(mask == 255)]
            # Compute median color
            median_color_lab = np.median(pixel_values, axis=0)
            median_color_lab = [int(median_color_lab[0]), int(median_color_lab[1]), int(median_color_lab[2])]

        if len(faces) == 0:
            print('No face detected in image:', image_path)
            return [int(0), int(0), int(0)]

        return median_color_lab
    
    def get_skin_color(self, image_path):
        # Load the pre-trained face detector
        detector = dlib.get_frontal_face_detector()
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        for face in faces:
            # Predict facial landmarks
            landmarks = self.predictor(gray, face)
            facepts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(1, 16)]
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.fillPoly(mask, [np.array(facepts)], (255, 255, 255))
            # Exclude lips region (landmarks 48-60)
            lips = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 61)]
            cv2.fillPoly(mask, [np.array(lips)], (0, 0, 0))

            image_lab = cv2.cvtColor(np.float32(image)/ 255., cv2.COLOR_BGR2LAB)
            # Extract pixel values within the mask
            pixel_values = image_lab[np.where(mask == 255)]
            # Compute median color
            median_color_lab = np.median(pixel_values, axis=0)
            median_color_lab = [int(median_color_lab[0]), int(median_color_lab[1]), int(median_color_lab[2])]
        if len(faces) == 0:
            print('No face detected in image:', image_path)
            return [int(0), int(0), int(0)]
        return median_color_lab

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_file_names = lines
        for i, file_name in enumerate(all_file_names):
            self.attr2idx[file_name] = i
            self.idx2attr[i] = file_name
            
        if os.path.exists(self.train_label) and os.path.exists(self.test_label):
            print('Loading existing datasets...')
            self.load_datasets()
            print('Datasets loaded successfully.')
            return

        """Preprocess the Makeup Transfer file."""

        random.seed(1234)
        train_files, test_files = train_test_split(all_file_names, test_size=0.2, random_state=1234)
        for i, filename in enumerate(tqdm(all_file_names)):

            cm_label = self.get_lips_color(self.image_dir+'/'+filename)
            cs_label = self.get_skin_color(self.image_dir+'/'+filename)
            label = [cm_label, cs_label]
            if filename in test_files:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
        print('Finished preprocessing the Makeup Transfer dataset...')
        # Save the datasets
        self.save_datasets()
        print('Datasets saved successfully.')

    def save_datasets(self):
        # Save train dataset
        with open(self.train_label, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Lips_Color', 'Skin_Color'])
            for item in self.train_dataset:
                writer.writerow([item[0], *[list(color) for color in item[1]]])

        # Save test dataset
        with open(self.test_label, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Lips_Color', 'Skin_Color'])
            for item in self.test_dataset:
                writer.writerow([item[0], *[list(color) for color in item[1]]])

    def load_datasets(self):
        # Load train dataset
        with open(self.train_label, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                lips_colors = [int(x) for x in row[1].strip('[]').split(',')]
                skin_colors = [int(x) for x in row[2].strip('[]').split(',')]
                self.train_dataset.append([row[0],[lips_colors,skin_colors]])

        # Load test dataset
        with open(self.test_label, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                lips_colors = [int(x) for x in row[1].strip('[]').split(',')]
                skin_colors = [int(x) for x in row[2].strip('[]').split(',')]
                self.test_dataset.append([row[0],[lips_colors,skin_colors],row[3]])

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if self.mode == 'train':
            dataset = self.train_dataset
            filename, label = dataset[index]
            image = Image.open(os.path.join(self.image_dir, filename))
            return self.transform(image), torch.FloatTensor(label)
        else:
            dataset = self.test_dataset
            filename, label, skintone = dataset[index]
            image = Image.open(os.path.join(self.image_dir, filename))
            return self.transform(image), torch.FloatTensor(label), filename, skintone
        

    def __len__(self):
        """Return the number of images."""
        return self.num_images
    

def get_loader(image_dir, attr_path, train_label, test_label, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    # transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize((image_size,image_size)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'MT':
        dataset = MT(image_dir, attr_path, train_label, test_label, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader