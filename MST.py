import numpy as np
import math
import pandas as pd
from skimage.color import rgb2lab
import matplotlib.pyplot as plt

import dlib
import cv2
import pandas as pd
import numpy as np

import os

import csv
from tqdm import tqdm

monk_skin_tones = {
        "Monk 01": {"hex": "#f6ede4", "rgb": [246, 237, 228]},
        "Monk 02": {"hex": "#f3e7db", "rgb": [243, 231, 219]},
        "Monk 03": {"hex": "#f7ead0", "rgb": [247, 234, 208]},
        "Monk 04": {"hex": "#eadaba", "rgb": [234, 218, 186]},
        "Monk 05": {"hex": "#d7bd96", "rgb": [215, 189, 150]},
        "Monk 06": {"hex": "#a07e56", "rgb": [160, 126, 86]},
        "Monk 07": {"hex": "#825c43", "rgb": [130, 92, 67]},
        "Monk 08": {"hex": "#604134", "rgb": [96, 65, 52]},
        "Monk 09": {"hex": "#3a312a", "rgb": [58, 49, 42]},
        "Monk 10": {"hex": "#292420", "rgb": [41, 36, 32]}
    }

def show_monk_skin_tone_distribution(df, save_path):
    skin_tone_counts = df['Monk_Skin_Tone'].value_counts()
    print(skin_tone_counts)
    skin_tone_counts = skin_tone_counts.sort_index()
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    skin_tone_counts.plot(kind='bar', color='skyblue')
    plt.title('Occurrences of Monk Skin Tones')
    plt.xlabel('Monk Skin Tone')
    plt.ylabel('Occurrences')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)  # Save the plot
        print(f"Plot saved as {save_path}")
    else:
        plt.show()

def find_nearest_color(input_color_lab, target_colors_lab):
    min_distance = float('inf')
    nearest_color_index = None
    for i, target_color_lab in enumerate(target_colors_lab):
        distance = color_distance(input_color_lab, target_color_lab.tolist())
        if distance < min_distance:
            min_distance = distance
            nearest_color_index = i
    
    return nearest_color_index, target_colors_lab[nearest_color_index]

def color_distance(color1, color2):
    # color1 = [int(x) for x in color1.strip('[]').split(',')]
    l1, a1, b1 = int(color1[0]), int(color1[1]), int(color1[2])
    l2, a2, b2 = color2
    return math.sqrt((l2 - l1)**2 + (a2 - a1)**2 + (b2 - b1)**2)

def get_lips_color(image, predictor, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(gray, face)
        lips = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 61)]
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.fillPoly(mask, [np.array(lips)], (255, 255, 255))
        
        # Compute LAB color space
        image_lab = cv2.cvtColor(np.float32(image)/ 255., cv2.COLOR_BGR2LAB)
        # Compute median color from the extracted pixel values within the mask
        median_color_lab = np.median(image_lab[np.where(mask == 255)], axis=0)
        median_color_lab = [int(median_color_lab[0]), int(median_color_lab[1]), int(median_color_lab[2])]

        # Compute RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Compute median color from the extracted pixel values within the mask
        median_color_rgb = np.median(image_rgb[np.where(mask == 255)], axis=0)
        median_color_rgb = [int(median_color_rgb[0]), int(median_color_rgb[1]), int(median_color_rgb[2])]

    return median_color_lab, median_color_rgb

def get_skin_color(image, predictor, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(gray, face)
        facepts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(1, 16)]
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.fillPoly(mask, [np.array(facepts)], (255, 255, 255))
        # Exclude lips region (landmarks 48-60)
        lips = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 61)]
        cv2.fillPoly(mask, [np.array(lips)], (0, 0, 0))

        # Compute LAB color space
        image_lab = cv2.cvtColor(np.float32(image)/ 255., cv2.COLOR_BGR2LAB)
        # Compute median color from the extracted pixel values within the mask
        median_color_lab = np.median(image_lab[np.where(mask == 255)], axis=0)
        median_color_lab = [int(median_color_lab[0]), int(median_color_lab[1]), int(median_color_lab[2])]

        # Compute RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Compute median color from the extracted pixel values within the mask
        median_color_rgb = np.median(image_rgb[np.where(mask == 255)], axis=0)
        median_color_rgb = [int(median_color_rgb[0]), int(median_color_rgb[1]), int(median_color_rgb[2])]
    
    return median_color_lab, median_color_rgb

def face_check(image, predictor, detector):
    # Load the pre-trained face detector
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) != 1:
        return False
    return True

def label_color(attr_list, image_dir):

    """Preprocess the Makeup Transfer file."""
    all_file_names = attr_list
    clean_file_names = []
 
    dataset = []
    # Load the pre-trained face detector
    predictor = dlib.shape_predictor("./pretrained_model/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()

    for i, filename in enumerate(tqdm(all_file_names)):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        if not face_check(image, predictor, detector):
            print('No face or multiple faces detected in image:', image_path)
            os.remove(image_path)
            continue
        # Get lips and skin color 
        cm_label_lab, cm_label_rgb = get_lips_color(image, predictor, detector)
        cs_label_lab, cs_label_rgb = get_skin_color(image, predictor, detector)

        # Calculate Monk Skin Tone
        target_colors_lab = [skin_tone['rgb'] for skin_tone in monk_skin_tones.values()] 
        target_colors_lab = [rgb2lab([x / 255. for x in color]) for color in target_colors_lab] # Get monk skin tone lab

        nearest_monk_index, nearest_monk_color_lab = find_nearest_color(cs_label_lab, target_colors_lab) # Find nearset monk skin tone
        nearest_monk_name = list(monk_skin_tones.keys())[nearest_monk_index]

        label = [filename, cm_label_lab, cs_label_lab, cm_label_rgb, cs_label_rgb, nearest_monk_name, nearest_monk_color_lab]
        dataset.append(label)
        clean_file_names.append(filename)

    print('Finished preprocessing the Makeup Transfer dataset...')

    return clean_file_names, dataset

if __name__ == '__main__':
    base_path = '/Users/kuyuanhao/Documents/Customized/'
    attr_path = base_path + 'clean.txt'
    image_dir = base_path
    label_path = base_path + 'label.csv'
    # label_color(attr_path, image_dir, label_path)
    # train_file ="data/mt/train_class.csv"
    df = pd.read_csv(label_path)
    save_dir = base_path + 'class.csv'
    show_monk_skin_tone_distribution(df)
    

    
