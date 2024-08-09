import dlib
import cv2
import numpy as np

def image_lab2bgr(image,color):
    if(image.shape[-1] != 3):
        image = image.astype(np.float32)
        image = np.transpose(image, (1, 2, 0))
        updated_image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)  * 255.
        updated_image = np.transpose(updated_image, (2, 0, 1))
        return updated_image
    return cv2.cvtColor(image, cv2.COLOR_LAB2)  * 255 

def lipstick_color(index):
    lips_color_dic = {0: [[30, 27, 13],[24, 25, 13],[26, 24, 10],[32, 33, 15],[22, 17, 5]],     # Shuyi
                      1: [[32, 47, 42],[35, 49, 45],[36, 48, 44],[36, 47, 46],[35, 41, 42]],    # Katia
                      2: [[40, 48, 46],[33, 42, 40],[31, 41, 39],[32, 39, 39],[31, 38, 37]],    # Howard
                      3: [[26, 40, 34],[30, 46, 40],[27, 39, 35],[26, 42, 36],[18, 28, 24]],    # Nicole
                      4: [[42, 48, 22],[30, 45, 22],[39, 46, 16],[48, 52, 23],[35, 35, 8]],     # Shuyi-1
                      5: [[10, 24, 9],[6, 19, 6],[6, 13, 3],[9, 21, 8],[8, 22, 7]],}    # Base
    try:
        return lips_color_dic[index]
    except KeyError:
        print(f"Invalid index: {index}. Valid indices are: {list(lips_color_dic.keys())}")
        return None

