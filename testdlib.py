import dlib
import cv2
import numpy as np
import os

from torchvision import transforms as T
from data_loader import MT

import csv
def get_color(image_path):
    image = cv2.imread(image_path)

    # Compute LAB color space
    image_lab = cv2.cvtColor(np.float32(image)/ 255., cv2.COLOR_BGR2LAB)
    # Compute median color from the extracted pixel values within the mask
    mask = np.zeros(image.shape[:2], dtype="uint8")
    median_color_lab = np.median(image_lab[np.where(mask == 0)], axis=0)
    median_color_lab = [int(median_color_lab[0]), int(median_color_lab[1]), int(median_color_lab[2])]


    return median_color_lab

def get_skin_color(image_path):
    # Load the pre-trained face detector
    detector = dlib.get_frontal_face_detector()

    # Load the facial landmark predictor
    predictor = dlib.shape_predictor("./pretrained_model/shape_predictor_68_face_landmarks.dat")

    # Read the image
    image = cv2.imread(image_path)
    

    # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(image)

    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(image, face)
        
        # Extract face points (usually landmarks 1-15)
        facepts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(1, 16)]

        # Create a mask for the face
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.fillPoly(mask, [np.array(facepts)], (255, 255, 255))

        # Exclude lips region (landmarks 48-60)
        lips = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 61)]
        cv2.fillPoly(mask, [np.array(lips)], (0, 0, 0))
        
        # Compute average color of the face
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        pixel_values_lab = image_lab[np.where(mask == 255)]

        median_color_lab = np.median(pixel_values_lab, axis=0)
        print("Median LAB color of the skin:", median_color_lab)

        pixel_values = image[np.where(mask == 255)]

        median_color = np.median(pixel_values, axis=0)
        

        # Convert to integer BGR values
        median_color_bgr = (int(median_color[0]), int(median_color[1]), int(median_color[2]))

        print("Average RGB color of the skin:", median_color_bgr)

        # Create a new image filled with the average color
        color_image = np.full_like(image, median_color_bgr, dtype=np.uint8)
        cv2.imwrite("./rgb_color_skin.jpg", color_image)

        # Draw the face contour on the image
        cv2.polylines(image, [np.array(facepts)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(image, [np.array(lips)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Save the image
        cv2.imwrite("./output_image_with_face.jpg", image)
    return median_color_bgr


def get_lips_color(image_path):
    # Load the pre-trained face detector
    detector = dlib.get_frontal_face_detector()

    # Load the facial landmark predictor
    predictor = dlib.shape_predictor("./pretrained_model/shape_predictor_68_face_landmarks.dat")

    # Read the image
    image = cv2.imread(image_path)
    print(image.shape)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)
    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(gray, face)

        # Extract lip points (usually landmarks 48-60)
        lips = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 61)]
        # Separate x and y coordinates
        x_coords = [point[0] for point in lips]
        y_coords = [point[1] for point in lips]

        # Calculate the average x and y coordinates
        avg_x = np.mean(x_coords) +5
        avg_y = np.mean(y_coords)
        print(avg_x, avg_y)

        # Create a mask for the lips
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(lips)], (255, 255, 255))
        
        # Compute average color of the lips
        average_color = cv2.mean(image, mask=mask)

        # Convert to integer BGR values
        average_color_bgr = (int(average_color[0]), int(average_color[1]), int(average_color[2]))

        print("Average BGR color of the lips:", average_color_bgr)

        # Create a new image filled with the average color
        color_image = np.full_like(image, average_color_bgr, dtype=np.uint8)
        cv2.imwrite("./rgb_color_lips.jpg", color_image)
        # Draw the square on the image
        rec_1 = (int(avg_x) - 56, int(avg_y) - 28)
        rec_2 = (int(avg_x) + 56, int(avg_y) + 28)
        x1 = int(avg_x) - 64
        y1 = int(avg_y) - 32
        x2 = int(avg_x) + 64
        y2 = int(avg_y) + 32
        cropped_image = image[y1:y2, x1:x2]
        cv2.imwrite("./cropped_image.jpg", cropped_image)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Or save the image
        # Draw the lips on the image
        cv2.polylines(image, [np.array(lips)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imwrite("./output_image_with_lips.jpg", image)

    if len(faces) == 0:
        print("No face detected")
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Calculate the size of the square
        square_size = min(height, width) // 5

        # Calculate the coordinates of the square
        top_left_x = (width - square_size) // 2
        top_left_y = (height - square_size) // 2 + square_size
        bottom_right_x = top_left_x + square_size//2
        bottom_right_y = top_left_y + square_size//2
        # Extract the region of interest (ROI) corresponding to the square area
        square_roi = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        # Draw the square on the image
        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
        cv2.imwrite("./output_image_with_lips.jpg", image)

        # Compute the mean color of the ROI
        average_color = cv2.mean(square_roi)
        # Convert to integer RGB values
        average_color_bgr = (int(average_color[0]), int(average_color[1]), int(average_color[2]))
        # Create a new image filled with the average color
        color_image = np.full_like(image, average_color_bgr, dtype=np.uint8)
        cv2.imwrite("./rgb_color_lips.jpg", color_image)
            
    print(average_color_bgr)
    return average_color_bgr

def save_filenames_to_txt(path, txt_file):
    # Get the list of filenames in the specified directory
    filenames = os.listdir(path)

    # Write the filenames to the text file
    with open(txt_file, 'w') as file:
        for filename in filenames:
            file.write(filename + '\n')

def test_MT():
    # Create a dataset object
    transform = []
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(256))
    transform.append(T.Resize(128))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    dataset = MT(image_dir="./mtdataset/images/makeup", attr_path="./mtdataset/makeuptest.txt",transform = transform,mode = "train")
                 
def verify_label(train_label):
    # Load train dataset
    base_path ="./data/mt/images/makeup"
    train_dataset = []
    with open(train_label, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            lips_colors = [int(x) for x in row[1].strip('[]').split(',')]
            skin_colors = [int(x) for x in row[2].strip('[]').split(',')]
            train_dataset.append([row[0],[lips_colors,skin_colors]])
    image_list = []
    image_org = []
    for file_name,colors in train_dataset[10:20]:
        image_path = os.path.join(base_path,file_name)
        image = cv2.imread(image_path)
        color_image = np.full_like(image, colors[0], dtype=np.float32)
        rbg_image = cv2.cvtColor(color_image, cv2.COLOR_LAB2BGR) * 255.
        image_list.append(rbg_image)
        image_org.append(image)
    image_cat = np.concatenate(image_list, axis=0)
    image_cat_org = np.concatenate(image_org, axis=0)
    cv2.imwrite("./rgb_color_lips_strip.jpg", np.concatenate([image_cat_org,image_cat], axis=1))

if __name__ == '__main__':
    # get_lips_color("./data/LABImage/p_1_55_3000.JPG")
    folder = "/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Canon/0502/target color"
    files = ['s_55_5000.JPG','s_60_5000.JPG','s_65_5000.JPG','s_70_5000.JPG','s_80_5000.JPG']
    for file in files:
        lab = get_color(os.path.join(folder,file))
        print(lab)
    