import os
import dlib
import cv2
import numpy as np
from tqdm import tqdm

def clean_folder(image_dir):
    all_file_names = os.listdir(image_dir)
    
    clean_file_names = []
    for file_name in all_file_names:
        # Check if the file ends with '.jpg' or '.png'
        if file_name.lower().endswith(('.jpg','.jepg', '.png','.JPG','.JEPG', 'PNG')):
            clean_file_names.append(file_name)
        else:
            os.remove(os.path.join(image_dir, file_name))
    return clean_file_names

def get_lips_bg(image, predictor, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(gray, face)
        lips = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 61)]
        

        height, width, _ = image.shape
        # Separate x and y coordinates
        x_coords = [point[0] for point in lips]
        y_coords = [point[1] for point in lips]

        # Calculate the average x and y coordinates
        avg_x = np.mean(x_coords) + 5
        avg_y = np.mean(y_coords)

        # Define the cropping boundaries
        x1 = max(int(avg_x) - 64, 0)
        y1 = max(int(avg_y) - 32, 0)
        x2 = min(int(avg_x) + 64, width - 1)
        y2 = min(int(avg_y) + 32, height - 1)
        cropped_image = image[y1:y2, x1:x2]
        
        # Calculate the cropped image background color
        mask = np.zeros(image.shape[:2], dtype="uint8")
        mask[y1:y2, x1:x2] = (255, 255, 255)
        cv2.fillPoly(mask, [np.array(lips)], (0, 0, 0))

        # Compute LAB color space
        image_lab = cv2.cvtColor(np.float32(image)/ 255., cv2.COLOR_BGR2LAB)
        # Compute median color from the extracted pixel values within the mask
        median_color_lab = np.median(image_lab[np.where(mask == 255)], axis=0)
        median_color_lab = [int(median_color_lab[0]), int(median_color_lab[1]), int(median_color_lab[2])]

    return median_color_lab

def get_lips(image_path):
    # Load the pre-trained face detector
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(image_path)
    if image is None:
        print('Image not found:', image_path)
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) == 0:
        # print('No face detected in image:', image_path)
        return False
    return True

def crop_lips(image_dir, target_dir, file_name):
    try:
        # Load the pre-trained face detector
        detector = dlib.get_frontal_face_detector()

        # Load the facial landmark predictor
        predictor = dlib.shape_predictor("./pretrained_model/shape_predictor_68_face_landmarks.dat")

        # Read the image
        image_path = image_dir + file_name
        image = cv2.imread(image_path)
        if image is None:
            print('Image not found:', image_path)
            return [0,0,0,0]
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
            avg_x = np.mean(x_coords) + 5
            avg_y = np.mean(y_coords)

            # Draw the square on the image
            x1 = int(avg_x) - 64
            y1 = int(avg_y) - 32
            x2 = int(avg_x) + 64
            y2 = int(avg_y) + 32
            cropped_image = image[y1:y2, x1:x2]
            if cropped_image.size == 0:
                print('Cropped image is empty:', image_path)
                continue
            cv2.imwrite(target_dir + '/' + file_name, cropped_image)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Or save the image
            # Draw the lips on the image
            # cv2.polylines(image, [np.array(lips)], isClosed=True, color=(0, 255, 0), thickness=2)
            # cv2.imwrite("./output_image_with_lips.jpg", image)
        return [y1,y2,x1,x2]
    except cv2.error as e:
            print('OpenCV Error:', e)
            return [0, 0, 0, 0]
            
def clean_file(dir, attribute_path = None):
    image_dir = dir+'image/'
    if attribute_path is None:
        all_file_names = os.listdir(image_dir)
    else:
        all_file_names = [line.rstrip() for line in open(attribute_path, 'r')]
    clean_file_names = []
    for i, filename in enumerate(tqdm(all_file_names)):
        image_path = image_dir+filename
        if get_lips(image_path):
            clean_file_names.append(filename)
        else:
            print('Lips not detected in image:', image_path)
            os.remove(image_path)

    # Write clean filenames to the new attribute file makeup_clean.txt
    with open(dir+'clean.txt', 'w') as file:
        for filename in clean_file_names:
            file.write(filename + '\n')

def build_crop_data(attribute_path, dir, target_dir):
    image_dir = dir+'image/'
    all_file_names = [line.rstrip() for line in open(attribute_path, 'r')]
    crop_file_coordinate = {}
    for i, filename in enumerate(tqdm(all_file_names)):
        coordinate = crop_lips(image_dir, target_dir, filename)
        crop_file_coordinate[filename] = coordinate

    # Write coordinates to a text file
    with open(dir+'non-makeup_crop_coordinate.txt', 'w') as file:
        for filename, coordinate in crop_file_coordinate.items():
            file.write(f"{filename}: {coordinate}\n")

if __name__ == '__main__':
    image_dir = './data/Data0506/'
    # attribute_path = './data/mt/makeup_clean.txt'
    # image_dir = '/Users/kuyuanhao/Documents/LABImage/'
    # clean_file(image_dir)

    attribute_path = './data/Data0506/clean.txt'
    target_dir = './data/Data0506/image_cropped'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    build_crop_data(attribute_path, image_dir,target_dir)
    


             