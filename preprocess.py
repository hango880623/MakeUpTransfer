from clean import clean_file, build_crop_data
from MST import label_color, show_monk_skin_tone_distribution
import pandas as pd
import cv2
import os
import shutil

from clean import clean_folder

def preprocess_folder(source_folder,target_folder):
    target_image_folder = os.path.join(target_folder, 'image/')
    source_image_folder = os.path.join(source_folder, 'image/')
    if not os.path.exists(target_image_folder):
        os.makedirs(target_image_folder)
        print(f"Created directory: {target_image_folder}")

    attribute_list = clean_folder(source_image_folder)
    clean_file_names, dataset = label_color(attribute_list, source_image_folder)
    # Convert dataset list to pandas DataFrame
    df = pd.DataFrame(dataset, columns=['Filename', 'Lips_Color', 'Skin_Color', 'Lips_Color_RGB', 'Skin_Color_RGB', 'Monk_Skin_Tone', 'Monk_Skin_Tone_Color'])
    df.to_csv(os.path.join(target_folder, 'label.csv'), index=False)
    print('Dataset saved to', os.path.join(target_folder, 'label.csv'))

     # Write clean filenames to the new attribute file makeup_clean.txt
    with open(os.path.join(target_folder, 'clean.txt'), 'w') as file:
        for filename in clean_file_names:
            file.write(filename + '\n')
    
    show_monk_skin_tone_distribution(df,os.path.join(target_folder, 'monk.png'))

def rename_images(folder_path):
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif','webp'))]
    
    # Sort the image files alphabetically
    image_files.sort()
    
    # Rename each image file with an ordered number
    for i, image_file in enumerate(image_files):
        _, ext = os.path.splitext(image_file)
        new_name = f"y_image_{i+1}.png"
        os.rename(os.path.join(folder_path, image_file), os.path.join(folder_path, new_name))
        print(f"Renamed {image_file} to {new_name}")

def copy_file(csv_path, source_folder, target_folder):
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        file_name = row['Filename']
        source_path = os.path.join(source_folder, file_name)
        target_path = os.path.join(target_folder, file_name)
        shutil.copy(source_path, target_path)
        print(f"Copied {file_name} from {source_path} to {target_path}")
        
def change_image_extensions(folder_path):
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith('.jpg')]
    
    # Change the file extensions
    for image_file in image_files:
        old_path = os.path.join(folder_path, image_file)
        new_path = os.path.join(folder_path, os.path.splitext(image_file)[0] + '.JPG')
        os.rename(old_path, new_path)
        print(f"Changed {image_file} to {os.path.basename(new_path)}")

if __name__ == '__main__':
    base_dir = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Pixel/0116/lips_merge'
    target_dir = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Pixel/0116/lips_merge'
    preprocess_folder(base_dir,target_dir)

    # source = '/Users/kuyuanhao/Documents/nicole/image'
    # target = './data/Data0421/new'
    # copy_file('/Users/kuyuanhao/Documents/nicole/new_n.csv', source, target)

    # folder_path = "/Users/kuyuanhao/Documents/lips_all"
    # change_image_extensions(folder_path)

