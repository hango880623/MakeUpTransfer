import time
from PIL import Image
import requests
import base64
import io
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By

import os
import shutil

def crawl_image():
    images_path = '/Users/kuyuanhao/Documents/Crawl/face/'  # enter your desired image path

    options = ChromeOptions()
    options.add_argument("--start-maximized")
    # options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    driver = webdriver.Chrome(options=options)

    # url = ("https://www.google.com/search?q={s}&tbm=isch&tbs=sur%3Afc&hl=en&ved=0CAIQpwVqFwoTCKCa1c6s4-oCFQAAAAAdAAAAABAC&biw=1251&bih=568")
    url = ("https://www.google.com/search?q={s}&sca_esv=e4952731b002d90d&sca_upv=1&rlz=1C5CHFA_enTW1017TW1018&udm=2&biw=2327&bih=1176&sxsrf=ACQVn08RjN3H3qOv_v34xsgpEdx9WGPMaA%3A1712730600095&ei=6DEWZpO1BYvw0PEP1ciy0AY&oq=&gs_lp=Egxnd3Mtd2l6LXNlcnAiACoCCAAyBxAjGOoCGCcyBxAjGOoCGCcyBxAjGOoCGCcyBxAjGOoCGCcyBxAjGOoCGCcyBxAjGOoCGCcyBxAjGOoCGCcyBxAjGOoCGCcyBxAjGOoCGCcyBxAjGOoCGCdI5wxQAFgAcAF4AJABAJgBAKABAKoBALgBAcgBAPgBAZgCAaACE6gCCpgDE5IHATGgBwA&sclient=gws-wiz-serp")
    key = 'light skin color makeup'
    key = key.replace(' ', '+')
    driver.get(url.format(s=key))

    for x in range(10):
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
        time.sleep(2)

    imgResults = driver.find_elements(By.XPATH,"//img[contains(@class,'YQ4gaf')]")

    src = [img.get_attribute('src') for img in imgResults]
    print(len(src))
    key = key.replace('+', '_')
    for i in range(len(src)):
        # check if the image is None
        if src[i] is None:
            pass
        else:
            # if it's base64 images
            if src[i].startswith('data'):
                imgdata = base64.b64decode(str(src[i]).split(',')[1])
                img = Image.open(io.BytesIO(imgdata))
                img.save(images_path+key+"{}1.png".format(i))
            # if it's image url
            else:
                img = Image.open(requests.get(src[i], stream=True).raw).convert('RGB')
                img.save(images_path+key+"{}.png".format(i))

import os
import shutil

def move_images(src, trg):
    # Define the mapping of suffixes to target directories
    classes = ['55', '60', '65', '70', '80']
    
    # Create target directories if they don't exist
    if not os.path.exists(trg):
        os.makedirs(trg)
        for dir_name in classes:
            os.makedirs(os.path.join(trg, 'lips', dir_name))
    
    # Move images to appropriate directories
    for file in os.listdir(src):
        if file.endswith('.jpg'):
            print(file)
            suffix = file.split('-')[1]
            if suffix in classes:
                target_dir = os.path.join(trg, 'lips', suffix)
                shutil.move(os.path.join(src, file), os.path.join(target_dir, file))


if __name__ == '__main__':
    src = './cagan_Data0421_new/test'
    trg = './cagan_Data0421_new/#6Lips'
    move_images(src,trg)