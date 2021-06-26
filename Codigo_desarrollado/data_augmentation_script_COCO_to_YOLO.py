# Usage: python3 imgaug_test_final.py --images_path /home/sergioballines/tests/others/23_images/ --labels_path /home/sergioballines/tests/others/23_labels/

import imgaug as ia
import imageio
import random
import glob
import os
import shutil
import argparse
import sys
from imgaug import augmenters as iaa
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument("--images_path", type=str, default="", nargs='?', help="Path of the images for data augmentation")
parser.add_argument("--labels_path", type=str, default="", nargs='?', help="Path of the labels of the input images")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

#images_main_path = '/home/dev/data_source/cam_mandela/candidates/2020/10/23/'               #Cambiar según la carpeta

images_main_path = args.images_path

images = glob.glob(images_main_path + "*.jpeg") + glob.glob(images_main_path + "*.jpg") + glob.glob(images_main_path + "*.JPG")

#labels = "/home/dev/data_source/cam_mandela/labels/2020/10/23/"                             #Cambiar según la carpeta

labels = args.labels_path

def fog_image(img):
    image = imageio.imread(img)
    fog = iaa.Fog()
    image_aug = fog(image=image)
    image_aug = Image.fromarray(image_aug)

    image_name = os.path.splitext(img)[0]
    basename_image = os.path.basename(image_name)
    img_extension = os.path.splitext(img)[1]
    if not os.path.exists(images_main_path + 'fog/'):
        os.makedirs(images_main_path + 'fog/')
    images_save = images_main_path + 'fog/'
    image_aug.save(images_save + basename_image + "_fog" + img_extension)
    
    label = labels + basename_image
    if not os.path.exists(labels + 'fog/'):
        os.makedirs(labels + 'fog/')
    labels_save = labels + 'fog/'
    label_aug = labels_save + basename_image
    shutil.copyfile(label + ".txt", label_aug + "_fog.txt")
    
def rain_image(img):
    image = imageio.imread(img)
    rain = iaa.Rain()
    image_aug = rain(image=image)
    image_aug = Image.fromarray(image_aug)
    
    image_name = os.path.splitext(img)[0]
    basename_image = os.path.basename(image_name)
    img_extension = os.path.splitext(img)[1]
    if not os.path.exists(images_main_path + 'rain/'):
        os.makedirs(images_main_path + 'rain/')
    images_save = images_main_path + 'rain/'
    image_aug.save(images_save + basename_image + "_rain" + img_extension)
    
    label = labels + basename_image
    if not os.path.exists(labels + 'rain/'):
        os.makedirs(labels + 'rain/')
    labels_save = labels + 'rain/'
    label_aug = labels_save + basename_image
    shutil.copyfile(label + ".txt", label_aug + "_rain.txt")
    
def cloud_image(img):
    image = imageio.imread(img)
    clouds = iaa.Clouds()
    image_aug = clouds(image=image)
    image_aug = Image.fromarray(image_aug)
    
    image_name = os.path.splitext(img)[0]
    basename_image = os.path.basename(image_name)
    img_extension = os.path.splitext(img)[1]
    if not os.path.exists(images_main_path + 'clouds/'):
        os.makedirs(images_main_path + 'clouds/')
    images_save = images_main_path + 'clouds/'
    image_aug.save(images_save + basename_image + "_clouds" + img_extension)
    
    label = labels + basename_image
    if not os.path.exists(labels + 'clouds/'):
        os.makedirs(labels + 'clouds/')
    labels_save = labels + 'clouds/'
    label_aug = labels_save + basename_image
    shutil.copyfile(label + ".txt", label_aug + "_clouds.txt")
    
def noise_image(img):
    image = imageio.imread(img)
    noise = iaa.imgcorruptlike.GaussianNoise(severity=random.randint(1,3))
    image_aug = noise(image=image)
    image_aug = Image.fromarray(image_aug)
    
    image_name = os.path.splitext(img)[0]
    basename_image = os.path.basename(image_name)
    img_extension = os.path.splitext(img)[1]
    if not os.path.exists(images_main_path + 'noise/'):
        os.makedirs(images_main_path + 'noise/')
    images_save = images_main_path + 'noise/'
    image_aug.save(images_save + basename_image + "_noise" + img_extension)
    
    label = labels + basename_image
    if not os.path.exists(labels + 'noise/'):
        os.makedirs(labels + 'noise/')
    labels_save = labels + 'noise/'
    label_aug = labels_save + basename_image
    shutil.copyfile(label + ".txt", label_aug + "_noise.txt")
    
def flip_image(img):
    image = imageio.imread(img)
    flip = iaa.Fliplr(1)
    image_aug = flip(image=image)
    image_aug = Image.fromarray(image_aug)
    
    image_name = os.path.splitext(img)[0]
    basename_image = os.path.basename(image_name)
    img_extension = os.path.splitext(img)[1]
    if not os.path.exists(images_main_path + 'flip/'):
        os.makedirs(images_main_path + 'flip/')
    images_save = images_main_path + 'flip/'
    image_aug.save(images_save + basename_image + "_flip" + img_extension)
    
    label = labels + basename_image
    if not os.path.exists(labels + 'flip/'):
        os.makedirs(labels + 'flip/')
    labels_save = labels + 'flip/'
    label_aug = labels_save + basename_image
    shutil.copyfile(label + ".txt", label_aug + "_flip.txt")
    f = open(label_aug + "_flip.txt", "r")
    new_label = ""
    l1 = f.readlines()
    for line in l1:
        cat, xcenter, ycenter, width, height = line.split()
        xcenter_mirror = 1 - float(xcenter)
        new_l = " ".join([cat, str(xcenter_mirror), ycenter, width, height])
        new_label += new_l + "\n"
    f.close()
    f1 = open(label_aug + "_flip.txt", "w")
    f1.write(new_label)
    f1.close()
    
def multiply_image(img):
    image = imageio.imread(img)
    multiply = iaa.Multiply((0.8, 1.2), per_channel=0.2)
    image_aug = multiply(image=image)
    image_aug = Image.fromarray(image_aug)
    
    image_name = os.path.splitext(img)[0]
    basename_image = os.path.basename(image_name)
    img_extension = os.path.splitext(img)[1]
    if not os.path.exists(images_main_path + 'multiply/'):
        os.makedirs(images_main_path + 'multiply/')
    images_save = images_main_path + 'multiply/'
    image_aug.save(images_save + basename_image + "_multiply" + img_extension)
    
    label = labels + basename_image
    if not os.path.exists(labels + 'multiply/'):
        os.makedirs(labels + 'multiply/')
    labels_save = labels + 'multiply/'
    label_aug = labels_save + basename_image
    shutil.copyfile(label + ".txt", label_aug + "_multiply.txt")
    
def pixelate_image(img):
    image = imageio.imread(img)
    pixelate = iaa.imgcorruptlike.Pixelate(severity=random.randint(1,3))
    image_aug = pixelate(image=image)
    image_aug = Image.fromarray(image_aug)
    
    image_name = os.path.splitext(img)[0]
    basename_image = os.path.basename(image_name)
    img_extension = os.path.splitext(img)[1]
    if not os.path.exists(images_main_path + 'pixelate/'):
        os.makedirs(images_main_path + 'pixelate/')
    images_save = images_main_path + 'pixelate/'
    image_aug.save(images_save + basename_image + "_pixelate" + img_extension)
    
    label = labels + basename_image
    if not os.path.exists(labels + 'pixelate/'):
        os.makedirs(labels + 'pixelate/')
    labels_save = labels + 'pixelate/'
    label_aug = labels_save + basename_image
    shutil.copyfile(label + ".txt", label_aug + "_pixelate.txt")

for image in images:
    print('\nCurrent image: ', image)
    fog_image(image)
    rain_image(image)
    cloud_image(image)
    noise_image(image)
    flip_image(image)
    multiply_image(image)
    pixelate_image(image)
