


"""

import os
folder = os.listdir()
print(folder)

"""



import os
import shutil

img = []
labels = []

# Getting the current work directory (cwd)
thisdir = os.getcwd()

# r=root, d=directories, f = files
for r, d, f in os.walk(thisdir):
    for file in f:
        if r != "dataset_final":
            instance = os.path.join(r, file)
        
            if file.endswith(".jpg"):
                img.append('\\'.join(instance.split("\\")[-2:]).split(".")[0])
          

            if file.endswith(".txt"):
                labels.append('\\'.join(instance.split("\\")[-2:]).split(".")[0])


if not os.path.isdir("./dataset_final"):
    os.mkdir("./dataset_final", 0o700)

labels_img = []
print(img[0],"-------------------")

for label in labels:
    if label in img:
        labels_img.append(label)



for instance in labels_img:
    if os.path.isfile("..\\"+ instance + ".txt"):
        shutil.copyfile( "..\\"+ instance + ".txt", "..\\dataset_final\\" + instance + ".txt")
    if os.path.isfile("..\\"+ instance + ".jpg"):
        shutil.copyfile("..\\"+ instance + ".jpg", "..\\dataset_final\\" + instance + ".jpg")













