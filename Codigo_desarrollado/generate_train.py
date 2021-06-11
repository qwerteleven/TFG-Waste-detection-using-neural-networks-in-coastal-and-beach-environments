
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
        instance = os.path.join(r, file)

        if file.endswith(".jpg"):
            img.append('\\'.join(instance.split("\\")[-2:]).split(".")[0])


        if file.endswith(".txt"):
            labels.append('\\'.join(instance.split("\\")[-2:]).split(".")[0])


# if not os.path.isdir("./dataset_final"):
#    os.mkdir("./dataset_final", 0o700)

labels_img = []
print(labels[2], "-------------------")

for label in labels:
    if label in img:
        labels_img.append(label)

f = open("train.txt", "w")

for instance in labels_img:
    f.write("data\\obj\\" + instance + ".jpg")
    f.write("\n")

f.close()






