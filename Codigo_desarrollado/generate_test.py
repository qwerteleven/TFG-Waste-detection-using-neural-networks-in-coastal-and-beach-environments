


import os
folder = os.listdir("D:\\LacosadeTrabajo\\WorkSpaces\\TFG\\datasets\\yolo_test\\darknet-master\\build\\darknet\\x64\\data\\obj")
print(folder)



img = []
labels = []

# Getting the current work directory (cwd)
thisdir = "D:\\LacosadeTrabajo\\WorkSpaces\\TFG\\datasets\\yolo_test\\darknet-master\\build\\darknet\\x64\\data\\obj"

# r=root, d=directories, f = files

for r, d, f in os.walk(thisdir):
    for file in f:

        if file.endswith(".JPG"):
            img.append(r + "\\" + file)

        if file.endswith(".jpg"):
            img.append(r + "\\" + file)


# if not os.path.isdir("./dataset_final"):
#    os.mkdir("./dataset_final", 0o700)

f = open("train.txt", "w")

for instance in img:
    f.write(instance)
    f.write("\n")

f.close()






