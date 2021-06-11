
"""

import os
folder = os.listdir()
print(folder)

"""


import os
import shutil

files = []

# Getting the current work directory (cwd)
thisdir = os.getcwd()

# r=root, d=directories, f = files
for r, d, f in os.walk(thisdir):
    for file in f:
        instance = os.path.join(r, file)

        if file.endswith(".jpg"):
            files.append(instance)


        if file.endswith(".txt"):
            files.append(instance)

for file in files:
    os.rename(file, file.replace(" ", ""))

print(files)




