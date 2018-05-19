from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
image_list = []
count = 0
OUT_DIR = "test/"
for filename in glob.glob('data128/train/*.jpg'): #assuming gif
    if count < 5:
        im=Image.open(filename)
        ar = np.array(im)
        print(np.amin(ar))
        plt.imsave(OUT_DIR + 'samples_test%i.png' % count, im, cmap=plt.cm.gray)
        image_list.append(np.array(im))
    count+=1

X_train = np.asarray(image_list)

#plt.imsave(OUT_DIR + '/samples_%07d.png' % n, img, cmap=plt.cm.gray)