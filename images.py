import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from PIL import Image


zed = 256
OUT_DIR = 'out/imgs' 
OUT_DIR_UPS = 'out/ups/imgs'
samples_z_const = np.random.normal(0., 0., (100, zed))

def load_images(filepath):
    image_list = []
    for filename in glob.glob(filepath): #assuming gif
        im=Image.open(filename)
        image_list.append(np.array(im, dtype='float32'))

    X_train = np.asarray(image_list, dtype='float32')
    X_train /= 255
    X_train *= 2
    X_train -= 1
    np.random.shuffle(X_train)
    return X_train
    

def get_batch(X_train, datagen, batch_size, length):
    minibatch = datagen.flow(X_train,batch_size = batch_size)
    print(minibatch.size)
    return minibatch
    #minibatch = minibatch[0]

def show(count, gm, num_samples, input_shape, ups=None, upsample=True, save=True):
    samples_z = np.random.normal(0., 1., (num_samples, zed))
    generated_images = gm.predict([samples_z])
    generated_images_ups = ups([gm.predict([samples_z])])
    

    #rescale_images
    generated_images += 1
    generated_images /= 2
    generated_images_ups += 1
    generated_images_ups /= 2

    rr = []
    for c in range(10):
        rr.append(
            np.concatenate(generated_images[c * 10:(1 + c) * 10]).reshape(
                input_shape[0] * 10, input_shape[1], 3))
    img = np.hstack(rr)

    rr1 = []
    for c in range(10):
        rr1.append(
            np.concatenate(generated_images_ups[c * 10:(1 + c) * 10]).reshape(
                input_shape[0]* 2 * 10, input_shape[1]*2, 3))
    img_ups = np.hstack(rr1)

    if save:
        plt.imshow(img)
        plt.imshow(img_ups)
        plt.imsave(OUT_DIR + '/samples_real_%07d.png' % count, img)
        plt.imsave(OUT_DIR_UPS + '/samples_real_%07d.png' % count, img_ups)
        count += 1

    return img