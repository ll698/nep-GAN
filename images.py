import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from PIL import Image


zed = 100
OUT_DIR = 'out/imgs' 

def load_images(filepath):
    image_list = []
    for filename in glob.glob(filepath): #assuming gif
        im=Image.open(filename)
        image_list.append(np.array(im, dtype='float32'))

    X_train = np.asarray(image_list, dtype='float32')
    X_train /= 255

    np.random.shuffle(X_train)
    return X_train
    

def get_batch(X_train, datagen, batch_size, length):
    minibatch = datagen.flow(X_train,batch_size = batch_size)
    print(minibatch.size)
    return minibatch
    #minibatch = minibatch[0]


def autoscaler(img):
    limit = 400.
    # scales = [0.1,0.125,1./6.,0.2,0.25,1./3.,1./2.] + range(100)
    scales = np.hstack([1./np.linspace(10,2,num=9), np.linspace(1,100,num=100)])

    imgscale = limit/float(img.shape[0])
    for s in scales:
        if s>=imgscale:
            imgscale=s
            break

    img = cv2.resize(img,dsize=(int(img.shape[1]*imgscale),int(img.shape[0]*imgscale)),interpolation=cv2.INTER_NEAREST)

    return img,imgscale

def flatten_multiple_image_into_image(arr):
    num,uh,uw,_ = arr.shape

    patches = int(num+1)
    height = int(math.sqrt(patches)*0.9)
    width = int(patches/height+1)

    img = np.zeros((height*uh+height, width*uw+width, 3),dtype='float32')

    index = 0
    for row in range(height):
        for col in range(width):
            if index>=num-1:
                break
            channels = arr[index]
            img[row*uh+row:row*uh+uh+row,col*uw+col:col*uw+uw+col,:] = channels
            index+=1

    img,imgscale = autoscaler(img)

    return img,imgscale


samples_z1 = np.random.normal(0., 1., (100, zed))

def show(count, gm, input_shape, num_samples, save=True):
    samples_z = np.random.normal(0., 1., (num_samples, zed))
    generated_images = gm.predict([samples_z])
    generated_images_same = gm.predict([samples_z1])
    print(generated_images.shape)

    rr = []
    for c in range(10):
        rr.append(
            np.concatenate(generated_images[c * 10:(1 + c) * 10]).reshape(
                input_shape[0] * 10, input_shape[1], 3))
    img = np.hstack(rr)

    rr1 = []
    for c in range(10):
        rr1.append(
            np.concatenate(generated_images_same[c * 10:(1 + c) * 10]).reshape(
                input_shape[0] * 10, input_shape[1], 3))
    img_same = np.hstack(rr1)
    if save:
        plt.imshow(img)
        plt.imsave(OUT_DIR + '/samples_real_%07d.png' % count, img)
        plt.imsave('out/same' + '/samples_real_%07d.png' % count, img_same)
        count += 1

    return img