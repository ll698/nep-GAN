
from __future__ import print_function

import tkinter
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.initializers import *
from keras.optimizers import *
from keras.utils import np_utils
import keras
import keras.backend as K
from PIL import Image

import math
import random
import glob

import numpy as np

import cv2

batch_size = 100
nb_classes = 10
nb_epoch = 200
eps=1e-12

RUN = 'F'
OUT_DIR = 'out/' + RUN

zed = 100
INPUT_SHAPE = (96,96,3)

image_list = []
for filename in glob.glob('data96/train/*.jpg'): #assuming gif
    im=Image.open(filename)
    image_list.append(np.array(im, dtype='float32'))

X_train = np.asarray(image_list, dtype='float32')
X_train /= 255
print(X_train)


def cifar():
    # input image dimensions
    img_rows, img_cols = 32, 32
    # the CIFAR10 images are RGB
    img_channels = 3

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255


    return X_train,Y_train,X_test,Y_test
# print('loading cifar...')
#X_train, _, _, _ = cifar()




def relu(i):
    return LeakyReLU(.2)(i)

def bn(i):
    return BatchNormalization()(i)

def gen2(): # generative network, 2
    s = INPUT_SHAPE[1]
    f = 1024


    start_dim = int(s / 16)
    nb_upconv = 4

    reshape_shape = (start_dim, start_dim, f)
    bn_axis = -1
    output_channels = INPUT_SHAPE[-1]

    gen_input = Input(shape=(zed,), name="generator_input")

    x = Dense(f * start_dim * start_dim, input_dim=zed)(gen_input)
    x = Reshape(reshape_shape)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation("relu")(x)

    # Transposed conv blocks
    print(nb_upconv)
    for i in range(nb_upconv):
        if i < 2:
            nb_filters = int(f / (2 ** (i + 1)))
            s = start_dim * (2 ** (i + 1))
            o_shape = (batch_size, s, s, nb_filters)
            x = Deconv2D(nb_filters, (3, 3), output_shape=o_shape, strides=(2, 2), padding="same")(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)
        else:
            x = UpSampling2D(size=(2, 2))(x)
            nb_filters = int(f / (2 ** (i + 1)))
            x = Conv2D(nb_filters, (3, 3), padding="same")(x)
            x = BatchNormalization(axis=1)(x)
            x = Activation("relu")(x)
            x = Conv2D(nb_filters, (3, 3), padding="same")(x)
            x = Activation("relu")(x)

    # Last block
    # s = start_dim * (2 ** (nb_upconv))
    # o_shape = (batch_size, s, s, output_channels)
    # x = Deconv2D(output_channels, (3, 3), output_shape=o_shape, strides=(2, 2), padding="same")(x)
    x = Conv2D(output_channels, (3, 3), name="gen_Conv2D_final", padding="same", activation='tanh')(x)
    #x = Activation("tanh")(x)

    generator_model = Model(inputs=[gen_input], outputs=[x], name='G')
    return generator_model
   

def concat_diff(i): # batch discrimination -  increase generation diversity.
    # return i
    bv = Lambda(lambda x:K.mean(K.abs(x[:] - K.mean(x,axis=0)),axis=-1,keepdims=True))(i)
    i = merge([i,bv],mode='concat')
    return i

def dis2(): # discriminative network, 2
    # inp = Input(shape=(None,None,3))
    img_dim = INPUT_SHAPE
    bn_axis = -1
    min_s = min(img_dim[:-1])

    disc_input = Input(shape=img_dim, name="discriminator_input")

    # Get the list of number of conv filters
    # (first layer starts with 64), filters are subsequently doubled
    nb_conv = int(np.floor(np.log(min_s // 4) / np.log(2)))
    list_f = [128 * min(8, (2 ** i)) for i in range(nb_conv)]

    # First conv with 2x2 strides
    x = Conv2D(list_f[0], (3, 3), strides=(2, 2), name="disc_conv2d_1",
               padding="same", use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.02))(disc_input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)
    x = concat_diff(x)

    # Conv blocks: Conv2D(2x2 strides)->BN->LReLU
    for i, f in enumerate(list_f[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name, padding="same", use_bias=False,
                   kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.2)(x)
        x = concat_diff(x)

    # Last convolution
    x = Conv2D(1, (2, 2), name="last_conv", padding="same", use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.02))(x)
    # Average pooling
    x = Activation('linear')(x)
    x = Activation('sigmoid')(x)

    discriminator_model = Model(inputs=[disc_input], outputs=[x], name="D")
    return discriminator_model


print('generating G...')
gm = gen2()
gm.summary()

print('generating D...')
dm = dis2()
dm.summary()


def gan(g,d):
    count = 0
    # initialize a GAN trainer

    # this is the fastest way to train a GAN in Keras
    # two models are updated simutaneously in one pass

    noise = Input(shape=g.input_shape[1:])
    real_data = Input(shape=d.input_shape[1:])

    generated = g(noise)
    gscore = d(generated)
    rscore = d(real_data)

    def log_eps(i):
        return K.log(i+1e-11)


    # dloss = K.mean(rscore * -np.ones(batch_size)) + K.mean(gscore * np.ones(batch_size))
    # gloss = K.mean(gscore * -np.ones(batch_size))
    # single side label smoothing: replace 1.0 with 0.9
    dloss = - K.mean(log_eps(1-gscore) + .1 * log_eps(1-rscore) + .9 * log_eps(rscore))
    gloss = - K.mean(log_eps(gscore))

    Adam = tf.train.AdamOptimizer
    SGD = tf.train.MomentumOptimizer

    lr,b1 = 2e-4,.2 # otherwise won't converge.
    opt_discriminator = Adam(lr,beta1=b1)

    lr,b1 = 2e-4,.2 # otherwise won't converge.
    opt_dcgan = Adam(lr,beta1=b1)

    # opt_dcgan = Adam(1E-4, 0.5, 0.999, 1e-08)
    # #opt_discriminator = SGD(1E-3, 0.9, use_nesterov=True)
    # opt_discriminator = Adam(2E-4, 0.5, 0.999, 1e-08)
    grad_loss_wd = opt_discriminator.compute_gradients(dloss, d.trainable_weights)
    update_wd = opt_discriminator.apply_gradients(grad_loss_wd)
  
    grad_loss_wg = opt_dcgan.compute_gradients(gloss, g.trainable_weights)
    update_wg = opt_dcgan.apply_gradients(grad_loss_wg)
 
    def get_internal_updates(model):
        # get all internal update ops (like moving averages) of a model
        inbound_nodes = model._inbound_nodes
        input_tensors = []
        for ibn in inbound_nodes:
            input_tensors+= ibn.input_tensors
        updates = [model.get_updates_for(i) for i in input_tensors]
        return updates

    other_parameter_updates = [get_internal_updates(m) for m in [d,g]]
    # those updates includes batch norm.

    print('other_parameter_updates for the models(mainly for batch norm):')
    print(other_parameter_updates)

    train_step = [update_wd, update_wg, other_parameter_updates]
    losses = [dloss,gloss]

    learning_phase = K.learning_phase()

    def gan_feed(sess,batch_image,z_input, iteration):
        # actual GAN trainer
        nonlocal train_step,losses,noise,real_data,learning_phase, count, update_wd, update_wg, other_parameter_updates
        
        if (iteration % 3 == 0 and iteration > 700):
             train_step = [update_wd, update_wg, other_parameter_updates]
        # elif iteration > 1000:
        #       train_step = [update_wd, other_parameter_updates]
        else:
            train_step = [update_wd, other_parameter_updates]
        res = sess.run([train_step,losses],feed_dict={
        noise:z_input,
        real_data:batch_image,
        learning_phase:True,
        # Keras layers needs to know whether
        # this run is training or testring (you know, batch norm and dropout)
        })

        loss_values = res[1]
        return loss_values #[dloss,gloss]

    return gan_feed

print('generating GAN...')
gan_feed = gan(gm,dm)

print('Ready. enter r() to train')
def r(ep=100000,noise_level=.01):
    sess = K.get_session()

    np.random.shuffle(X_train)
    shuffled_cifar = X_train

    datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    )

    datagen.fit(shuffled_cifar)
    print("test")

    length = len(shuffled_cifar)
    count = 0
    for i in range(ep):
        count +=1
        noise_level *= 0.99
        print('---------------------------')
        print('iter',i,'noise',noise_level)

        # sample from cifar
        j = i % int(length/batch_size)
        minibatch = shuffled_cifar[j*batch_size:(j+1)*batch_size]
        minibatch = datagen.flow(shuffled_cifar,batch_size =((j+1) * batch_size - j * batch_size))
        minibatch = minibatch[0]
        # minibatch += np.random.normal(loc=0.,scale=noise_level,size=subset_cifar.shape)

        z_input = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))

        # train for one step
        losses = gan_feed(sess,minibatch,z_input, i)
        print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

        if i==ep-1 or i % 100 == 0: show(count)

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
    import cv2
    num,uh,uw,depth = arr.shape

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
def show(count, save=True):
    samples_z = np.random.normal(0., 1., (100, zed))
    generated_images = gm.predict([samples_z])
    generated_images_same = gm.predict([samples_z1])
    print(generated_images.shape)

    rr = []
    for c in range(10):
        rr.append(
            np.concatenate(generated_images[c * 10:(1 + c) * 10]).reshape(
                INPUT_SHAPE[0] * 10, INPUT_SHAPE[1], 3))
    img = np.hstack(rr)

    rr1 = []
    for c in range(10):
        rr1.append(
            np.concatenate(generated_images_same[c * 10:(1 + c) * 10]).reshape(
                INPUT_SHAPE[0] * 10, INPUT_SHAPE[1], 3))
    img_same = np.hstack(rr1)
    if save:
        plt.imshow(img)
        plt.imsave(OUT_DIR + '/samples_real_%07d.png' % count, img)
        plt.imsave('out/same' + '/samples_real_%07d.png' % count, img_same)
        count += 1

    return img

run = r()