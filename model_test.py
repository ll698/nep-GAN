
from __future__ import print_function

import tensorflow as tf

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.initializers import *
# from keras.optimizers import *
from keras.utils import np_utils
import keras
import keras.backend as K
from PIL import Image

import math
import random
import glob

import numpy as np

import cv2

batch_size = 32
nb_classes = 10
nb_epoch = 200
eps=1e-11

zed = 100

image_list = []
for filename in glob.glob('data64/train/*.jpg'): #assuming gif
    im=Image.open(filename)
    image_list.append(np.array(im, dtype='float32'))

X_train = np.asarray(image_list, dtype='float32')
X_train /= 255.0
X_train -= 0.5


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

    X_train-=0.5
    X_test-=0.5

    return X_train,Y_train,X_test,Y_test
print('loading cifar...')
X_train,yt,xv,yv = cifar()


def relu(i):
    return LeakyReLU(.2)(i)

def bn(i):
    return BatchNormalization()(i)

def gen2(): # generative network, 2
    # inp = Input(shape=(zed,))
    # i = inp
    # i = Reshape((1,1,zed))(i)

    # ngf=24

    # def deconv(i,nop,kw,oh,ow,std=1,tail=True,bm='same'):
    #     global batch_size
    #     i = Deconvolution2D(nop,kw,kw,subsample=(std,std),border_mode=bm,output_shape=(batch_size,oh,ow,nop))(i)
    #     if tail:
    #         i = bn(i)
    #         i = relu(i)
    #     return i

    # i = deconv(i,nop=ngf*8,kw=4,oh=8,ow=8,std=1,bm='valid')
    # i = deconv(i,nop=ngf*4,kw=4,oh=16,ow=16,std=2)
    # i = deconv(i,nop=ngf*2,kw=4,oh=32,ow=32,std=2)
    # i = deconv(i,nop=ngf*1,kw=4,oh=32,ow=32,std=2)

    # i = deconv(i,nop=3,kw=4,oh=32,ow=32,std=1,tail=False) # out : 32x32
    # i = Activation('tanh')(i)

    # m = Model(input=inp,output=i)

    s = 32
    f = 512

    start_dim = int(s / 16)
    nb_upconv = 4

    reshape_shape = (start_dim, start_dim, f)
    bn_axis = -1
    output_channels = img_dim[-1]

    gen_input = Input(shape=noise_dim, name="generator_input")

    # Noise input and reshaping
    x = Dense(f * start_dim * start_dim, input_dim=noise_dim, use_bias=False)(gen_input)
    x = Reshape(reshape_shape)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation("relu")(x)

    # Transposed conv blocks: Deconv2D->BN->ReLU
    for i in range(nb_upconv - 1):
        nb_filters = int(f / (2 ** (i + 1)))
        s = start_dim * (2 ** (i + 1))
        o_shape = (batch_size, s, s, nb_filters)
        x = Deconv2D(nb_filters, (3, 3),
                     output_shape=o_shape, strides=(2, 2),
                     padding="same", use_bias=False,
                     kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

    # Last block
    s = start_dim * (2 ** (nb_upconv))
    o_shape = (batch_size, s, s, output_channels)
    x = Deconv2D(output_channels, (3, 3),
                 output_shape=o_shape, strides=(2, 2),
                 padding="same", use_bias=False,
                 kernel_initializer=RandomNormal(stddev=0.02))(x)
    x = Activation("tanh")(x)

    generator_model = Model(inputs=[gen_input], outputs=[x], name=model_name)
    visualize_model(generator_model)

    return generator_model

    ICT_LEN = 10
    EMBEDDING_LEN = 100

    # weights are initlaized from normal distribution with below params
    weight_init = RandomNormal(mean=0.0, stddev=0.1)

    # latent var
    input_z = Input(shape=(EMBEDDING_LEN, ), name='input_z')

     # Noise input and reshaping
    x = Dense(8 * 8 * 512)(input_z)
    x = Reshape((8,8,512))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    for i in range(4):
        x = UpSampling2D(size=(2, 2))(x)
        nb_filters = int(512 / (2 ** (i + 1)))
        x = Conv2D(nb_filters, (3, 3), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(nb_filters, (3, 3), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = Activation("relu")(x)

    #x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(
        3, (3, 3),
        padding='same',
        activation='tanh',
        name='output_generated_image',
        kernel_initializer=weight_init)(x)

    return Model(inputs=[input_z], outputs=x, name='G')
   

def concat_diff(i): # batch discrimination -  increase generation diversity.
    # return i
    bv = Lambda(lambda x:K.mean(K.abs(x[:] - K.mean(x,axis=0)),axis=-1,keepdims=True))(i)
    i = merge([i,bv],mode='concat')
    return i

def dis2(): # discriminative network, 2
    # inp = Input(shape=(None,None,3))
    inp = Input(shape=(32,32,3))
    i = inp

    ndf=24

    def conv(i,nop,kw,std=1,usebn=True,bm='same'):
        i = Convolution2D(nop,kw,kw,border_mode=bm,subsample=(std,std))(i)
        if usebn:
            i = bn(i)
        i = relu(i)
        return i

    i = conv(i,ndf*1,4,std=2,usebn=False)
    i = concat_diff(i)
    i = conv(i,ndf*2,4,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*4,4,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*8,4,std=2)
    i = concat_diff(i)

    # 1x1
    i = Convolution2D(1,2,2,border_mode='valid')(i)

    i = Activation('linear',name='conv_exit')(i)
    i = GlobalAveragePooling2D()(i)
    i = Activation('sigmoid')(i)

    i = Reshape((1,))(i)

    m = Model(input=inp,output=i)
    return m

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


    dloss = K.mean(rscore * -np.ones(batch_size)) + K.mean(gscore * np.ones(batch_size))
    gloss = K.mean(gscore * -np.ones(batch_size))
    # single side label smoothing: replace 1.0 with 0.9
    dloss = - K.mean(log_eps(1-gscore) + .1 * log_eps(1-rscore) + .9 * log_eps(rscore))
    gloss = - K.mean(log_eps(gscore))

    Adam = tf.train.AdamOptimizer

    lr,b1 = 1e-4,.2 # otherwise won't converge.
    optimizer = Adam(lr,beta1=b1)

    grad_loss_wd = optimizer.compute_gradients(dloss, d.trainable_weights)
    update_wd = optimizer.apply_gradients(grad_loss_wd)
  
    grad_loss_wg = optimizer.compute_gradients(gloss, g.trainable_weights)
    update_wg = optimizer.apply_gradients(grad_loss_wg)
 
    def get_internal_updates(model):
        # get all internal update ops (like moving averages) of a model
        inbound_nodes = model.inbound_nodes
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
        
        if iteration % 5 == 0 and iteration > 100:
            print('training gen')
            print(iteration)
            train_step = [update_wg, other_parameter_updates]
        else:
            print('training disc')
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
def r(ep=10000,noise_level=.001):
    sess = K.get_session()

    np.random.shuffle(X_train)
    shuffled_cifar = X_train
    length = len(shuffled_cifar)

    for i in range(ep):
        noise_level *= 0.99
        print('---------------------------')
        print('iter',i,'noise',noise_level)

        # sample from cifar
        j = i % int(length/batch_size)
        minibatch = shuffled_cifar[j*batch_size:(j+1)*batch_size]
        # minibatch += np.random.normal(loc=0.,scale=noise_level,size=subset_cifar.shape)

        z_input = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))

        # train for one step
        losses = gan_feed(sess,minibatch,z_input, i)
        print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

        if i==ep-1 or i % 10==0: show()

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

samples_z = np.random.normal(0., 1., (100, zed))
def show(save=False):
    #i = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))
    gened = gm.predict([samples_z])

    gened *= 0.5
    gened +=0.5

    im,ims = flatten_multiple_image_into_image(gened)
    cv2.imshow('gened scale:'+str(ims),im)
    cv2.waitKey(1)

    if save!=False:
        cv2.imwrite(save,im*255)


    # generated_classes = np.array(list(range(0, 10)) * 10)
    # generated_images = gm.predict([samples_z])
    # index = np.random.choice(len(X_train), 100, replace=False)
    # real_images = X_train[index]
    # print(generated_images)

    # rr = []
    # for c in range(10):
    #     rr.append(
    #         np.concatenate(generated_images[c * 10:(1 + c) * 10]).reshape(
    #             1280, 128, 3))
    # img = np.hstack(rr)
    # rr2 = []
    # for c in range(10):
    #     rr.append(
    #         np.concatenate(real_images[c * 10:(1 + c) * 10]).reshape(
    #             1280, 128, 3))
    # img = np.hstack(rr)

    # if save:
    #     plt.imsave(OUT_DIR + '/samples_real_%07d.png' % n, img)

    # return img

run = r()