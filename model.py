import os

import matplotlib.pyplot as plt


import keras.backend as K
from keras.datasets import mnist, cifar10
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.initializers import *
from keras.callbacks import *
from keras.utils.generic_utils import Progbar
from PIL import Image
import glob
from random import randint


RND = 1337

RUN = 'F'
OUT_DIR = 'out/' + RUN
TENSORBOARD_DIR = RUN

# GPU # 
GPU = "1"

# latent vector size
Z_SIZE = 100

# number of iterations D is trained for per each G iteration
D_ITERS = 20
G_ITERS = 5

BATCH_SIZE = 64
ITERATIONS = 1000

np.random.seed(RND)

if not os.path.isdir(OUT_DIR): os.makedirs(OUT_DIR)

K.set_image_dim_ordering('tf')

print("loading data")
#load in data
image_list = []
for filename in glob.glob('data128/train/*.jpg'): #assuming gif
    im=Image.open(filename)
    image_list.append(np.array(im, dtype='float32'))

X_train = np.asarray(image_list, dtype='float32')
X_train /= 255.0



# basically return mean(y_pred),
# but with ability to inverse it for minimization (when y_true == -1)
def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)

def concat_diff(i): # batch discrimination -  increase generation diversity.
    # return i
    bv = Lambda(lambda x:K.mean(K.abs(x[:] - K.mean(x,axis=0)),axis=-1,keepdims=True))(i)
    i = merge([i,bv],mode='concat')
    return i


def create_D():

    # # weights are initlaized from normal distribution with below params
    # weight_init = RandomNormal(mean=0., stddev=0.02)

    # input_image = Input(shape=(128, 128, 3), name='input_image')

    # x = Conv2D(
    #     64, (3, 3), strides = (2,2),
    #     name='conv_1',
    #     kernel_initializer=weight_init)(input_image)
    # x = LeakyReLU()(x)
    # #x = BatchNormalization(axis=1)(x)
    # x = MaxPool2D(pool_size=2)(x)
    # #x = concat_diff(x)

    # x = Conv2D(
    #     128, (3, 3),
    #     name='conv_2',
    #     kernel_initializer=weight_init)(x)
    # x = MaxPool2D(pool_size=1)(x)
    # #x = BatchNormalization(axis=1)(x)
    # x = LeakyReLU()(x)
    # #x = concat_diff(x)

    # x = Conv2D(
    #     256, (3, 3), strides = (2,2),
    #     name='conv_3',
    #     kernel_initializer=weight_init)(x)
    # x = LeakyReLU()(x)
    # #x = BatchNormalization(axis=1)(x)
    # x = MaxPool2D(pool_size=2)(x)
    # #x = concat_diff(x)

    # x = Conv2D(
    #     512, (3, 3),
    #     name='conv_4',
    #     kernel_initializer=weight_init)(x)
    # x = LeakyReLU()(x)
    # #x = BatchNormalization(axis=1)(x)
    # x = MaxPool2D(pool_size=1)(x)
    # #x = concat_diff(x)

    # x = Conv2D(1, (3, 3), name="last_conv", padding="same", use_bias=False,
    #         kernel_initializer=RandomNormal(stddev=0.02))(x)
    # # Average pooling
    # x = GlobalAveragePooling2D()(x)
    
    # return Model(
    #     inputs=[input_image], outputs=[x], name='D')
    img_dim = (128, 128, 3)
    bn_axis = -1
    min_s = min(img_dim[:-1])
    disc_input = Input(shape=img_dim, name="discriminator_input")

    # Get the list of number of conv filters
    # (first layer starts with 64), filters are subsequently doubled
    nb_conv = int(np.floor(np.log(min_s // 4) / np.log(2)))
    list_f = [64 * min(8, (2 ** i)) for i in range(nb_conv)]

    # First conv with 2x2 strides
    x = Conv2D(list_f[0], (3, 3), strides=(2, 2), name="disc_conv2d_1",
               padding="same", use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.02))(disc_input)
    #x = BatchNormalization(axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)

    # Conv blocks: Conv2D(2x2 strides)->BN->LReLU
    for i, f in enumerate(list_f[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name, padding="same", use_bias=False,
                   kernel_initializer=RandomNormal(stddev=0.02))(x)
        #x = BatchNormalization(axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)

    # Last convolution
    x = Conv2D(1, (3, 3), name="last_conv", padding="same", use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.02))(x)
    # Average pooling
    x = GlobalAveragePooling2D()(x)

    discriminator_model = Model(inputs=[disc_input], outputs=[x], name='D')

    return discriminator_model


def create_G(Z_SIZE=Z_SIZE):
    DICT_LEN = 10
    EMBEDDING_LEN = Z_SIZE

    # weights are initlaized from normal distribution with below params
    weight_init = RandomNormal(mean=0.0, stddev=0.1)

    # latent var
    input_z = Input(shape=(Z_SIZE, ), name='input_z')

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
    
D  = create_D()

D.compile(
    optimizer=RMSprop(),
    loss=[wasserstein])

input_z = Input(shape=(Z_SIZE, ), name='input_z_')
input_class = Input(shape=(1, ),name='input_class_', dtype='int32')



G = create_G()

# create combined D(G) model
output_is_fake = D(G(inputs=[input_z]))
DG = Model(inputs=[input_z], outputs=[output_is_fake])
DG.get_layer('D').trainable = False # freeze D in generator training faze

DG.compile(
    optimizer=RMSprop(),
    loss=[wasserstein]
)

# save 10x10 sample of generated images
samples_z = np.random.normal(0., 1., (100, Z_SIZE))
def generate_samples(n=0, save=True):

    generated_classes = np.array(list(range(0, 10)) * 10)
    generated_images = G.predict([samples_z])
    index = np.random.choice(len(X_train), 100, replace=False)
    real_images = X_train[index]
    print(generated_images)

    rr = []
    for c in range(10):
        rr.append(
            np.concatenate(generated_images[c * 10:(1 + c) * 10]).reshape(
                1280, 128, 3))
    img = np.hstack(rr)
    rr2 = []
    for c in range(10):
        rr.append(
            np.concatenate(real_images[c * 10:(1 + c) * 10]).reshape(
                1280, 128, 3))
    img = np.hstack(rr)

    if save:
        plt.imsave(OUT_DIR + '/samples_real_%07d.png' % n, img)

    return img
       

# fake = 1
# real = -1

progress_bar = Progbar(target=ITERATIONS)

DG_losses = []
D_true_losses = []
D_fake_losses = []

generate_samples(0)

print("training...")
for it in range(ITERATIONS):

    print("Iteration:")
    print(it)

    if it < 1:
        d_iters = 50
    else:
        d_iters = D_ITERS

    for d_it in range(d_iters):

        # unfreeze D
        D.trainable = True
        for l in D.layers: l.trainable = True
            
        # # restore D dropout rates
        # for l in D.layers:
        #     if l.name.startswith('dropout'):
        #         l.rate = l._rate

        # clip D weights

        for l in D.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -0.001, 0.001) for w in weights]
            l.set_weights(weights)

        # 1.1: maximize D output on reals === minimize -1*(D(real))

        # draw random samples from real images
        index = np.random.choice(len(X_train), BATCH_SIZE, replace=False)
        real_images = X_train[index]
        print(real_images.shape)
        print("training discriminator on real images")
        D_loss = D.fit(real_images, [-np.ones(BATCH_SIZE)])

        # 1.2: minimize D output on fakes 
        zz = np.random.normal(0., 1., (BATCH_SIZE, Z_SIZE))
        generated_images = G.predict([zz])
        print("training discriminator on fake images")
        D_loss = D.fit(generated_images, [np.ones(BATCH_SIZE)])

    # 2: train D(G) (D is frozen)
    # minimize D output while supplying it with fakes, telling it that they are reals (-1)

    # freeze D
    D.trainable = False
    for l in D.layers: l.trainable = False
        
    # # disable D dropout layers
    # for l in D.layers:
    #     if l.name.startswith('dropout'):
    #         l.rate = 0.


    print("training generator")

    for i in range(G_ITERS):
        zz = np.random.normal(0., 1., (BATCH_SIZE, Z_SIZE)) 
        DG_loss = DG.fit(
            [zz],
            [-np.ones(BATCH_SIZE)])


    generate_samples(it, save=True)