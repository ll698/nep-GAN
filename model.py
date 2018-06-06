
from __future__ import print_function

import tkinter
import tensorflow as tf


from keras.models import *
from keras.layers import *
from keras.initializers import *
from keras.optimizers import *
from keras.utils import np_utils
import keras
import keras.backend as K

import numpy as np

zed = 1024

def concat_diff(i): # batch discrimination -  increase generation diversity.
    # return i
    bv = Lambda(lambda x:K.mean(K.abs(x[:] - K.mean(x,axis=0)),axis=-1,keepdims=True))(i)
    i = merge([i,bv],mode='concat')
    return i


def residual_cell(input):
    x = Conv2D(input.shape, kernel_size=(3, 3), strides=(1,1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(input.shape, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(0.2)(x)

    x = add([input, x])
    return x

def gen(input_shape, f, batch_size): # generative network, 2
    s = input_shape[1]
    start_dim = int(s / 16)
    nb_upconv = 4

    output_channels = input_shape[-1]

    gen_input = Input(shape=(zed,), name="generator_input")
    x = Dense(f * start_dim * start_dim, input_dim=zed)(gen_input)
    x = Reshape((start_dim, start_dim, f))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    # Transposed conv blocks
    for i in range(nb_upconv):
        # if i < 2:
        nb_filters = int(f / (2 ** (i + 1)))
        s = start_dim * (2 ** (i + 1))
        o_shape = (batch_size, s, s, nb_filters)
        x = Deconv2D(nb_filters, (3, 3), output_shape=o_shape, strides=(2, 2), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        # else:
        #     x = UpSampling2D(size=(2, 2))(x)
        #     nb_filters = int(f / (2 ** (i + 1)))
        #     x = Conv2D(nb_filters, (3, 3), padding="same")(x)
        #     x = BatchNormalization(axis=1)(x)
        #     x = Activation("relu")(x)
        #     x = Conv2D(nb_filters, (3, 3), padding="same")(x)
        #     x = Activation("relu")(x)

    # Last block
    s = start_dim * (2 ** (nb_upconv))
    o_shape = (batch_size, s, s, output_channels)
    x = Deconv2D(output_channels, (3, 3), output_shape=o_shape, strides=(2, 2), padding="same")(x)
    #x = Conv2D(output_channels, (3, 3), name="gen_Conv2D_final", padding="same", activation='tanh')(x)

    #Residual cell
    shortcut = x

    x = Conv2D(output_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(0.2)(x)

    x = add([shortcut, x])
    x = Activation("tanh")(x)

    generator_model = Model(inputs=[gen_input], outputs=[x], name='G')
    return generator_model



def upsample_gen(input_shape):

    gen_input = Input(shape=input_shape, name="generator_input")
    x = UpSampling2D(size=(2, 2))(gen_input)
    x = Conv2D(3, (3, 3), padding="same")(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU(0.2)(x)
    x = residual_cell(x)
    x = residual_cell(x)
    x = Conv2D(3, (3, 3), name="gen_Conv2D_final", padding="same", activation='tanh')(x)
    upsample_model = Model(inputs=[gen_input], outputs=[x], name='U')
    return upsample_model

    

def dis(input_shape): # discriminative network, 2
    img_dim = input_shape
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

    #x = Activation('linear')(x)
    x = Activation('sigmoid')(x)

    discriminator_model = Model(inputs=[disc_input], outputs=[x], name="D")
    return discriminator_model


def gan(g, d, batch_size, wasserstein=False):
    # initialize a GAN trainer

    noise = Input(shape=g.input_shape[1:])
    real_data = Input(shape=d.input_shape[1:])

    generated = g(noise)
    gscore = d(generated)
    rscore = d(real_data)

    def log_eps(i):
        return K.log(i+1e-11)

    if wasserstein:
        dloss = K.mean(rscore * -np.ones(batch_size)) + K.mean(gscore * np.ones(batch_size))
        gloss = K.mean(gscore * -np.ones(batch_size))
    else:
        dloss = - K.mean(log_eps(1-gscore) + .1 * log_eps(1-rscore) + .9 * log_eps(rscore))
        gloss = - K.mean(log_eps(gscore))

    Adam = tf.train.AdamOptimizer

    lr,b1 = 2e-4,.2 # otherwise won't converge.
    opt_discriminator = Adam(lr,beta1=b1)
    opt_dcgan = Adam(lr,beta1=b1)

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

    def gan_feed(sess, batch_image,z_input, iteration):
        # actual GAN trainer
        nonlocal train_step,losses,noise,real_data,learning_phase, update_wd, update_wg, other_parameter_updates
        
        if (iteration % 3 == 0 and iteration > 700):
             train_step = [update_wd, update_wg, other_parameter_updates]
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
