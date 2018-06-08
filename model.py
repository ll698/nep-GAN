
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

zed = 256

def concat_diff(i): # batch discrimination -  increase generation diversity.
    # return i
    bv = Lambda(lambda x:K.mean(K.abs(x[:] - K.mean(x,axis=0)),axis=-1,keepdims=True))(i)
    i = merge([i,bv],mode='concat')
    return i


def residual_cell(input_shape):
    input = Input(shape=input_shape, name="generator_input")
    x = Conv2D(input_shape[2]*2, kernel_size=(3, 3), strides=(1,1), padding='same')(input)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(input_shape[2], kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(0.2)(x)

    x = add([input, x])

    res_cell = Model(inputs=[input], outputs=[x])
    return res_cell

def gen(input_shape, f, batch_size, upsampling=True): # generative network, 2
    s = input_shape[1]
    start_dim = int(s / 16)
    nb_upconv = 4

    output_channels = input_shape[-1]

    gen_input = Input(shape=(zed,), name="generator_input")
    x = Dense(f * start_dim * start_dim, input_dim=zed)(gen_input)
    x = Reshape((start_dim, start_dim, f))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    if upsampling:
            # Upscaling blocks: Upsampling2D->Conv2D->ReLU->BN->Conv2D->ReLU
        for i in range(nb_upconv):
            x = UpSampling2D(size=(2, 2))(x)
            nb_filters = int(f / (2 ** (i + 1)))
            x = Conv2D(nb_filters, (3, 3), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(x)
            x = BatchNormalization(axis=1)(x)
            x = Activation("relu")(x)
            x = Conv2D(nb_filters, (3, 3), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(x)
            x = Activation("relu")(x)

        # Last block
        x = Conv2D(output_channels, (3, 3), name="gen_conv2d_final",
                padding="same", activation='tanh', kernel_initializer=RandomNormal(stddev=0.02))(x)


    else:
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

    generator_model = Model(inputs=[gen_input], outputs=[x], name='G')
    return generator_model



def upsample_gen(input_shape, output_shape, batch_size):

    gen_input = Input(shape=input_shape, name="generator_input")
    # x = UpSampling2D(size=(2, 2))(gen_input)
    # x = Conv2D(3, (3, 3), padding="same")(x)
    # x = BatchNormalization(axis=1)(x)
    # x = LeakyReLU(0.2)(x)
    res1 = residual_cell(input_shape)
    res2 = residual_cell((128, 128, 12))
    # x = Conv2D(3, (3, 3), name="gen_Conv2D_final", padding="same", activation='tanh')(x)
    x = res1(gen_input)
    x = UpSampling2D(size=(2, 2))(x)
            
    x = Conv2D(24, (3, 3), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation("relu")(x)
    x = Conv2D(12, (3, 3), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(x)
    x = Activation("relu")(x)

    x = res2(x)
    x = Conv2D(6, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(3, (3, 3), name="gen_conv2d_final",
                padding="same", activation='tanh', kernel_initializer=RandomNormal(stddev=0.02))(x)
    

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

    x = Activation('linear')(x)
    x = Activation('sigmoid')(x)

    discriminator_model = Model(inputs=[disc_input], outputs=[x], name="D")
    return discriminator_model


def gan(g, d, batch_size, ups=None, d_ups=None, upsample=False, wasserstein=False):
    # initialize a GAN trainer

    noise = Input(shape=g.input_shape[1:])
    real_data_64 = Input(shape=d.input_shape[1:])
    real_data_128 = Input(shape=d.input_shape[1:])

    generated = g(noise)
    gscore = d(generated)
    rscore = d(real_data_64)

    if upsample:
        ups_generated = ups(generated)
        ups_score = d_ups(ups_generated)
        d_ups_score = d_ups(real_data_128)


    def log_eps(i):
        return K.log(i+1e-11)

    if wasserstein:
        dloss = K.mean(rscore * -np.ones(batch_size)) + K.mean(gscore * np.ones(batch_size))
        gloss = K.mean(gscore * -np.ones(batch_size))
        
        if upsample:
            d_ups_loss = K.mean(d_ups_score * -np.ones(batch_size)) + K.mean(ups_score * np.ones(batch_size))
            ups_loss = K.mean(ups_score * -np.ones(batch_size))
    else:
        dloss = - K.mean(log_eps(1-gscore) + .1 * log_eps(1-rscore) + .9 * log_eps(rscore))
        gloss = - K.mean(log_eps(gscore))

        if upsample:
            ups_loss = - K.mean(log_eps(ups_score))
            d_ups_loss = - K.mean(log_eps(1-ups_score) + .1 * log_eps(1-d_ups_score) + .9 * log_eps(d_ups_score))


    Adam = tf.train.AdamOptimizer

    lr,b1 = 2e-4,.2 # otherwise won't converge.
    opt_dis = Adam(lr,beta1=b1)
    opt_gen = Adam(lr,beta1=b1)
    opt_ups = Adam(lr, beta1=b1)
    opt_ups_dis = Adam(lr, beta1=b1)

    grad_loss_wd = opt_dis.compute_gradients(dloss, d.trainable_weights)
    update_wd = opt_dis.apply_gradients(grad_loss_wd)
  
    grad_loss_wg = opt_gen.compute_gradients(gloss, g.trainable_weights)
    update_wg = opt_gen.apply_gradients(grad_loss_wg)

    if upsample:
        grad_loss_ups_d = opt_ups_dis.compute_gradients(d_ups_loss, d_ups.trainable_weights)
        update_w_ups_d = opt_ups_dis.apply_gradients(grad_loss_ups_d)

        grad_loss_ups = opt_ups.compute_gradients(ups_loss, ups.trainable_weights)
        update_w_ups = opt_ups.apply_gradients(grad_loss_ups)
 
    def get_internal_updates(model):
        # get all internal update ops (like moving averages) of a model
        inbound_nodes = model._inbound_nodes
        input_tensors = []
        for ibn in inbound_nodes:
            input_tensors+= ibn.input_tensors
        updates = [model.get_updates_for(i) for i in input_tensors]
        return updates

    other_parameter_updates = [get_internal_updates(m) for m in [d,g]]
    if upsample:
        other_parameter_updates = [get_internal_updates(m) for m in [d,g,ups, d_ups]]
    # those updates includes batch norm.

    print('other_parameter_updates for the models(mainly for batch norm):')
    print(other_parameter_updates)

    train_step = [update_wd, update_wg, other_parameter_updates]
    losses = [dloss,gloss]

    if upsample:
        train_step = [update_wd, update_wg, update_w_ups, other_parameter_updates]
        losses = [dloss, gloss, ups_loss, d_ups_loss]

    learning_phase = K.learning_phase()

    def gan_feed(sess, batch_image_64, batch_image_128, z_input, iteration):
        # actual GAN trainer
        nonlocal train_step,losses,noise,real_data_64, real_data_128
        nonlocal learning_phase, update_wd, update_wg, update_w_ups, update_w_ups_d, upsample, other_parameter_updates
        
        # if (iteration % 3 == 0 and iteration > 700):
        #     if upsample:
        train_step = [update_wd, update_wg, update_w_ups, update_w_ups_d, other_parameter_updates]
        #     train_step = [update_wd, update_wg, other_parameter_updates]
        # else:
        #     train_step = [update_wd, other_parameter_updates]

        res = sess.run([train_step,losses],feed_dict={
        noise:z_input,
        real_data_64:batch_image_64,
        real_data_128:batch_image_128,
        learning_phase:True,
        # Keras layers needs to know whether
        # this run is training or testring (you know, batch norm and dropout)
        })

        loss_values = res[1]
        return loss_values #[dloss,gloss]

    return gan_feed

