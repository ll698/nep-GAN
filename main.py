import model
import images
import glob
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
import keras.backend as K

INPUT_SHAPE_64 = (64,64,3)
INPUT_SHAPE_128 = (128, 128, 3)
FPATH64 = 'data/data64/train/*.jpg'
FPATH128 = 'data/data128/train/*.jpg'
batch_size = 100
ep=100000
zed=256
print("test")

if __name__ == '__main__':
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )

    X_train_64 = images.load_images(FPATH64)
    X_train_128 = images.load_images(FPATH128)

    gen_model = model.gen(INPUT_SHAPE_64, 256, batch_size)
    dis_model = model.dis(INPUT_SHAPE_64)
    ups_gen_model = model.upsample_gen(INPUT_SHAPE_64, INPUT_SHAPE_128, batch_size)
    ups_dis_model = model.dis(INPUT_SHAPE_128)
    gan_feed = model.gan(gen_model, dis_model, batch_size, ups_gen_model, ups_dis_model, upsample=True)

    print('Generator...')
    gen_model.summary()
    print('Discriminator...')
    dis_model.summary()
    print('Upsampler...')
    ups_gen_model.summary()
    

    sess = K.get_session()
    for i in range(ep):
        print('---------------------------')
        print('iter',i)


        z_input = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))
        minibatch_64 = datagen.flow(X_train_64, batch_size = batch_size)
        minibatch_128 = datagen.flow(X_train_128, batch_size = batch_size)

        # train for one step
        losses = gan_feed(sess, minibatch_64[0], minibatch_128[0], z_input, i)
        print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

        if i==ep-1 or i % 100 == 0: images.show(i, gen_model, 100, INPUT_SHAPE_64, ups_gen_model, upsample=True)


