import model
import images
import glob
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

INPUT_SHAPE = (96,96,3)
FPATH = 'data96/train/*.jpg'
batch_size = 100
ep=100000
zed=1024
print("test")

if __name__ == '__main__':
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )

    X_train = images.load_images(FPATH)

    gen_model = model.gen(INPUT_SHAPE, 1024, batch_size)
    dis_model = model.dis(INPUT_SHAPE)
    gan_feed = model.gan(gen_model,dis_model, batch_size)
    ups_model = model.upsample_gen(INPUT_SHAPE)

    print('Generator...')
    gen_model.summary()
    print('Discriminator...')
    dis_model.summary()
    print('Upsampler...')
    ups_model.summary()


    sess = K.get_session()
    for i in range(ep):
        print('---------------------------')
        print('iter',i)

        z_input = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))
        minibatch = datagen.flow(X_train,batch_size = batch_size)

        # train for one step
        losses = gan_feed(sess,minibatch[0],z_input,i)
        print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

        if i==ep-1 or i % 100 == 0: images.show(i, gen_model, 100, INPUT_SHAPE)

    print("train upsampling")
    for i in range(ep):
        print('---------------------------')
        print('iter',i)

        z_input = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))
        minibatch = datagen.flow(X_train,batch_size = batch_size)

        # train for one step
        losses = gan_feed(sess,minibatch,z_input,i)
        print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

        if i==ep-1 or i % 100 == 0: images.show(i, gen_model, 100, INPUT_SHAPE)
    

