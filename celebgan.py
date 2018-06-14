import cv2
from keras.models import Model,Sequential
from keras.layers import Embedding,BatchNormalization,ZeroPadding2D,Dense,Activation,RepeatVector
from keras.layers import add,Flatten,Conv2D,Conv2DTranspose,Input,UpSampling2D,Reshape,multiply,LeakyReLU,Dropout,Concatenate,concatenate
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical

import keras.backend as K

from keras.models import load_model

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CGAN():
    def __init__(self):

        #Input Shape
        self.img_height = 64
        self.img_width = 64
        self.img_channels = 3
        self.img_shape = (self.img_height,self.img_width,self.img_channels)
        self.num_classes = 4

        #Latent Dimensions
        self.latent_dims = 100

        optimizer = Adam(0.0002,0.5)

        #Discriminator
        #self.discriminator = self.build_discriminator()
        self.discriminator = load_model('/u/discriminator.hd5')
        self.discriminator.compile(loss = ['binary_crossentropy'],optimizer = optimizer,metrics=['accuracy'])

        #Generator
        #self.generator = self.build_generator()
        self.generator = load_model('/u/generator.hd5')


        noise = Input(shape=(self.latent_dims,))
        img = self.generator(noise)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        #Combined model
        self.combined = Model(noise,valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer= optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256*4*4,input_dim=self.latent_dims,activation='relu'))
        model.add(Reshape((4,4,256)))
        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.img_channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape = (self.latent_dims,))
        img = model(noise)

        return Model(noise,img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32,kernel_size=3,strides=2,input_shape=self.img_shape,padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)

        validity = model(img)

        return Model(img, validity)



    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        #(X_train, y_train), (_, _) = mnist.load_data()
        X_train = np.load('/t/images.npy')


        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        #X_train = np.expand_dims(X_train, axis=3)
        #print(X_train.shape)
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dims))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.generator.save('/output/generator.hd5')
                self.discriminator.save('/output/discriminator.hd5')



    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dims))
        sampled_labels = np.arange(0, self.num_classes)
        sampled_labels = to_categorical(sampled_labels,num_classes = self.num_classes)
        gen_imgs = self.generator.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(cv2.cvtColor(((gen_imgs[cnt, :,:]+1)*127.5).astype(np.float32),cv2.COLOR_BGR2RGB))
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/output/face_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=10000, batch_size=32, save_interval=100)
