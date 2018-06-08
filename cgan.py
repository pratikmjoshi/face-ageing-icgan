import cv2
from keras.models import Model,Sequential
from keras.layers import Embedding,BatchNormalization,ZeroPadding2D,Dense,Activation,RepeatVector
from keras.layers import add,Flatten,Conv2D,Conv2DTranspose,Input,UpSampling2D,Reshape,multiply,LeakyReLU,Dropout,Concatenate,concatenate
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical

import keras.backend as K

import numpy as np
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

        #Discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = ['binary_crossentropy'],optimizer = Adam(0.0002,0.5),metrics=['accuracy'])

        #Generator
        self.generator = self.build_generator()

        noise = Input(shape=(self.latent_dims,))
        label = Input(shape=(self.num_classes,))
        img = self.generator([noise,label])

        self.discriminator.trainable = False

        valid = self.discriminator([img,label])

        #Combined model
        self.combined = Model([noise,label],valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer= Adam(0.0002,0.5))


    def build_generator(self):

        model = Sequential()

        model.add(Dense(128*16*16,input_dim=self.latent_dims+self.num_classes,activation='relu'))
        model.add(Reshape((16,16,128)))

        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3,padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.img_channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape = (self.latent_dims,))
        label = Input(shape = (self.num_classes,))

        #label_embedding = Flatten()(Embedding(self.num_classes,self.latent_dims)(label))

        model_input = concatenate([noise,label])
        img = model(model_input)

        return Model([noise,label],img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32,kernel_size=3,strides=2,input_shape=(self.img_height,self.img_width,self.img_channels+self.num_classes),padding='same'))
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
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(self.num_classes,))


        #label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        #flat_img = Flatten()(img)
        label_2d = RepeatVector(self.img_width)(label)
        label_2d_array = [label_2d for _ in range(self.img_height)]
        label_3d = Lambda(lambda x: K.stack(x,axis=1))(label_2d_array)
        #label_3d = concatenate(label_2d_array,axis=1)

        model_input = Concatenate(axis=3)([img,label_3d])
        print(model_input.shape)
        #reshaped_model_input = Reshape(self.img_shape)(model_input)

        validity = model(model_input)

        return Model([img, label], validity)



    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        #(X_train, y_train), (_, _) = mnist.load_data()
        X_train = np.load('images.npy')
        y_train = np.load('labels.npy')

        # Rescale -1 to 1
        X_train = X_train / 255.
        #X_train = np.expand_dims(X_train, axis=3)
        print(X_train.shape)
        y_train = to_categorical(y_train)
        print(y_train.shape)
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs,labels = X_train[idx],y_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dims))
            gen_imgs = self.generator.predict([noise,labels])

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch([imgs,labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs,labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, self.num_classes, batch_size)
            sampled_labels = to_categorical(sampled_labels,num_classes = self.num_classes)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dims))
        sampled_labels = np.arange(0, self.num_classes)
        sampled_labels = to_categorical(sampled_labels,num_classes = self.num_classes)
        gen_imgs = self.generator.predict([noise,sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(cv2.cvtColor((gen_imgs[cnt, :,:]*255.).astype(np.float32),cv2.COLOR_BGR2RGB))
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/face_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=100, batch_size=32, save_interval=10)
