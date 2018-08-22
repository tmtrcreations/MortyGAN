#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 2018

Author: 
Modified By: tmtrcreations
"""

# --------------------------
# Import the needed modules
# --------------------------

from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, Activation, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.layers.noise import GaussianNoise
from keras import backend as K
import tensorflowjs as tfjs
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -----------
#  GAN Class 
# -----------

class GAN():
    
    # --------------------
    #  Define Initializer 
    # --------------------
    
    def __init__(self):
        self.img_rows = 160
        self.img_cols = 160
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.stddev = K.variable(K.cast_to_floatx(10))

        optimizer_d = SGD(0.002)
        optimizer_g = Adam(0.002)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer_d,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        img = GaussianNoise(stddev=self.stddev)(img)

        # For the combined model we will only train the generator
        validity = self.discriminator
        validity.trainable = False
        # The discriminator takes generated images as input and determines validity
        validity = validity(img)
        

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_g)

    # ------------------
    #  Define Generator 
    # ------------------

    def build_generator(self):

        model = Sequential()
        
        model.add(Dense(128 * 20 * 20, activation='relu', input_dim=self.latent_dim))
        model.add(Reshape((20, 20, 128)))
        model.add(UpSampling2D())
        
        model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))  
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        
        model.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        
        model.add(Conv2D(self.channels, kernel_size=3, strides=1, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)
    
    # ----------------------
    #  Define Discriminator 
    # ----------------------

    def build_discriminator(self):

        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), padding='valid', input_shape=self.img_shape))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(AveragePooling2D((3, 3)))
        
        model.add(Conv2D(64, (3, 3), padding='valid'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(AveragePooling2D((3, 3)))
        
        model.add(Conv2D(128, (3, 3), padding='valid'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(AveragePooling2D((3, 3)))
        
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
    
    # -----------------
    #  Define Training 
    # -----------------

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = 255*np.ones((195, self.img_rows, self.img_cols, self.channels))
        count = 0
        for image_ind in range(1, 199):
            if((image_ind != 115) & (image_ind != 116) & (image_ind != 146)):
                img = Image.open("../../Datasets/Sample_Sprites/Morty" + "%03d" % image_ind + ".png")
                X_train[count, 17:142, 6:-6, :] = np.array(img)
                count = count + 1

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            if epoch < 100:
                stddev = (10*(np.cos(np.pi*epoch/100)+1)/2)
                K.set_value(self.stddev, K.cast_to_floatx(stddev))
                imgs = imgs + np.random.normal(0, stddev, imgs.shape)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (to have the discriminator label samples as valid)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                
            if epoch % 1000 == 0:
                gan.generator.save("../../Trained_Models/Aug21_e" + str(epoch) + ".h5")
                tfjs.converters.save_keras_model(gan.generator, "../../Trained_Models/Aug21_e" + str(epoch) + ".json")

    # --------------------------
    #  Define Sample Generation 
    # --------------------------

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, :])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("../../Results/Sample_Sprites/%d.png" % epoch)
        plt.close()

# ------
#  Main
# ------
        
if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=100000, batch_size=32, sample_interval=200)
    gan.generator.save("../../Trained_Models/Aug21.h5")
    tfjs.converters.save_keras_model(gan.generator, "../../Trained_Models/Aug21.json")
