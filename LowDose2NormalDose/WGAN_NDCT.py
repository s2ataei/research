from __future__ import print_function, division

from keras.layers import Input, Dense, Flatten, Concatenate, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils import shuffle
import keras.backend as K
from RGANMODELS import build_generator
import matplotlib.pyplot as plt

import h5py
import numpy as np

class WGAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.latent_dim = 100
        self.df = 64
        self.gf = 64

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.img_shape))
        NDCT = Input(shape=(64,64,1))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic([img,NDCT])

        # The combined model  (stacked generator and critic)
        self.combined = Model([NDCT,z], [valid,img])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)
        

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def build_critic(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=(64,64,1))
        img_B = Input(shape=(64,64,1))

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)


        out = Flatten()(d4)
        validity = Dense(1)(out)
#        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        return Model([img_A, img_B], validity)


    def train(self, epochs, batch_size=32, sample_interval=10):

        # Load the dataset
        f = h5py.File(r'C:\Users\s2ataei\Documents\AAPM Dataset\high_all_patches.mat')
        for k, v in f.items():
             y_train = np.array(v)
             
                             
        f = h5py.File(r'C:\Users\s2ataei\Documents\AAPM Dataset\low_all_patches.mat')
        for k, v in f.items():            
             X_train = np.array(v)  

        y_train = y_train[:,:,0:281632] 
        X_train = X_train[:,:,0:281632]   
        X_train  = np.moveaxis(X_train, -1, 0)
        y_train = np.moveaxis(y_train, -1, 0)

        X_train = np.expand_dims(X_train, axis=3)
        y_train = np.expand_dims(y_train, axis=3)
        
        test_input = X_train[2896,:,:,:]
        test_input =np.expand_dims(test_input, axis=0)

        test_target = y_train[455,:,:]
        test_target  =np.expand_dims(test_target, axis=0)
        
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        
        for epoch in range(epochs):
                       
            out = self.generator.predict(test_input);
            
            fig = plt.figure()
            plt.imshow(out[0,:,:,0], cmap='gray')
            fig.savefig(r"K:\image_at_epoch%d.png" % epoch)        
            plt.show() 
            plt.close()
            
            randHigh,randLow = shuffle(y_train,X_train)
            
            imgshigh = np.array_split(randHigh,8801)
            imgslow = np.array_split(randLow,8801)
            
            idx = 0
           
            for idx in range(len(imgshigh)-5):
                for _ in range(self.n_critic):
                
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
      
                    imgs = imgshigh[idx]
                    noise = imgslow[idx]
                    noise1 = noise[:,:,:,0]
                    noise1 = np.expand_dims(noise1,axis=3)

                    # Generate a batch of new images
                    gen_imgs = self.generator.predict(noise)
    
                    # Train the critic
                    d_loss_real = self.critic.train_on_batch([imgs,noise1], valid)
                    d_loss_fake = self.critic.train_on_batch([gen_imgs,noise1], fake)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                    
                    idx = idx + 1;
     
                # ---------------------
                #  Train Generator
                # ---------------------
    
                g_loss = self.combined.train_on_batch([imgs,noise], [valid,imgs])
    
                # Print the progress
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
    
if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=20, batch_size=32 , sample_interval=10)
    wgan.generator.save_weights(r'K:\weights\MSE_MAE_diff\MSE_MAE_G.h5')
    wgan.critic.save_weights(r'K:\weights\MSE_MAE_diff\MSE_MAE_D.h5')
