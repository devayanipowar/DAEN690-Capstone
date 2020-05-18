#!/usr/bin/env python

## Defining the U-Net Model
from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow as tf
import gc

class UNet(tf.keras.Model):

    def __init__(self, num_classes):
        super(UNet, self).__init__()

        self.c0 = layers.Conv2D(16, 3, padding='same', activation='elu')
        self.c1 = layers.Conv2D(32, 3, padding='same', activation='elu')
        self.c2 = layers.Conv2D(64, 3, padding='same', activation='elu')
        self.c2a = layers.Conv2D(64, 3, padding='same', activation='elu')
        self.c3 = layers.Conv2D(128, 3, padding='same', activation='elu')
        self.c3a = layers.Conv2D(128, 3, padding='same', activation='elu')
        self.c4 = layers.Conv2D(256, 3, padding='same', activation='elu')
        self.c4a = layers.Conv2D(256, 3, padding='same', activation='elu')
        self.c5 = layers.Conv2D(512, 3, padding='same', activation='elu')
        self.c5a = layers.Conv2D(512, 3, padding='same', activation='elu')

        self.up_samp = layers.UpSampling2D(size=(2,2))
        self.up_samp1 = layers.UpSampling2D(size=(2,2))
        self.up_samp2 = layers.UpSampling2D(size=(2,2))
        self.up_samp3 = layers.UpSampling2D(size=(2,2))

        self.max_pool = layers.MaxPooling2D((2,2))
        self.max_pool1 = layers.MaxPooling2D((2,2))
        self.max_pool2 = layers.MaxPooling2D((2,2))
        self.max_pool3 = layers.MaxPooling2D((2,2))

        self.d4 = layers.Conv2D(256, 3, padding='same', activation='elu')
        self.d3 = layers.Conv2D(128, 3, padding='same', activation='elu')
        self.d3a = layers.Conv2D(128, 3, padding='same', activation='elu')
        self.d2 = layers.Conv2D(64, 3, padding='same', activation='elu')
        self.d2a = layers.Conv2D(64, 3, padding='same', activation='elu')      
        self.d1 = layers.Conv2D(32, 3, padding='same', activation='elu')
        self.d1a = layers.Conv2D(32, 3, padding='same', activation='elu')
        self.d0 = layers.Conv2D(16, 3, padding='same', activation='elu')

        self.concat = layers.Concatenate(axis=-1)

        self.final_conv = layers.Conv2D(num_classes, 3, padding='same', activation="softmax")

    def conv_block(self, x, num_channels, max_pool=False):
        if max_pool:
            x = self.max_pool(x)
        x = layers.Conv2D(num_channels, 3,
                          padding="same", activation="elu")(x)
        x = layers.BatchNormalization()(x)
        return x

    def call(self, inputs):

        # Image size is 1024 x 1024 unless cropped
        # Image input [1024, 1024, n_channels]
        # n_channels is 6 - first 3 are pre, second 3 are post

        a, b = inputs

        #Beginning the downsampling
        gc.collect()
        a0, b0 = layers.BatchNormalization()(self.c0(a)), \
            layers.BatchNormalization()(self.c0(b)) # [1024, 1024, 16]
        a1, b1 = layers.BatchNormalization()(self.c1(a0)), \
            layers.BatchNormalization()(self.c1(b0)) # [1024, 1024, 32]
        del a0, b0
        a2, b2 = layers.BatchNormalization()(self.c2(self.max_pool(a1))), \
            layers.BatchNormalization()(self.c2(self.max_pool(b1))) # [512, 512, 64]
        a3, b3 = layers.BatchNormalization()(self.c2a(a2)), \
            layers.BatchNormalization()(self.c2a(b2)) # [512, 512, 64]
        del a2, b2
        a4, b4 = layers.BatchNormalization()(self.c3(self.max_pool(a3))), \
            layers.BatchNormalization()(self.c3(self.max_pool(b3))) # [256, 256, 128]
        a5, b5 = layers.BatchNormalization()(self.c3a(a4)), \
            layers.BatchNormalization()(self.c3a(b4)) # [256, 256, 128]  
        del a4, b4
        a6, b6 = layers.BatchNormalization()(self.c4(self.max_pool(a5))), \
            layers.BatchNormalization()(self.c4(self.max_pool(b5))) # [128, 128, 256]
        a7, b7 = layers.BatchNormalization()(self.c4a(a6)), \
            layers.BatchNormalization()(self.c4a(b6)) # [128, 128, 256]
        del a6, b6
        a8 = layers.BatchNormalization()(self.c5a(self.max_pool(a7))) # [64, 64, 512]
                
        # taking the output of the post image
        e9 = self.up_samp(layers.BatchNormalization()(self.d4(a8))) # [128, 128, 256]
        del a8
        gc.collect()
        d9 = layers.BatchNormalization()(self.d4(self.concat([tf.math.abs(a7-b7), e9]))) # [128, 128, 256]
        del a7, b7, e9
        d8 = self.up_samp(layers.BatchNormalization()(self.d3(d9))) # [256, 256, 128]
        del d9
        d7 = layers.BatchNormalization()(self.d3a(self.concat([tf.math.abs(a5-b5), d8]))) # [256, 256, 128])
        del a5, b5, d8
        d6 = self.up_samp(layers.BatchNormalization()(self.d2(d7))) # [512, 512, 64]
        del d7
        d5 = layers.BatchNormalization()(self.d2a(self.concat([tf.math.abs(a3-b3), d6]))) # [512, 512 64]
        del a3, b3, d6
        d4 = self.up_samp(layers.BatchNormalization()(self.d1(d5))) # [1024, 1024, 32]
        del d5
        d3 = layers.BatchNormalization()(self.d1a(self.concat([tf.math.abs(a1-b1), d4]))) # [1024, 1024, 32]
        del a1, b1, d4
        d2 = layers.BatchNormalization()(self.d0(d3)) # [1024, 1024, 32]
        del d3
        d1 = self.final_conv(d2) # [1024, 1024, num_classes]
        del d2
        gc.collect()

        return d1

    def model(self, img_shape):
        x1 = layers.Input(shape=(img_shape))
        x2 = layers.Input(shape=(img_shape))
        return tf.keras.models.Model(inputs=[x1, x2], outputs=self.call([x1, x2]))
