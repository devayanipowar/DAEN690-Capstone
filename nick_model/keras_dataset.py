#!/usr/bin/env python3
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import cv2

class LabeledImageDataset(keras.utils.Sequence):
    """
    Image data generator for the xView2 image data
    """
    def __init__(self, num_batches=None, augmentations=None, 
                 pattern="images/*pre_disaster*.png",
                 shuffle=False, n_classes=5, batch_size=1, dim=(1024,1024),
                 n_channels=3, normalize=False):
        self.num_batches = num_batches 
        self.augmentations = augmentations
        self.pattern = pattern
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.pre = glob.glob(pattern)
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.normalize = normalize

        self.epoch_num = 0

        # Calculated the channel-wise mean and standard deviation across all
        # the training images.
        self.pre_mean = np.array([[67.45251904, 89.45889871, 86.02207342]])
        self.pre_std = np.array([[25.1677521 , 27.05929372, 30.51677248]])
        self.post_mean = np.array([[66.13790823, 87.51666638, 84.22436819]])
        self.post_std = np.array([[23.88051895, 26.1042146 , 29.05019877]])


        if num_batches:
            self.indexes = np.arange(len(self.pre))
            np.random.shuffle(self.indexes)
            self.pre = [self.pre[i] for i in self.indexes[:(num_batches*self.batch_size)]]
            

        self.post = [fn.replace("pre_disaster", "post_disaster") for fn in self.pre]
        self.prey = [
            fn.replace("_disaster", "_mask_disaster").replace("images",
            "targets") for fn in self.pre
        ]
        self.posty = [
            fn.replace("_disaster", "_mask_disaster").replace("images", "targets")
            for fn in self.post
        ]
        self.on_epoch_end()

    def __len__(self):
        return len(self.pre)//self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.pre))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, gen_idx):
        indexes = self.indexes[gen_idx*self.batch_size:(gen_idx+1)*self.batch_size]
        return self.__data_generator(indexes)

    def __data_generator(self, list_idxs):
        pre = np.empty((len(list_idxs), *self.dim, self.n_channels)).astype(np.uint8)
        post = np.empty((len(list_idxs), *self.dim, self.n_channels)).astype(np.uint8)
        y = np.empty((len(list_idxs), *self.dim)).astype(np.uint8)
        
        # Generate the data to output:
        for idx, num in enumerate(list_idxs):
            pre[idx,] = cv2.imread(self.pre[num])
            post[idx,] = cv2.imread(self.post[num])
            # only grabbing the classification channel
            # These values are 0, 63, 126, 189, 252
            y[idx,] = cv2.imread(self.posty[num])[:,:,0]

            if self.augmentations:
                augs = self.augmentations(image=pre[idx,], post_im=post[idx,],
                                         mask=y[idx,])
                pre[idx,], post[idx,], y[idx,] = augs["image"],\
                    augs["post_im"], augs["mask"]

        # Adjusting the label values so that they fall in the discrete range
        # of 0-5
        y = y.astype(np.int64) // 63
        if self.normalize: 
            # Normalizing the pre and post images based on the mean and std dev
            # of all the training images
            pre = (pre-self.pre_mean) / self.pre_std
            post = (post-self.post_mean) / self.post_std


        return [pre.astype(np.float16), post.astype(np.float16)], tf.keras.utils.to_categorical(y,
            num_classes=self.n_classes).astype(np.float16)


class TestDataset(keras.utils.Sequence):
    """
    Test image data generator for the xView2 image data
    """
    def __init__(self, 
                 pattern="images/*pre_disaster*.png",
                 n_classes=5, 
                 dim=(1024,1024),
                 n_channels=3):
        self.pattern = pattern
        self.n_classes = n_classes
        self.pre = glob.glob(pattern)
        self.batch_size = 1
        self.dim = dim
        self.n_channels = n_channels

        self.indexes = np.arange(len(self.pre))

        # Calculated the channel-wise mean and standard deviation across all
        # the training images.
        self.pre_mean = np.array([[67.45251904, 89.45889871, 86.02207342]])
        self.pre_std = np.array([[25.1677521 , 27.05929372, 30.51677248]])
        self.post_mean = np.array([[66.13790823, 87.51666638, 84.22436819]])
        self.post_std = np.array([[23.88051895, 26.1042146 , 29.05019877]])


        self.post = [fn.replace("pre_disaster", "post_disaster") for fn in self.pre]

    def __len__(self):
        return len(self.pre)

    def __getitem__(self, gen_idx):
        indexes = self.indexes[gen_idx*self.batch_size:(gen_idx+1)*self.batch_size]
        filenames = ["_".join(os.path.basename(self.pre[index]).split("_")[:2]) \
            for index in list(indexes)]
        return self.__data_generator(indexes), filenames

    def __data_generator(self, list_idxs):
        pre = np.empty((len(list_idxs), *self.dim, self.n_channels)).astype(np.uint8)
        post = np.empty((len(list_idxs), *self.dim, self.n_channels)).astype(np.uint8)
        
        # Generate the data to output:
        for idx, num in enumerate(list_idxs):
            pre[idx,] = cv2.imread(self.pre[num])
            post[idx,] = cv2.imread(self.post[num])


        # Normalizing the pre and post images based on the mean and std dev
        # of all the training images
        pre = (pre-self.pre_mean) / self.pre_std
        post = (post-self.post_mean) / self.post_std

        

        return [pre, post]
