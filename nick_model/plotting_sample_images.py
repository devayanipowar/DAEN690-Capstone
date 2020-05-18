#!/usr/bin/env python3
"""
Contains a class that easily extends plotting image masks to a pretrained
keras model.

Example:
model = pretrained_model()
model.load_weights()
file_pattern = 'path/to/image/with/*pattern*.png
image_plots = ImagePlots(file_pattern)
image_plots.plot_sample_result(model=model, target=False, normalized=True)

"""
import glob
import os
import numpy as np
import cv2
from PIL import Image

class ImagePlots:
    """
    A class used to represent a directory of images and labels to be plotted.
    This class allows images to randomly be sampled so that performance can
    be visualized.

    """
    def __init__(self, directory_pattern="images/*pre_disaster*.png"):
        """
        Parameters
        ----------
        directory_pattern : str
            Path to the images directory with the pattern needed to grab all
            of the images

        Returns
        -------
        self.pre_dir : list
            List of strings that reference the file paths for the pre-disaster
            images
        self.post_dir : list
            List of strings that reference the file paths for the post-disaster
            images
        self.posty : list
            List of strings that reference the file paths for the post-disaster
            labels
        """
        self.pre_dir = glob.glob(directory_pattern)
        self.post_dir = [fn.replace("pre_disaster", "post_disaster") for fn in
                         self.pre_dir]
        self.posty = [fn.replace("_disaster", "_mask_disaster").replace(
            "images", "targets") for fn in self.post_dir]

    def plot_sample_result(self, model, filename=None, target=False,
                           normalized=False):
        """
        Plots either the target mask or the predicted mask image over the post-
        disaster image.

        Parameters
        ----------
        model : tensorflow.python.keras.engine.training.Model
            Pretrained Keras model
        filename : str, optional
            The filename of the image to be displayed. If None (default), then
            a random image is chosen.
        target : bool, optional
            If False (default), then the predicted mask is displayed. Otherwise,
            the target mask is displayed.

        Returns
        -------
        img : PIL Image
            Returns an RGBA image in the shape [1024, 1024, 4]

        """
        idx = self.get_indices(filename)
        train_images = self.load_sample(self.pre_dir[idx], self.post_dir[idx])

        if target:
            mask = cv2.imread(self.posty[idx])[:, :, 0]//63
            mask = self.img_to_mask(mask)
        else:
            if normalized:
                norm_train_images = self.load_sample(
                    self.pre_dir[idx], self.post_dir[idx], normalize=True
                )
                mask = model.predict(norm_train_images)
            else:
                mask = model.predict(train_images)
            mask = self.img_to_mask(np.argmax(mask, axis=-1).squeeze())

        img = self.plot_results(train_images[1], mask)
        print(f"Plotting: {os.path.basename(self.post_dir[idx])}")
        return img

    def get_indices(self, filename):
        """
        Returns the index needed to subset the filename lists.

        Parameters
        ----------
        filename : str
            The filename of the image to be displayed (post-disaster).
            If None, then a random image is chosen.

        Returns
        -------
        idx : int
            The index to subset the filename list
        """
        if filename:
            idx = self.post_dir.index(filename)
        else:
            idx = np.random.randint(0, len(self.pre_dir))
        return idx

    @staticmethod
    def load_sample(pre_img, post_img, normalize=False):
        """
        Reads in the pre and post disaster images and converts them to arrays
        to feed into the model

        Parameters
        ----------
        pre_img : str
            Path to the pre-disaster image
        post_img : str
            Path to the post-disaster image

        Returns
        -------
        list with the pre and post disaster images

        """
        pre_mean = np.array([[67.45251904, 89.45889871, 86.02207342]])
        pre_std = np.array([[25.1677521, 27.05929372, 30.51677248]])
        post_mean = np.array([[66.13790823, 87.51666638, 84.22436819]])
        post_std = np.array([[23.88051895, 26.1042146, 29.05019877]])

        pre_img = cv2.imread(pre_img)[np.newaxis]
        post_img = cv2.imread(post_img)[np.newaxis]
        if normalize:
            pre_img = (pre_img - pre_mean) / pre_std
            post_img = (post_img - post_mean) / post_std


        return [pre_img, post_img]

    @staticmethod
    def img_to_mask(image_array):
        """
        Takes an input image with shape (1024, 1024) and labels 0-4 and converts
        to a mask with shape (1024, 1024, 4)

        Parameters
        ----------
        pred_img_path : str
            Path to the predicted image input.

        Returns
        -------
        masked_img : numpy array
            Returns an RGBA array in the shape [1024, 1024, 4]
        """
        damage_dict = {
            1: [0, 255, 0, 175],
            2: [0, 0, 255, 175],
            3: [239, 0, 255, 175],
            4: [255, 0, 0, 175],
        }

        masked_image = np.zeros((1024, 1024, 4))
        for key, value in damage_dict.items():
            masked_image[image_array == key] = value
        return masked_image.astype(np.uint8)

    @staticmethod
    def plot_results(input_image, input_label):
        """
        Takes an input image and the corresponding label as arrays and outputs
        images with the overlap showing the masked regions.


        Parameters
        ----------
        input_image : array or tensor
            RGB array with the shape [image_height, image_width, 3] representing the
            original image.
        input_label : array or tensor
            Mask array with the shape [image_height, image_width, num_classes]
            representing the masked regions.
        normalized : boolean
            Whether the input_image data were normalized.
        mean : string
            If the input_image data were normalized, the path to the mean.npy file
            needs to be provided to un-normalized the data.

        Returns
        -------
        img : PIL Image
            Returns an RGBA array in the shape [1024, 1024, 4]

        """
        if len(input_image.shape) == 4:
            input_image = input_image.squeeze(0)
        input_image = input_image.astype(np.uint8)

        mask = Image.fromarray(input_label).convert('L')
        mask = np.array(mask)
        mask[mask == 0] = 255
        mask[mask < 255] = 0

        img = Image.composite(Image.fromarray(input_image),
                              Image.fromarray(input_label),
                              Image.fromarray(mask))
        return img
