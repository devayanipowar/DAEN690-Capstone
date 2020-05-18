#!/usr/bin/env python3
import os
import argparse
import sys
import datetime
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


from albumentations import (
    ISONoise,
    RandomFog,
    VerticalFlip,
    Compose,
    RandomRotate90,
    RandomBrightnessContrast,
    RandomGamma,
)

from keras_dataset import LabeledImageDataset
from dual_image_unet import UNet

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    """
    Callback for Keras that outputs the loss after each batch, and the loss and
    categorical accuracy after each epoch.
    """
    def on_train_batch_end(self, batch, logs={}):
        print('\tFor batch {}, loss is {:7.2f}.'
              .format(batch, logs.get("loss"))
             )

    def on_test_batch_end(self, batch, logs={}):
        print('\tFor batch {}, loss is {:7.2f}.'
              .format(batch, logs.get("loss"))
             )

    def on_epoch_end(self, epoch, logs={}):
        print("\tThe average loss for epoch {} is {:7.2f} and accuracy is "
              "{:7.2f}."
              .format(epoch, logs.get("loss"), logs.get('categorical_accuracy'))
             )


class CombinedLoss:
    """
    Generates an equal combination of weighted cross entropy loss and dice loss.
    """
    def __init__(self, class_weights):
        self.class_weights = class_weights

    def cross_entropy_loss(self, y_true, y_pred):
        # Prevent inf values from occurring
        y_pred = tf.clip_by_value(y_pred,
                                  tf.keras.backend.epsilon(),
                                  (1. - tf.keras.backend.epsilon())
        )
        # Cross entropy part - incorporating the weights
        loss = y_true * tf.math.log(y_pred) * self.class_weights
        # Summing across channels to return pixel-wise cross entropy
        loss = -tf.math.reduce_sum(loss, axis=-1)
        return tf.reduce_mean(loss)

    def dice_loss(self, y_true, y_pred):
        numerator = 2. * tf.math.reduce_sum(y_pred * y_true, axis=-1)
        denominator = tf.reduce_sum(y_pred + y_true, axis=-1)
        # adding 1 for numerical stability
        return tf.reduce_mean(1 - (numerator + 1) / (denominator + 1))

    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.cross_entropy_loss(y_true, y_pred) + \
            self.dice_loss(y_true, y_pred)

def save_model(model, pause=0):
    """
    Defining a function that saves the model on keyboard interrupt, so progress
    isn't completely lost.
    """
    if pause > 0:
        sys.stderr.write("Saving model weights to file.")
        for i in list(range(0, 6))[::-1]:
            sys.stderr.write(f"{i}...\n")
            time.sleep(pause)
    sys.stderr.write("Saving file...")
    return model.save_weights('interrupted_model.hdf5')

def train(file_pattern, train_num_batches=None, train_aug=False,
          train_batch_size=1, val_batch_size=1, learning_rate=1e-3,
          epochs=1, verbosity=2, file_directory=None, resume=None,
          train_shuffle=True, pre_image_mean=None, post_image_mean=None):
    """
    Function to train the UNet model
    Parameters
    ----------
    file_pattern : string
        Location where the image folder is for the data. Example format:
        "images/*pre_disaster*.png"
    train_num_batches : int
        Number of batches for the training set, if none, the full dataset will
        be used.
    train_aug : bool
        If true, augmentations are performed.
    train_batch_size : int, default 5
        Batch size for the training set.
    val_batch_size : int, default 5
        Batch size for the validation set.
    learning_rate : float, default 0.00001
        Learning rate for the UNet.
    epochs : int, default 1
        How many epochs for the training to run.
    verbosity : int, default 2
        How verbose you'd like the output to be.
    file_directory : string, default None:
        Directory where you'd like the output files saved.
    resume : string, default None
        Enter in a string for the saved model file and training will resume
        from this instance.
    train_shuffle : bool
        If True, the training data is shuffled for each epoch.
    pre_image_mean : str
        The filepath for the pre image mean numpy array file.
    post_image_mean : str
        The filepath for the post image mean numpy array file.
    Returns
    -------
    Saves the model weights, csv logs, and tensorboard files in the original
    directories specified.

    """
    if file_directory is None:
        file_directory = os.path.abspath(
            os.path.join(os.getcwd(), "saved_models")
        )

    tensorboard_path = os.path.join(
        file_directory, "logs", "tboard_{}"
        .format(datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    )
    weights_path = os.path.join(
        file_directory, "unet_weights_{}"
        .format(datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    )
    csv_logger_path = os.path.join(
        file_directory, "log_unet_{}{}"
        .format(datetime.datetime.now().strftime("%Y%m%d-%H%M"), ".csv")
    )

    if train_aug:
        train_augs = Compose(
            [
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                ISONoise(p=0.5),
                RandomBrightnessContrast(p=0.5),
                RandomGamma(p=0.5),
                RandomFog(fog_coef_lower=0.025, fog_coef_upper=0.1, p=0.5),

            ]
        )

    else:
        train_augs = None

    # Weighted categorical cross entropy weights
    # class_weights = tf.constant([0.1, 1.0, 2.0, 2.0, 2.0])
    # class_weights = tf.constant([1.0, 1.0, 0.5, 0.5, 0.5])
    class_weights = tf.constant([1.0, 1.0, 3.0, 3.0, 3.0])


    train_data = LabeledImageDataset(num_batches=train_num_batches,
                                     augmentations=train_augs,
                                     pattern=file_pattern,
                                     shuffle=train_shuffle,
                                     n_classes=5,
                                     batch_size=train_batch_size,
                                     normalize=True
    )

    # Using random samples from train for validation
    val_data = LabeledImageDataset(num_batches=100,
                                   augmentations=train_augs,
                                   pattern=file_pattern,
                                   shuffle=train_shuffle,
                                   n_classes=5,
                                   batch_size=val_batch_size,
                                   normalize=True
    )
    if resume:
        try:
            print("the pretrained model was loaded")
            model = UNet(num_classes=5).model((None, None, 3))
            model.load_weights(resume)
        except OSError:
            print("The model file could not be found. "
                  "Starting from a new model instance")
            model = UNet(num_classes=5).model((None, None, 3))
    else:
        model = UNet(num_classes=5).model((None, None, 3))


    metrics = [tf.keras.metrics.CategoricalAccuracy()]
    for i in range(5):
        metrics.append(Precision(class_id=i, name=f"prec_class_{i}"))
        metrics.append(Recall(class_id=i, name=f"rec_class_{i}"))

    model.compile(optimizer=keras.optimizers.RMSprop(lr=learning_rate),
                  loss=CombinedLoss(class_weights),
                  metrics=metrics)


    # Creating a checkpoint to save the model after every epoch if the
    # validation loss has decreased
    model_checkpoint = ModelCheckpoint("dual_unet_{epoch:02d}-{loss:.2f}.hdf5",
                                       monitor='loss',
                                       save_best_only=False,
                                       mode='min',
                                       save_weights_only=True,
                                       verbose=verbosity)

    csv_logger = CSVLogger(csv_logger_path, append=True, separator=',')


    lr_logger = ReduceLROnPlateau(monitor='loss',
                                  factor=0.2,
                                  patience=1,
                                  verbose=verbosity,
                                  mode='min',
                                  min_lr=1e-6)

    tensorboard_cb = TensorBoard(log_dir=tensorboard_path,
                                 write_images=True)



    try:
        model.fit(train_data,
                  epochs=epochs,
                  verbose=verbosity,
                  callbacks=[LossAndErrorPrintingCallback(),
                             model_checkpoint,
                             csv_logger,
                             lr_logger,
                             tensorboard_cb],
                  validation_data=val_data,
                  workers=6
        )

    except KeyboardInterrupt:
        save_model(model, pause=1)
        sys.exit()
    except Exception as exc:
        save_model(model, pause=0)
        raise exc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--file-pattern', help='''Pattern for directory
                         containing images - "images/*pre_disaster*.png''')
    parser.add_argument('--train-num-batches', type=int, default=None,
                        help="Number of batches for input images")
    parser.add_argument('--train-aug', type=bool, default=False,
                        help="If True, the training images are augmented.")
    parser.add_argument('--train-batch-size', type=int, default=1,
                        help='Number of images in each training batch.')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        help='Number of images in each validation batch.')
    parser.add_argument('--learning-rate', type=int, default=0.0001,
                        help='Learning rate for the UNet.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs over the training set.')
    parser.add_argument('--verbosity', type=int, default=2,
                        help='How much output text you would like displayed.')
    parser.add_argument('--file-directory', default=None,
                        help="Directory to save all model files.")
    parser.add_argument('--resume', default=None,
                        help="Location of the trained model file to load.")
    parser.add_argument('--train-shuffle', default=False,
                        help="If True, the training data are shuffled.")
    args = parser.parse_args()

    train(file_pattern=args.file_pattern,
          train_num_batches=args.train_num_batches,
          train_aug=args.train_aug,
          train_batch_size=args.train_batch_size,
          val_batch_size=args.val_batch_size,
          learning_rate=args.learning_rate,
          epochs=args.epochs,
          verbosity=args.verbosity,
          file_directory=args.file_directory,
          resume=args.resume,
          train_shuffle=args.train_shuffle)
