#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import cv2
from keras_dataset import TestDataset
from dual_image_unet import UNet

def convert_for_eval(pre_filenames, post_filenames):
    """
    Takes a list of pre disaster and post disaster filenames and converts the
    filenames and image formats so that they can be fed into the evaluation
    script.

    Parameters
    ----------
    pre_filenames : List[str]
        A list of all the pre disaster file paths.
    post_filenames : List[str]
        A list of all the post disaster file paths.

    Returns
    -------
    filemap_name : dict
        Dictionary that with the original file basepath as keys and the index as
        values, so that the prediction images can be mapped to the correct index.

    Examples:
    -------
    pre_labels = "path_to_test_label_images/*pre_mask_disaster*.png"
    pre_label_list = glob.glob(pre_labels)
    post_label_list = [fn.replace("pre_mask_disaster", "post_mask_disaster")
                       for fn in pre_label_list]
    final_map = convert_for_eval(pre_label_list, post_label_list)

    """
    filename_map = {}
    assert len(pre_filenames) == len(post_filenames)
    for i in tqdm(range(len(pre_filenames))):
        filename_map[
            "_".join(os.path.basename(pre_filenames[i]).split("_")[:2])] = i
        change_img_format(pre_filenames[i])
        change_img_format(post_filenames[i])
        change_filename(pre_filenames[i], "localization", i, "target")
        change_filename(post_filenames[i], "damage", i, "target")

    return filename_map

def change_filename(file, loc_or_dmg, idx, targ_or_pred, test_or_hold="test"):
    """
    Converts the filename in the format necessary for scoring.
    """
    new_name = f"{test_or_hold}_{loc_or_dmg}_{idx}_{targ_or_pred}.png"
    os.rename(file, os.path.join(os.path.dirname(file), new_name))

def change_img_format(file):
    """
    Changes the image format so that it"s a (1024, 1024) shaped image where
    each pixel represents the class label.
    """
    img = cv2.imread(file)
    cv2.imwrite(file, img[:, :, 0] // 63)


def run_test_inference(pre_labels, model_resume, test_dir, prediction_dir):
    """
    Outputs the model predictions for each test image in the specified
    prediction_dir.

    Parameters
    ----------
    pre_labels : str
        Path to the directory with the target image labels
    model_resume : str
        Path to the hdf5 file containing the model weights
    test_dir : str
        Path to the directory with the test images
    prediction_dir : str
        Path to the directory where the prediction images will be saved
    Returns
    -------
    None.

    """
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    pre_label_list = glob.glob(
        os.path.join(pre_labels, "*pre_mask_disaster*.png")
    )
    post_label_list = [
        fn.replace("pre_mask_disaster", "post_mask_disaster")
        for fn in pre_label_list
    ]

    final_map = convert_for_eval(pre_label_list, post_label_list)
    test_dataset = TestDataset(os.path.join(test_dir, "*pre_disaster*.png"))
    pretrained_model = UNet(num_classes=5).model((None, None, 3))
    pretrained_model.load_weights(model_resume)

    for i in tqdm(range(len(test_dataset))):
        images, filename = test_dataset[i]
        preds = pretrained_model.predict(images)

        dmg_preds = (np.argmax(preds, axis=-1).squeeze()).astype(np.uint8)

        loc_preds = dmg_preds.copy()
        loc_preds[loc_preds > 0] = 1

        image_index = final_map[filename[0]]


        # write localization
        cv2.imwrite(
            os.path.join(prediction_dir, "test_localization_{}_prediction.png"
                         .format(image_index)), loc_preds
        )

        # write damage
        cv2.imwrite(
            os.path.join(prediction_dir, "test_damage_{}_prediction.png"
                         .format(image_index)), dmg_preds
        )

    print(f"-----Images Written To: {prediction_dir}-----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-dir", help="Path to the directory "
                        " containing the target image labels")
    parser.add_argument("--resume", help="Path to the file containing the "
                        "saved model weights")
    parser.add_argument("--test-imgs", help="Path to the directory containing "
                        "the test images")
    parser.add_argument("--pred-dir", help="Path to the directory where you'd "
                        "like the predicted images saved")            
    args = parser.parse_args()

    run_test_inference(args.targets_dir, args.resume, args.test_imgs,
                       args.pred_dir)
