#!/usr/bin/env python3

"""Preprocessing file for the xView2 data.
This file, when executed, creates [1024,1024,3] shaped images as masks for
both the pre and post disaster labels. The third dimension (channels) varies
for both:
For the localization images, the first channel represents the presence of a
building (0 or 255). Channel 2 represents the presence of an edge (0 or 255).
Channel 3 represents a heatmap for the edges (0, 63, 129, 189, 252).
For the damage images, the first channel represents the level of damage
(63, 126, 189, 252). Channel 2 represents the presence of an edge (0 or 255).
Channel 3 represents a heatmap for the edges (0, 63, 129, 189, 252).
Example:
    In order to run this from the command line, execute like below:
        $ python new_unet_preprocess.py /path/to/train_or_test_image_dir
"""
import glob
import json
import os
import argparse
import numpy as np
import cv2
from shapely import wkt
from tqdm import tqdm

def shrinkpolygon(polygon, border=2):
    """
    Controls the size of the border, and if the polygon is below a certain size,
    the border is set to zero. This is in an attempt to increase the clarity
    of the masks.
    """
    if polygon.area < 37:
        border = 0
    cx, cy = polygon.centroid.coords[0]
    pts = polygon.exterior.coords

    def shrink(pt):
        (x, y) = pt
        if x < cx:
            x += border
        elif x > cx:
            x -= border

        if y < cy:
            y += border
        elif y > cy:
            y -= border
        return [int(x), int(y)]

    return np.array([shrink(pt) for pt in pts])

def getjsons(pattern):
    """
    Gets a list of json files from the pattern supplied.
    """
    return glob.glob(pattern)

def json2dict(filename):
    """
    Loads the json file from the filename
    """
    with open(filename) as f:
        j = json.load(f)
    return j

def prejson2img(filename):
    """
    Converts the pre disaster images to image masks.
    """
    j = json2dict(filename)
    polygons = []
    for feat in j["features"]["xy"]:
        polygon = wkt.loads(feat["wkt"])
        coords = shrinkpolygon(polygon)
        polygons.append(coords)
    blank = np.zeros((1024, 1024, 1)) # represents the presence of a building
    cv2.fillPoly(blank, polygons, (255, 255, 255))
    edgelines = np.zeros((1024, 1024, 1))
    cv2.polylines(edgelines, polygons, True, (255, 255, 255), thickness=2)
    edgeheatmap = np.zeros((1024, 1024, 1))
    cv2.polylines(edgeheatmap, polygons, True, (63, 63, 63), thickness=13)
    cv2.polylines(edgeheatmap, polygons, True, (126, 126, 126), thickness=9)
    cv2.polylines(edgeheatmap, polygons, True, (189, 189, 189), thickness=5)
    cv2.polylines(edgeheatmap, polygons, True, (252, 252, 252), thickness=3)
    return blank, edgelines, edgeheatmap, len(polygons)

totalarea, area1, area2, area3, area4 = 0, 0, 0, 0, 0

def postjson2img(filename):
    """
    Converts the post disaster images to image masks.
    """
    global totalarea, area1, area2, area3, area4
    j = json2dict(filename)
    polygons = []
    damages = []
    for feat in j["features"]["xy"]:
        polygon = wkt.loads(feat["wkt"])
        coords = polygon
        polygons.append(coords)
        damages.append(feat["properties"]["subtype"])

    # polytypes0 = [polygons[i] for i, d in enumerate(damages) if d == "un-classified"]
    polytypes1 = [polygons[i] for i, d in enumerate(damages) if d == "no-damage"]
    polytypes2 = [polygons[i] for i, d in enumerate(damages) if d == "minor-damage"]
    polytypes3 = [polygons[i] for i, d in enumerate(damages) if d == "major-damage"]
    polytypes4 = [polygons[i] for i, d in enumerate(damages) if d == "destroyed"]

    totalarea += 1024 * 1024
    area1 += sum(p.area for p in polytypes1)
    area2 += sum(p.area for p in polytypes2)
    area3 += sum(p.area for p in polytypes3)
    area4 += sum(p.area for p in polytypes4)

    # represents the damage level (divide by 63 to get 0-4 range)
    blank = np.zeros((1024, 1024, 1))
    color = (63, 63, 63)
    cv2.fillPoly(blank, [shrinkpolygon(p) for p in polytypes1], color)
    color = (126, 126, 126)
    cv2.fillPoly(blank, [shrinkpolygon(p) for p in polytypes2], color)
    color = (189, 189, 189)
    cv2.fillPoly(blank, [shrinkpolygon(p) for p in polytypes3], color)
    color = (252, 252, 252)
    cv2.fillPoly(blank, [shrinkpolygon(p) for p in polytypes4], color)

    polygons = [
        shrinkpolygon(p) for p in polytypes1 + polytypes2 + polytypes3 + polytypes4
    ]

    edgelines = np.zeros((1024, 1024, 1))
    cv2.polylines(edgelines, polygons, True, (255, 255, 255), thickness=9)

    edgeheatmap = np.zeros((1024, 1024, 1))
    cv2.polylines(edgeheatmap, polygons, True, (63, 63, 63), thickness=13)
    cv2.polylines(edgeheatmap, polygons, True, (126, 126, 126), thickness=9)
    cv2.polylines(edgeheatmap, polygons, True, (189, 189, 189), thickness=5)
    cv2.polylines(edgeheatmap, polygons, True, (252, 252, 252), thickness=3)
    return blank, edgelines, edgeheatmap, len(polygons)

def prepostjson2img(prefilename):
    """
    Saves the newly created pre and post disaster masks into a new 'targets'
    directory.
    """
    postfilename = prefilename.replace("pre_disaster", "post_disaster")
    preloc, preedge, preweight, precount = prejson2img(prefilename)
    postdmg, postedge, postweight, postcount = postjson2img(postfilename)
    prefilename = (
        prefilename.replace("pre_disaster", "pre_mask_disaster")
        .replace(".json", ".png")
        .replace("labels", "targets")
    )
    postfilename = (
        postfilename.replace("post_disaster", "post_mask_disaster")
        .replace(".json", ".png")
        .replace("labels", "targets")
    )
    premask = np.concatenate([preloc, preedge, preweight], axis=2)
    cv2.imwrite(prefilename, premask)
    postmask = np.concatenate([postdmg, postedge, postweight], axis=2)
    cv2.imwrite(postfilename, postmask)
    return precount, postcount


def calc_image_mean(dataloader):
    """
    Computing the image mean for each channel
    """
    sum_color_pre = 0
    sum_color_post = 0
    print("computing the image mean")
    N = len(dataloader)
    for image, _ in tqdm(dataloader):
        sum_color_pre += image[0].mean(axis=(1, 2), keepdims=False)
        sum_color_post += image[1].mean(axis=(1, 2), keepdims=False)
    sum_color_pre = sum_color_pre / N
    sum_color_post = sum_color_post / N

    np.save("mean_pre", sum_color_pre)
    np.save("mean_post", sum_color_post)

def calc_image_std(dataloader):
    """
    Computing the average image standard deviation for each channel
    """
    sum_color_pre = 0
    sum_color_post = 0
    print("computing the image mean")
    N = len(dataloader)
    for image, _ in tqdm(dataloader):
        sum_color_pre += image[0].std(axis=(1, 2), keepdims=False)
        sum_color_post += image[1].std(axis=(1, 2), keepdims=False)
    sum_color_pre = sum_color_pre / N
    sum_color_post = sum_color_post / N

    np.save("std_pre", sum_color_pre)
    np.save("std_post", sum_color_post)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help="""Path to the directory containing
                        the image and label folders""")
    args = parser.parse_args()

    pattern = os.path.join(args.directory, "images", "*pre*")
    emptyimages = []
    for f in tqdm(getjsons(pattern)):
        fn = f.replace("images", "labels").replace(
            "png", "json"
        )
        precount, postcount = prepostjson2img(fn)
        if not precount or not postcount:
            emptyimages.append(fn + "\n")

    with open(os.path.join(args.directory, "emptyimages.txt"), "w") as f:
        f.writelines(emptyimages)

    print(
        "Total area: {} , Pixel class fractions: no-damage {} , "
        "minor-damage {} , major-damage {} , destroyed {} ".format(
            totalarea,
            area1 / totalarea,
            area2 / totalarea,
            area3 / totalarea,
            area4 / totalarea,
        )
    )
