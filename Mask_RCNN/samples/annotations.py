import json
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_poly_coordinates(folder_path):
    """
    Load pixel coordinates as a list of list from the csv file

    """
    df = pd.read_csv(folder_path)

    # Get pixel coordinates of the building
    s = str(df["polygon_x_y"])
    s = re.search("\((.+?)\)", s).group(0)[2:-1]  # Strip out POLYGON and other stuff

    #print(s)

    bldg_pix_co = [[float(i) for i in x.split()] for x in s.split(',')]

    return bldg_pix_co


def get_bbox_coordinates(polypix, bbox_format="XYWH"):
    """
    Get bbox coordinates from the minimum rectangle that encloses the polygon i.e. get the bbox of a single building in the scene

    polypix     : Polygon Pixels: list of lists format - #bldgpixco
    bbox_format : XYWH  -> Displays bbox coordinates in the format    : [xmin, ymin, width, height]
                : LTRB  -> Displays bbox coordinates in the format    : [xmin, ymax, xmax, ymin]
                : CWH  -> Displays bbox coordinates in the format    : [xcenter, ycenter, width, height]
                : all   -> Displays all bbox coordinates in the format: [(x1,y1),(x2,y2),(x3,y3),(x1,y1)]

    Returns bbox coordinates of the format: [xmin,ymin,width,height]
    """

    gpbc = np.array(poly2rect(polypix))

    xmin = np.min(gpbc[:, 0])
    xmax = np.max(gpbc[:, 0])
    ymin = np.min(gpbc[:, 1])
    ymax = np.max(gpbc[:, 1])

    xcenter = (xmin + xmax) / 2
    ycenter = (ymin + ymax) / 2

    width = round(xmax - xmin, 3)
    height = round(ymax - ymin, 3)

    if bbox_format == "XYWH":
        bbox_coordinates = [xmin, ymin, width, height]
    elif bbox_format == "LTRB":
        bbox_coordinates = [xmin, ymax, xmax, ymin]
    elif bbox_format == "CWH":
        bbox_coordinates = [xcenter, ycenter, width, height]
    elif bbox_format == "all":
        bbox_coordinates = [gpbc[0][0], gpbc[0][1], gpbc[1][0], gpbc[1][1], gpbc[2][0], gpbc[2][1], gpbc[3][0],
                            gpbc[3][1]]

    return bbox_coordinates


def poly2rect(polypixlist):
    """
    Converts polygon coordinates to minimum bounding rectangle coordinates

    polypixlist : polygon coordinates (list of lists) - #bldgpixco
    Returns : minimum bounding rectangle coordinates (list of lists)


    """

    geom = polypixlist
    mabr = minimum_bounding_rectangle(np.array(geom))
    return mabr.tolist()


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
    TODO: Incorporate orientation
    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """

    from scipy.spatial import ConvexHull
    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)


    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def PolyArea(x, y):
    """
    Calculates area of the polygon
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def coco_ann(bldg_set_id, category_id, polypix, obj_id):
    """
    Returns a dict of annotations in the MS COCO format
    """


    p = np.array(polypix)
    x = p[:, 0]
    y = p[:, 1]
    area = PolyArea(x, y)

    # BBox Coordinates
    bbox = get_bbox_coordinates(polypix)

    id_sc = obj_id
    #image_id = imgcat_id + "_" + bldg_set_id
    iscrowd = 0
    category_id = category_id

    poly_segmentation = [[x for t in polypix for x in t]]

    ann = {'area': area,
           'bbox': bbox,
           'category_id': category_id,
           'id': id_sc,
           'image_id': bldg_set_id[:-4],
           'iscrowd': 0,
           'segmentation': poly_segmentation}

    return ann


def get_img_info(img_dir, img_files):
    iminfo = []

    for filename in img_files:
        df = pd.read_csv("C:\\Users\\dpawa\\PycharmProjects\\daen690\\xBD\\guatemala-volcano\\images.csv")
        for index, row in df.iterrows():
            if filename == row["image_name"]:
                id = row["id"]
                iminfo.append({
                                  'coco_url': '',
                                  'date_captured': row["capture_date"],
                                  'file_name': filename,
                                  'flickr_url': '',
                                  'height': 1024,
                                  'width': 1024,
                                  'id': filename[:-4],
                                  'license': 3})
    return iminfo





def main():
    # For all objects in all images

    ROOT_DIR = os.getcwd()

    train_path = os.path.join(ROOT_DIR, "images")
    bldg_dir = os.path.join(ROOT_DIR, "polygons")
    #img_path = os.path.join(ROOT_DIR, "image")

    space_coco_ann = {}
    space_ann = []

    src_dir_name = "image"
    img_path = os.path.dirname('/path/images/')

    obj_id = 900000  # Annotation ID

    img_files = [i for ind in os.walk(img_path) for i in ind[2] ]

    #print("Loading COCO style annotations for", len(img_files), "images")
    #print(get_img_info(img_path, img_files))
    for each_img in tqdm(img_files):

        # Load the dataframe

        df = pd.read_csv("/path/polygons.csv")
        for index, row in df.iterrows():
            if each_img == row["image_name"]:
                dfuid = row['image_name']
                if (df.feature_type == "building").any() == True:
                    category_id = 1
                else:
                    category_id = 0
                s = str(row["polygon_x_y"])
                # s = s[12:-4] # Strip out POLYGON and other stuff
                s = re.search("\((.+?)\)", s).group(0)[2:-1]  # Strip out POLYGON and other stuff

                #print(s)

                bldg_pix_co = [[float(i) for i in x.split()] for x in s.split(',')]
                ann = coco_ann(dfuid, category_id, bldg_pix_co, obj_id)
                space_ann.append(ann)

    space_coco_ann['annotations'] = space_ann
    space_coco_ann['info'] = {'description': 'xview2 competition',
                              'url': 'https://www.xview2.org/',
                              'version': '1.0',
                              'year': 2019,
                              'contributor': 'xview',
                              'date_created': '2019/12/22'}
    space_coco_ann['categories'] = [{'supercategory': 'building', 'id': 1, 'name': 'building'}]
    space_coco_ann['images'] = get_img_info(img_path, img_files)

    with open('space_coco_annotations.json', 'w') as fp:
        json.dump(space_coco_ann, fp)




if __name__ == '__main__':
    main()