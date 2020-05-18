import cv2
from collections import defaultdict
from shapely.geometry import box, Point, Polygon, MultiPolygon, GeometryCollection
import shapely
from shapely import geometry
import shapely.wkt 
from shapely.wkt import dumps
import geopandas as gpd
import json
import numpy as np
import warnings
from PIL import Image, ImageDraw
warnings.simplefilter(action='ignore', category=FutureWarning)



def mask_to_polygons(mask, epsilon=10., min_area=10.):
    """Convert a mask ndarray (binarized image) to Multipolygons"""
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(mask,
                                  cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_NONE)
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    all_polygons = MultiPolygon(all_polygons)

    return all_polygons


def katana(geometry, threshold, count=0):
    """Split a Polygon into two parts across it's shortest dimension"""
    bounds = geometry.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    if max(width, height) <= threshold or count == 250:
        # either the polygon is smaller than the threshold, or the maximum
        # number of recursions has been reached
        return [geometry]
    if height >= width:
        # split left to right
        a = box(bounds[0], bounds[1], bounds[2], bounds[1]+height/2)
        b = box(bounds[0], bounds[1]+height/2, bounds[2], bounds[3])
    else:
        # split top to bottom
        a = box(bounds[0], bounds[1], bounds[0]+width/2, bounds[3])
        b = box(bounds[0]+width/2, bounds[1], bounds[2], bounds[3])
    result = []
    for d in (a, b,):
        c = geometry.intersection(d)
        if not isinstance(c, GeometryCollection):
            c = [c]
        for e in c:
            if isinstance(e, (Polygon, MultiPolygon)):
                result.extend(katana(e, threshold, count+1))
    if count > 0:
        return result
    # convert multipart into singlepart
    final_result = []
    for g in result:
        if isinstance(g, MultiPolygon):
            final_result.extend(g)
        else:
            final_result.append(g)
    return shapely.geometry.MultiPolygon(final_result)



def getxymgrs(inputs):
    order = inputs[0]
    g = inputs[1]
    NWLon = inputs[2]
    NWLat = inputs[3]
    longpixelrate = inputs[4]
    latpixelrate = inputs[5]
    im = inputs[6]

    temp = shapely.affinity.translate(g, xoff=-NWLon, yoff=-NWLat)
    temp = shapely.affinity.scale(temp, xfact=1/longpixelrate, yfact=-1/latpixelrate, origin = (0,0)).wkt
    temp = dumps(shapely.wkt.loads(temp),rounding_precision=1)
    tempjson = json.loads(gpd.GeoSeries([shapely.wkt.loads(temp)]).to_json())
    temp = np.array(tempjson['features'][0]['geometry']['coordinates']).squeeze().astype(int)
    filteredtemp = temp.copy()
    filteredtemp[[temp<0]] = 0
    filteredtemp[[temp>im.shape[0]]] = im.shape[0]-1
    return (order, filteredtemp)


def getdamagecost(inputs):
    order = inputs[0]
    g = inputs[1]
    im = inputs[2]
    sqftcost = inputs[3]
    pixarea = inputs[4]   
    pic = inputs[5]
    idx = list(map(tuple, g))
    mask = Image.new('L', im.shape, False)
    ImageDraw.Draw(mask).polygon(idx, outline=1, fill=True)
    mask = np.array(mask).astype(bool)
    pixels = mask.sum()
    out = pic[mask].sum() * sqftcost * pixarea # * pix / pix
    return (order, out)