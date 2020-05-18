from multiprocessing import Pool
#   our functions
from mgrsplotter import getlatlons, buildingcostfunc
from polygonhelper import mask_to_polygons, katana, getxymgrs, getdamagecost
#   shapely
from PIL import Image, ImageDraw
import shapely
import shapely.wkt
import numpy as np
import mgrs
import pandas as pd
import geopandas as gpd
from multiprocessing import Pool
from shapely.geometry import box, Point, Polygon, MultiPolygon, GeometryCollection
import shapely.ops as ops
from functools import partial
import pyproj 
import warnings
import string
import utm
import cv2
from collections import defaultdict
from shapely.wkt import dumps
import json
import psutil
#    PLOTTING
from bokeh.plotting import figure, save
from bokeh.models import GeoJSONDataSource, LogColorMapper, HoverTool, WheelZoomTool, Range1d, CategoricalColorMapper, LinearColorMapper
from bokeh.palettes import Viridis6 as palette
import matplotlib.pyplot as plt
from bokeh.io import output_file
from bokeh.embed import components
import time

#get prepped
m = mgrs.MGRS()
warnings.simplefilter(action='ignore', category=FutureWarning)
cores = 1
if psutil.cpu_count(logical = False) > 2: #can be changed if needed  psutil.cpu_count(logical = True) #true = threads, false = cores
    cores = 3

def costplotter(inputs):
    print(cores)
    print(psutil.cpu_count(logical = True))
    print(psutil.cpu_count(logical = False))
    start = time.time()
    distroyedfile = inputs[0]
    predfile = inputs[1]
    SELat = float(inputs[2])
    SELon = float(inputs[3])
    NWLat = float(inputs[4])
    NWLon = float(inputs[5])
    im = inputs[6]
#   derive latlon deltas and image polygon
    latpixelrate =  abs(abs(NWLat) - abs(SELat) ) /im.shape[1]
    longpixelrate =  abs(abs(NWLon) - abs(SELon) ) /im.shape[0]

    latlons = [Point(NWLat,NWLon),
                Point(NWLat,SELon),
                Point(SELat,SELon),
                Point(SELat,NWLon)]
    poly = Polygon([(p.y, p.x) for p in latlons]) #lon, lat

    #   get image area + pixel area
    geom_area = ops.transform(
    partial(
        pyproj.transform,
        pyproj.Proj(init='EPSG:4326'), #https://gis.stackexchange.com/questions/127607/area-in-km-from-polygon-of-coordinates
        pyproj.Proj(
            proj='aea',
            lat_1=poly.bounds[1],
            lat_2=poly.bounds[3])), poly)
    lengthpixmeter =  np.sqrt(geom_area.area)/im.shape[0]
    pixrounding = int(10/lengthpixmeter)-3
    lengthfeet= np.sqrt(geom_area.area) * 3.28084 #width of picture in feet
    pixarea = np.square(lengthfeet/im.shape[0]) # area per pixel

    #   read file using CV2 for polygon division speed
    img = cv2.imread(predfile, cv2.IMREAD_UNCHANGED) #change
    Bw = cv2.convertScaleAbs(img)
    polygons = mask_to_polygons(Bw, min_area=pixrounding)

    #katana
    print('startforreal')
    print(time.time()-start)
    polygons = katana(polygons,pixrounding, count = 0)
    print(pixrounding)
    print('katana')
    print(time.time()-start)

    #   translate polygons into lat/lon coordinates
    polygons = shapely.affinity.scale(polygons, xfact=longpixelrate, yfact=-latpixelrate, origin = (0,0))
    polygons = shapely.affinity.translate(polygons, xoff=NWLon, yoff=NWLat)
    print('scale')
    print(time.time()-start)

    #   find centroids of divided polygons
    centroids = []
    for poly in polygons:
        centroids.append((poly.centroid.x,poly.centroid.y))
    print('centroid')
    print(time.time()-start)

    #   get MGRS list for centroid points
    mgrslist = []
    for x,y in centroids:
        s = m.toMGRS(y, x).decode()
        main = s[:5]
        coords = s[5:]
        hundredk = main
        tenk = main + coords[0] + coords[5]
        onek = main + coords[0:2]+ coords[5:7]
        hundredmeter = main + coords[0:3]+ coords[5:8]
        tenmeter = main + coords[0:4]+ coords[5:9]
        mgrslist.extend([hundredk,tenk,onek,hundredmeter,tenmeter,s])
    mgrslisted = set(mgrslist) #remove duplicates
    mgrslist = [item for item in mgrslisted if len(item) <= 13]
    #remove 1m

    print('mgrsdedup')
    print(time.time()-start)


    #   get cost per sqft for zipcode
    imagelonlat = m.toMGRS(polygons.centroid.y, polygons.centroid.x).decode()
    main = imagelonlat[:5]
    coords = imagelonlat[5:]
    onek = main + coords[0:2]+ coords[5:7] 
    try:
        df=pd.read_csv('../Exploration/artifacts/polygons_meta.csv')
        sqftcost = df[df['MGRS1k']==onek]['cost_sqft'].values[0]
    except:
        sqftcost = 250 #default value for now
    sqftcost = sqftcost * 2 
    print('xzpcode')
    print(time.time()-start)

    #   build dataframe of grid, length, points, and eventually cost
    conversiondict = {'5':'100k',
                    '7':'10k',
                    '9':'1k',
                    '11':'100m',
                    '13': '10m',
                    '15': '1m'}
    df = pd.DataFrame(mgrslist, columns=['grid'])
    df['length'] = df['grid'].apply( lambda x : conversiondict[str(len(str(x)))])#prep for bokeh
    df['geom'] = df['grid'].apply( lambda x : shapely.wkt.loads(getlatlons(x)[1].wkt))#polygon points
    print('polygons of mgrs')
    print(time.time()-start)

    #   back convert mgrs polygons to image coords and sum up pixel data within polygons
    xymgrs = []
    inputs = [NWLon,NWLat,longpixelrate,latpixelrate, im]
    into = []
    for i, g in enumerate(df.geom.values):
            val = [i,g] + inputs
            into.append(val)
    with Pool(processes=cores) as pool: # or whatever your hardware can support
            outputs = pool.map(getxymgrs, into)
            pool.terminate()
            pool.join()

    sorter = sorted(outputs, key=lambda x: x[0])
    df['geomxy'] = [x[1] for x in sorter]
    print('backconvert')
    print(time.time()-start)
    #   sum within polygons using cost data

    pic = im.copy()#alter the image pixel values for financial anaylsis
    #dam*area*sqft + dam*area*sqft ===(dam+dam+dam)*sqft*area
    #0,  63, 126, 189, 252 values 
    pic[[im<64]] = 0
    pic[[im==126]] = 0.5
    pic[[im==189]] = 0.8
    pic[[im==252]] = 1
    print('relabel image points')
    print(time.time()-start)

    #draw polygon and sum pixels inside
    inputs = [im, sqftcost, pixarea, pic]
    into = []
    for i, g in enumerate(df.geomxy.values):
            val = [i,g] + inputs
            into.append(val)
    with Pool(processes=cores) as pool: # or whatever your hardware can support
            outputs = pool.map(getdamagecost, into)
            pool.terminate()
            pool.join()

    sorter = sorted(outputs, key=lambda x: x[0])
    totaldamagecost = [x[1] for x in sorter]
    print('get total cost')
    print(time.time()-start)

    df['cost'] = totaldamagecost
    df = df.drop(columns=['geomxy'])
    print('final df')
    print(time.time()-start)

    #   make plot
    gridsgeo = gpd.GeoDataFrame(df, geometry='geom')

    categories = [str(i) for i in gridsgeo['length'].unique()] #get string of MGRS precisions
    categories = ['100k','10k','1k','100m', '10m']
    scale ={}
    for category in categories: #define for plot
        gridsgeoofcat = gridsgeo['cost'][gridsgeo['length']== category]
        scale[category] = (min(gridsgeoofcat), max(gridsgeoofcat))
    gridsgeo['costcolor'] = gridsgeo.apply(lambda row: buildingcostfunc(row,scale), axis=1) #set color scale bins

    #geojson
    gdf = gridsgeo
    gdf_json = gdf.to_json()
    gjson = json.loads(gdf_json)

    TOOLS = "pan,wheel_zoom,box_zoom,hover, reset,save"

    p = figure(match_aspect = True, min_border = 0,
            tools=TOOLS,
                title="Building Damage Cost Results", height=1024, width=1024,
                x_range=( NWLon,SELon), y_range=(SELat, NWLat)) 

    width = longpixelrate * im.shape[0]
    height = latpixelrate * im.shape[1]
    p.image_url(url=[distroyedfile], x=NWLon, y=NWLat, w=width, h=height, angle=0) #x and y wrt NW corner width and height in lon/lat scale based on GSD


    #Geojson mapping
    source_shapes = {}
    for category in categories:
        source_shapes[category] = {"type": "FeatureCollection", "features": []}
    for item in gjson['features']:
        source_shapes[item['properties']['length']]['features'].append(item)

    mypal = ["white", "yellow", 'red']
    cmap = CategoricalColorMapper(palette = ["orange", "purple", "pink", "brown", "blue","teal"], 
                                factors = categories)

    #costcoloring
    types = ['one','two','three','four','five']
    othercmap = CategoricalColorMapper(palette = ["white", "yellow", "orange", "red", "green"], factors = types)
    for t in types:
        source_shapes[t] = {"type": "FeatureCollection", "features": []}
    for item in gjson['features']:
        source_shapes[item['properties']['costcolor']]['features'].append(item)
    levels = {}
    for category in categories:
        source_shape = GeoJSONDataSource(geojson = json.dumps(source_shapes[category]))
        levels[category] = p.patches('xs', 'ys', fill_color = {'field': 'costcolor', 'transform': othercmap},
                            line_color = 'black', line_width = 0.5, alpha=.7,
                            legend_label = category, source = source_shape,)
    levels['100k'].visible = False
    levels['10k'].visible = False
    levels['1k'].visible = False
    levels['100m'].visible = False

    p.background_fill_color="#f5f5f5"
    p.grid.grid_line_color="white"
    p.xaxis.axis_label = 'Longitude'
    p.yaxis.axis_label = 'Latitude'
    p.axis.axis_line_color = None
    hover = p.select_one(HoverTool)
    hover.point_policy = "follow_mouse" 
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    hover.tooltips = [
            ("USNG", "@grid"),
            # ("(Long, Lat)", "($x, $y)"),
            ("Cost", "@cost{($ 0.00 a)}"),
            ]
    p.legend.click_policy = 'hide'
    # output_file("index.html")
    # save(p)
    plot_script, plot_div = components(p)
    print('plot')
    print(time.time()-start)
    return [True, plot_script, plot_div]