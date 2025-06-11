
##### IMPORT PACKAGES ####
import shapely
import os
import pickle
import matplotlib.pyplot as plt

# Import Mask RCNN
ROOT_DIR = os.path.abspath('./CoreProcessingPipelineScripts/CNN/')
print('ROOT_DIR', ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library
from functions.src_get_centerline import get_centerline

#### LOAD POLYGON ####
polygon_file = 'CoreProcessingPipelineScripts/CNN/testing_and_development/5px_test_shapely_polygons/shapely_polygon0.pkl'
with open(polygon_file, 'rb') as file:
    polygon = pickle.load(file)

xp,yp = polygon.boundary.coords.xy # check the number of points

#cline = get_centerline(polygon, segmentize_maxlen=10, max_points=300, simplification=0.05)
cline = get_centerline(polygon, segmentize_maxlen=0.5, max_points=600, simplification=0.1, segmentize_maxlen_post=11, smooth_sigma=5)

xl, yl = cline.coords.xy
# plot polygon and the line
plt.plot(xp,yp)
plt.plot(xl, yl)
plt.show()

###### TEST SIMPLIFICATION ####
segmentize_maxlen=200
max_points = 300
simplification= 0.05
outline=polygon.exterior
outline_s=outline.segmentize(segmentize_maxlen)
outline_points = outline_s.coords
simplification_updated = simplification
        while len(outline_points) > max_points:
            #print('outline_points_while_A:', len(outline_points))
            # if geometry is too large, apply simplification until geometry
            # is simplified enough (indicated by the "max_points" value)
            simplification_updated += simplification
            outline_points = outline_s.simplify(simplification_updated).coords
            #print('outline_points_while_B:',  len(outline_points))


#polygon_simple = polygon.simplify(20)
xps,yps = outline_points.xy
plt.plot(xp,yp)
plt.plot(xps, yps)
plt.show()

#### test in original file ####
geom = polygon