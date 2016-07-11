from __future__ import division
import tornado.ioloop
import tornado.web
import tornado.httputil
import tornado.gen
import tornado.escape

import numpy as np
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from netCDF4 import Dataset

import datetime
import glob
import io
import os
import pyproj
import sys

from scipy import interpolate
from scipy.interpolate import interp2d
from scipy import ndimage as nd


from mpl_toolkits.basemap import Basemap 

from PIL import Image, ImageDraw
import shapefile
#import osgeo

import ConfigParser

from mappy.WMS import get_capabilities
from mappy.WMS import WMSGetMapRequest
from mappy.WMS.style import StyleReader
from mappy.Data import DataCollection

TEST_NC_FILE = '/share/data/gwrf/fc_northsea/netcdf/gwrf2016071100/wrf.ns.24km.*'

TEST_VAR = 'mean_sea_level_pressure'
TEST_VAR = 'water_temperature'

TEST_SHAPEFILE = '/share/data/GEOG/gshhs/GSHHS_shp/i/GSHHS_i_L1.shp'

class Server(object):
    def __init__(self):
        self.contact_person = 'Dave Sproson'
        self.contact_organization = 'A Company'
        self.contact_position = 'A job title'
        self.address = 'Street Address'
        self.city = 'Anyton'
        self.state_or_province = 'Someshire'
        self.postcode = 'AB12 1AB'
        self.country = 'UK'
        self.contact_voice_telephone = '0123456789'
        self.contact_electronic_mail_address = 'My@Email.com'
        self.fees = 'None'
        self.access_constraints = 'Commercial and Restricted'
        self.ip_address = 'fgwfcluster3'
        self.port = '8888'
        self.wms_version = '1.3.0'
        self.projections = dict()

        self.__init_projections()

    def __init_projections(self):
        config = ConfigParser.SafeConfigParser()
        config.read('projections.ini')
        for section in config.sections():
            if section == 'projections':
                for key, value in config.items(section):
                    self.projections.update({key.replace('_',':'): value})

        print self.projections


server = Server()

styles = StyleReader('styles.ini').styles

def PIL2np(img):
        return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def bboxes_intersect(b1, b2):
    def intervals_intersect(x1, x2):
        return x1[1] >= x2[0] and x2[1] >= x1[0]
    return (intervals_intersect((b1[0], b1[2]), (b2[0], b2[2])) and
            intervals_intersect((b1[1], b1[3]), (b2[1], b2[3])))

def mask_data(lon, lat, data, proj, shapes, layer):

#    if layer.mask is not None:
#        print "USING SAVED MASK"
#        return np.ma.masked_where(np.logical_or(layer.mask>0, np.isnan(data)), data)

    # Size of the domain in the requested CRS
    latlon = pyproj.Proj('+proj=latlong +a=6378137 +b=6378137')
    llon, llat = pyproj.transform(proj, latlon, lon, lat)


    lon = lon[0, :]
    lat = lat[:, 0]
    llon = llon[0, :]
    llat = llat[:, 0]

#    print "lon = {}".format(lon)
#    print('lat = {}'.format(lat))
#    print "llon = {}".format(llon)
#    print('llat = {}'.format(llat))

    xdist = lon[-1] - lon[0] #bbox[2] - bbox[0]
    ydist = lat[-1] - lat[0] #bbox[3] - bbox[1]

    bbox = [llon[0], llat[0], llon[-1], llat[-1]]
#    print(bbox)


    # Image width & height
    iwidth = np.shape(data)[1]
    iheight = np.shape(data)[0]
    xratio = iwidth/xdist
    yratio = iheight/ydist
    pixels = []

    img = Image.new("RGB", (iwidth, iheight), "white")
    draw = ImageDraw.Draw(img)

    def nearest(array, val):
        temp = np.abs(array - val)
        return temp.tolist().index(np.min(temp))

    for shape in shapes:
        if not bboxes_intersect(shape.bbox, bbox):
            continue


        pixels = []
        for p_x, p_y in shape.points:
            x, y = pyproj.transform(latlon, proj, p_x, p_y)

#            px = int(iwidth - ((bbox[2] - x) * xratio))
            px = int(iwidth - ((lon[-1] - x) * xratio))
#            px = nearest(lon, x)
#            py = int((bbox[3] - y) * yratio)
            py = int((lat[-1] - y) * yratio)
#            py = nearest(lat, y)

            pixels.append((px,py))

        draw.polygon(pixels, outline="rgb(0,0,0)", fill="rgb(0,0,0)")

    mdata = np.flipud(np.mean(PIL2np(img), axis=2).astype(int))

#    plt.pcolor(mdata)
#    plt.show()

    layer.mask = mdata
    data = np.ma.masked_where(np.logical_or(mdata>0, np.isnan(data)), data)

    return data


class Layer(object):
    def __init__(self, data_source=None, crop=False, crop_inverse=False,
                    crop_file=None, colormap=None, refine_data=0,
                    gshhs_resolution=None, var_name=None, 
                    native_projection=None, style=None, enable_time=False):

        self.data_source = data_source
        self.crop = crop
        self.var_name = var_name
        self.crop_inverse = crop_inverse
        self.crop_file = crop_file
        self.colormap = colormap
        self.refine_data = refine_data
        self.gshhs_resolution = gshhs_resolution
        self.native_projection = pyproj.Proj('+init=EPSG:4326')
        self.shapes = []
        self.style = style
        self.enable_time = enable_time
        self.mask = None

        self.bbox = [-20.358, 39.419, 35.509, 64.749]

        if crop and crop_inverse:
            raise ValueError('crop and crop_inverse cannot both be True')

    def set_shapes(self):
        nc = Dataset(glob.glob(TEST_NC_FILE)[0])
        lat = nc['latitude'][:]
        lon = nc['longitude'][:]
        nc.close()

        bbox = [lon[0], lat[0], lon[-1], lat[-1]]

        r = shapefile.Reader(TEST_SHAPEFILE)
        for shape in r.shapes():
            if not(bboxes_intersect(shape.bbox, bbox)):
                continue
            self.shapes.append(shape)

        print "In-memory caching of shapefiles for layer..."
        print "Size: {}".format(sys.getsizeof(self.shapes))
        print "Shapes: {}".format(len(self.shapes))


data = DataCollection(file_glob=TEST_NC_FILE,
                      lat_var='latitude', lon_var='longitude',
                      elevation_var='elevation', time_var='time',
                      data_type='netcdf')


test_layer = Layer(crop=False, refine_data=0, gshhs_resolution='i',
                   var_name=TEST_VAR, style=styles[2],
                   data_source=data, enable_time=True)

layers = [test_layer]

def refine_data(lon, lat, f, refine):
    lon = lon[0, :]
    lat = lat[:, 0]

    dlon = lon[1] - lon[0]
    dlat = lat[1] - lat[0]

    lat_hi = np.arange(lat[0],lat[-1],dlat/refine)
    lon_hi = np.arange(lon[0],lon[-1],dlon/refine)

    nx = len(lon_hi)
    ny = len(lat_hi)

    a = np.array(f.mask).astype(int)

    f[np.isnan(f)] = 100000

    ipol = interp2d(lon, lat, f)
    apol = interp2d(lon, lat, a)
    f = ipol(lon_hi, lat_hi)
    a = apol(lon_hi, lat_hi)
    f = np.ma.masked_where(a>.2, f)


    lon_hi, lat_hi = np.meshgrid(lon_hi, lat_hi)
    return lon_hi, lat_hi, f


def crop_to_bbox(lon, lat, data, bbox, nx=200, ny=200):
#    lati = np.where(np.logical_and(lat>=bbox[0]-1, lat<=bbox[2]+1))[0]
#    loni = np.where(np.logical_and(lon>=bbox[1]-1, lon<=bbox[3]+1))[0]
#    lon = lon[loni[0]:loni[-1]] 
#    lat = lat[lati[0]:lati[-1]] 
#    data = data[lati[0]:lati[-1], loni[0]:loni[-1]]
#    return lon, lat, data
    def __fill(data):
        invalid = np.isnan(data)

        ind = nd.distance_transform_edt(invalid,
                                        return_distances=False,
                                        return_indices=True)

        return data[tuple(ind)]


    lat_min = bbox[0]
    lat_max = bbox[2]
    lon_min = bbox[1]
    lon_max = bbox[3]

    a = np.array(data.mask).astype(int)
    print a
    a[:, 0] = 1
    a[-1, :] = 1
    print(a)

    data = __fill(data)
#    data[np.isnan(data)] = 9e99



    dlat = (lat_max - lat_min) / ny
    dlon = (lon_max - lon_min) / nx

    lat_hi = np.arange(lat_min, lat_max + dlat, dlat)
    lon_hi = np.arange(lon_min, lon_max + dlon, dlon)

    ipol = interp2d(lon, lat, data)
    apol = interp2d(lon, lat, a)

    data_hi = ipol(lon_hi, lat_hi)
    mask_hi = apol(lon_hi, lat_hi)

    data_hi = np.ma.masked_where(mask_hi>.2, data_hi)

    return lon_hi, lat_hi, data_hi
    





@tornado.gen.coroutine
def render(layer, width=100, height=100, request=None):


    # Get the lat/lon variables for the dataset
    nc = Dataset(glob.glob(TEST_NC_FILE)[0], 'r')
    lon = np.squeeze(nc['longitude'][:])
    lat = np.squeeze(nc['latitude'][:])
    nc.close()

    # Get the data from the layer's data_source
    w = layer.data_source.get_data_layer(var_name=TEST_VAR,
                   time=datetime.datetime(2016,7,11)+datetime.timedelta(hours=100))

    
    print "*** shape(data) = {}".format(np.shape(w))
#    lon, lat = np.meshgrid(lon,lat)

    # Save out some useful stuff from the request
    bbox = request.bbox                 # BBOX in the requested CRS
    wgs84_bbox = request.wgs84_bbox     # BBOX in WGS84 CRS
    crs = request.crs                   # The requested CRS

    # Initialise pyproj projections for the requested and WGS84 CRS's
    proj_to = pyproj.Proj(server.projections[crs.lower()])
    wgs84 = pyproj.Proj(server.projections['epsg:4326'])
    latlon = pyproj.Proj('+proj=latlong +a=6378137 +b=6378137')

    latlon_bbox = [0, 0, 0, 0]
    latlon_bbox[0], latlon_bbox[2] = pyproj.transform(wgs84, latlon, wgs84_bbox[0], wgs84_bbox[2])
    latlon_bbox[1], latlon_bbox[3] = pyproj.transform(wgs84, latlon, wgs84_bbox[1], wgs84_bbox[3])

    # Build a bounding box for the data (assume this is in WGS84)
#    lon = lon[0, :]
#    lat = lat[:, 0]
    data_bbox = [lon[0], lat[0], lon[-1], lat[-1]]

#    lon,lat = np.meshgrid(lon,lat)

    # Supersample the data, if requested
#    if layer.refine_data:
#        lon, lat, w = refine_data(lon, lat, w, layer.refine_data)

#    if True:
#        lon, lat, w = crop_to_bbox(lon, lat, w, wgs84_bbox, nx=50, ny=50)

    if True: #layer.crop or layer.crop_inverse:
        if not layer.shapes:
            layer.set_shapes()

#        w = mask_data(lon[0, :], lat[:, 0], w, layer.shapes)


    # Reproject to requested CRS
    if crs != 'EPSG:4326':
        lon, lat = np.meshgrid(lon, lat)
        p_lon, p_lat = pyproj.transform(wgs84, proj_to, lon, lat)
        minx, miny = pyproj.transform(wgs84, proj_to, lon[0], lat[0])
        maxx, maxy = pyproj.transform(wgs84, proj_to, lon[-1], lat[-1])
    else:
#        p_lon, p_lat = lon[0, :], lat[:, 0]
        minx, miny = lon[0], lat[0]
        maxx, maxy = lon[-1], lat[-1]
        p_lon, p_lat = np.meshgrid(lon, lat)

#    if layer.crop:
#        w = mask_data(p_lon[0, :], p_lat[:, 0], w, layer.shapes, wgs84, proj_to, layer)     

    if True:
        lon, lat, w = crop_to_bbox(p_lon[0, :], p_lat[:, 0], w, bbox, nx=400, ny=200)
        p_lon, p_lat = np.meshgrid(lon, lat)

    if True: #layer.crop:
        w = mask_data(p_lon, p_lat, w, proj_to, layer.shapes, layer)     


    fig = plt.figure(frameon=False)
#    fig.set_size_inches(width/100, height/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    print "***w = {}".format(np.shape(w))
    function_args = {}
    for key, value in layer.style.render_args.iteritems():
        if 'fn:' in value:
            fn = ''.join(value.split(':')[1:])
            value = eval(fn)
        function_args.update({key: value})

    print function_args


#    lon, lat = np.meshgrid(lon, lat)
    getattr(plt, layer.style.render_function)(
       p_lon, p_lat, w, **function_args)

#    plt.pcolormesh(p_lon, p_lat, w)


    if '+proj=longlat' in server.projections[crs.lower()]:
        ax.set_xlim([bbox[0], bbox[2]])
        ax.set_ylim([bbox[1], bbox[3]])
    else:
        ax.set_xlim([bbox[1], bbox[3]])
        ax.set_ylim([bbox[0], bbox[2]])
    
#    print "   ...done"

    memdata = io.BytesIO()
    plt.savefig(memdata, format='png', dpi=100, transparent=True)
    plt.close()
    image = memdata.getvalue()
    memdata.close()

    print "request complete"
    

    raise tornado.gen.Return(image)


def nan_helper(y):
	return np.isnan(y), lambda z: z.nonzero()[0]

class MainHandler(tornado.web.RequestHandler):

    @tornado.gen.coroutine
    def get(self):

        s = datetime.datetime.utcnow()
        
        print self.request.query_arguments

        try:
            request = self.get_argument('REQUEST')
        except:
            request = self.get_argument('request')

        if request == 'GetCapabilities':
            capes = get_capabilities(server, layers)
            self.set_header('Content-type', 'text/xml')
            self.write(capes)
            return

        if request == 'GetMap':
            map_request = WMSGetMapRequest(**self.request.query_arguments)

            image = yield render(test_layer, 
                               width=map_request.width,
                               height=map_request.height, 
                               request=map_request)

            self.set_header('Content-type', 'image/png')
            self.write(image)
            print (datetime.datetime.utcnow() - s).total_seconds()


def make_app():
    return tornado.web.Application([
        (r'/wms', MainHandler),
        (r'', MainHandler),
    ])


if __name__ == '__main__':
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
