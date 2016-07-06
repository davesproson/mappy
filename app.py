from __future__ import division
import tornado.ioloop
import tornado.web
import tornado.httputil
import tornado.gen
import tornado.escape

import numpy as np
import matplotlib
matplotlib.use('Agg')
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

from mpl_toolkits.basemap import Basemap 

from PIL import Image, ImageDraw
import shapefile
#import osgeo

import ConfigParser

from mappy.WMS import get_capabilities
from mappy.WMS import WMSGetMapRequest
from mappy.WMS.style import StyleReader
from mappy.Data import DataCollection

TEST_NC_FILE = '/share/data/gwrf/fc_northsea/netcdf/gwrf2016070600/wrf.ns.24km.*'

TEST_VAR = 'mean_sea_level_pressure'

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

def crop_data(data, bbox, wgs84_bbox, proj_to, shapes):
    xdist = bbox[2] - bbox[0]
    ydist = bbox[3] - bbox[1]

    print "X_d = {}, Y_d = {}".format(xdist, ydist)

    # Image width & height
    iwidth = np.shape(data)[1]
    iheight = np.shape(data)[0]
    xratio = iwidth/xdist
    yratio = iheight/ydist
    pixels = []

    print "iwidth = {}".format(iwidth)
    print "iheight = {}".format(iheight)
    print "xratio = {}".format(xratio)
    print "yratio = {}".format(yratio)

    cnt = 0
    img = Image.new("RGB", (iwidth, iheight), "white")
    draw = ImageDraw.Draw(img)

    wgs84 = pyproj.Proj(server.projections['epsg:4326'])

    for shape in shapes:
        slon = [shape.bbox[0], 
                shape.bbox[2]]


        slat = [shape.bbox[1], 
                shape.bbox[3]]

        rlon, rlat = slon, slat
        r_bbox = [rlon[0], rlat[0], rlon[1], rlat[1]]

        if not bboxes_intersect(r_bbox, wgs84_bbox):
            continue


        pixels = []
        for x_p, y_p in shape.points:
            x, y = pyproj.transform(wgs84, proj_to, x_p, y_p)

            px = int(iwidth - ((bbox[2] - x) * xratio))
            py = int((bbox[3] - y) * yratio)

            pixels.append((px,py))

        draw.polygon(pixels, outline="rgb(0,0,0)", fill="rgb(0,0,0)")

    mdata = np.flipud(np.mean(PIL2np(img), axis=2).astype(int))

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


data = DataCollection(file_glob=TEST_NC_FILE,
                      lat_var='latitude', lon_var='longitude',
                      elevation_var='elevation', time_var='time',
                      data_type='netcdf')

#data = DataCollection(file_glob=TEST_NC_FILE,
#                      lat_var='latitude', lon_var='longitude',
#                      elevation_var='depth', time_var='time',
#                      data_type='netcdf')


test_layer = Layer(crop=True, refine_data=0, gshhs_resolution='i',
                   var_name=TEST_VAR, style=styles[1],
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

def reproject(lon, lat, data, native, requested):
    b_lon = lon
    b_lat = lat
    w = data


    b_lon, b_lat = np.meshgrid(b_lon, b_lat)
    p_lon, p_lat = pyproj.transform(native, requested, b_lon, b_lat)
    b_lon = b_lon[0, :]
    b_lat = b_lat[:, 0]
    p_lon = p_lon[0, :]
    p_lat = p_lat[:, 0]


    b_lat = (b_lat - b_lat.min()) / ((b_lat - b_lat.min()).max())
    b_lon = (b_lon - b_lon.min()) / ((b_lon - b_lon.min()).max())


    p_lat_s = (p_lat - p_lat.min()) / ((p_lat - p_lat.min()).max())
    p_lon_s = (p_lon - p_lon.min()) / ((p_lon - p_lon.min()).max())


    i2d = interp2d(b_lon, b_lat, w)
    w = i2d(p_lon_s, p_lat_s)

    return p_lon, p_lat, w

def crop_to_bbox(lon, lat, data, bbox, nx=None, ny=None):

    pass


@tornado.gen.coroutine
def render(layer, width=100, height=100, request=None):



    nc = Dataset(glob.glob(TEST_NC_FILE)[0], 'r')
#    w = np.squeeze(nc[layer.var_name][0, :, :])
    lon = np.squeeze(nc['longitude'][:])
    lat = np.squeeze(nc['latitude'][:])
    nc.close()

    w = layer.data_source.get_data_layer(var_name=TEST_VAR,
                   time=datetime.datetime(2016,7,4)+datetime.timedelta(hours=100))

    
    print "***w = {}".format(np.shape(w))
    lon, lat = np.meshgrid(lon,lat)

    bbox = request.bbox
    wgs84_bbox = request.wgs84_bbox
    crs = request.crs

    proj_to = pyproj.Proj(server.projections[crs.lower()])
    wgs84 = pyproj.Proj(server.projections['epsg:4326'])

    lon = lon[0, :]
    lat = lat[:, 0]
    data_bbox = [lon[0], lat[0], lon[-1], lat[-1]]

#    print data_bbox
    lon,lat = np.meshgrid(lon,lat)

    # Supersample the data, if requested
    if layer.refine_data:
        lon, lat, w = refine_data(lon, lat, w, layer.refine_data)

    if None:
        lon, lat, w = crop_to_bbox(lon, lat, w, wgs84_bbox, nx=50, ny=50)

    # Reproject to requested CRS
    if not crs == 'EPSG:4326':
        p_lon, p_lat = pyproj.transform(wgs84, proj_to, lon, lat)
        minx, miny = pyproj.transform(wgs84, proj_to, lon[0], lat[0])
        maxx, maxy = pyproj.transform(wgs84, proj_to, lon[-1], lat[-1])
    else:
        p_lon, p_lat = lon[0, :], lat[:, 0]
        minx, miny = lon[0], lat[0]
        maxx, maxy = lon[-1], lat[-1]
        p_lon, p_lat = np.meshgrid(p_lon, p_lat)

    if layer.crop or layer.crop_inverse:
        if not layer.shapes:
            layer.set_shapes()
#        w = crop_data(w, data_bbox, layer.shapes)
        w = crop_data(w, bbox, wgs84_bbox, proj_to, layer.shapes) 

    fig = plt.figure(frameon=False)
    fig.set_size_inches(width/100, height/100)
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
