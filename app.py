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

import io
import os
import pyproj
import sys

from scipy import interpolate
from scipy.interpolate import interp2d

from mpl_toolkits.basemap import Basemap 

from PIL import Image, ImageDraw
import shapefile

class Server(object):
    def __init__(self):
        self.contact_person = 'Dave Sproson'
        self.contact_organization = 'Fugro GEOS'
        self.contact_position = 'Principal MetOcean Modeller'
        self.address = 'Fugro House, Hitercroft Road'
        self.city = 'Wallingford'
        self.state_or_province = 'Oxfordshire'
        self.postcode = 'OX10 9RB'
        self.country = 'UK'
        self.contact_voice_telephone = '07920201045'
        self.contact_electronic_mail_address = 'D.Sproson@fugro.com'
        self.fees = 'None'
        self.access_constraints = 'Commercial and Restricted'

def PIL2np(img):
        return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def bboxes_intersect(b1, b2):
    def intervals_intersect(x1, x2):
        return x1[1] >= x2[0] and x2[1] >= x1[0]
    return (intervals_intersect((b1[0], b1[2]), (b2[0], b2[2])) and
            intervals_intersect((b1[1], b1[3]), (b2[1], b2[3])))

def crop_data(data, bbox, shapes):
    xdist = bbox[2] - bbox[0]
    ydist = bbox[3] - bbox[1]

    # Image width & height
    iwidth = np.shape(data)[1]
    iheight = np.shape(data)[0]
    xratio = iwidth/xdist
    yratio = iheight/ydist
    pixels = []

    cnt = 0
    img = Image.new("RGB", (iwidth, iheight), "white")
    draw = ImageDraw.Draw(img)

    for shape in shapes:
        if not bboxes_intersect(shape.bbox, bbox):
            continue
        pixels = []
        for x,y in shape.points:

            px = int(iwidth - ((bbox[2] - x) * xratio))
            py = int((bbox[3] - y) * yratio)
            pixels.append((px,py))

        draw.polygon(pixels, outline="rgb(0,0,0)", fill="rgb(0,0,0)")

    mdata = np.mean(PIL2np(img), axis=2)

    print "mdata size: {}".format(np.shape(mdata))
    print mdata

    data[np.flipud(mdata)<10] = np.nan
    return data

def test_plot_1(ax, data, layer, bbox):

    if layer.crop or layer.crop_inverse:
        if not layer.shapes:
            layer.set_shapes()

        data = crop_data(data, bbox, layer.shapes)

    # ax.imshow(data, origin='lower', aspect='auto',interpolation='nearest') 
    ax.contourf(data/100, levels=np.arange(920, 1030, 4))

class Layer(object):
    def __init__(self, data_file_glob=None, crop=False, crop_inverse=False,
                    crop_file=None, colormap=None, refine_data=4,
                    gshhs_resolution=None, var_name=None, 
                    native_projection=None):

        self.data_file_glob = data_file_glob
        self.crop = crop
        self.var_name = var_name
        self.crop_inverse = crop_inverse
        self.crop_file = crop_file
        self.colormap = colormap
        self.refine_data = refine_data
        self.gshhs_resolution = gshhs_resolution
        self.render_fn = test_plot_1
        self.native_projection = pyproj.Proj('+init=EPSG:4326')
        self.shapes = []

        if crop and crop_inverse:
            raise ValueError('crop and crop_inverse cannot both be True')

    def set_shapes(self):
        nc = Dataset(self.data_file_glob)
        lat = nc['latitude'][:]
        lon = nc['longitude'][:]
        nc.close()

        bbox = [lon[0], lat[0], lon[-1], lat[-1]]

        r = shapefile.Reader('C:\Users\Dave\gshhg\GSHHS_shp\l\GSHHS_l_L1')
        for shape in r.shapes():
            if not(bboxes_intersect(shape.bbox, bbox)):
                continue
            self.shapes.append(shape)

        print "In-memory caching of shapefiles for layer..."
        print "Size: {}".format(sys.getsizeof(self.shapes))



test_layer = Layer(data_file_glob='c:\Users\dave\python\wrf.ns.24km.2016050700.100.nc', 
                   crop=True, refine_data=8, gshhs_resolution='i',
                   var_name='mean_sea_level_pressure')

layers = [test_layer]

def refine_data(lon, lat, f, refine):
    dlon = lon[1] - lon[0]
    dlat = lat[1] - lat[0]

    lat_hi = np.arange(lat[0],lat[-1],dlat/refine)
    lon_hi = np.arange(lon[0],lon[-1],dlon/refine)

    nx = len(lon_hi)
    ny = len(lat_hi)

    print 'meshing'
    lon, lat = np.meshgrid(lon, lat)
    print 'done'

    print 'interpolate'
    ipol = interp2d(lon[0, :], lat[:, 0], f)
    print 'built'
    f = ipol(lon_hi, lat_hi)
    print 'done'

    return f

def reproject(lon, lat, data, native, requested):
    b_lon = lon
    b_lat = lat
    w = data

    print 'b_lon = {}..{}'.format(b_lon.min(), b_lon.max())
    print 'b_lat = {}..{}'.format(b_lat.min(), b_lat.max())
    print '-'*50

    b_lon, b_lat = np.meshgrid(b_lon, b_lat)
    p_lon, p_lat = pyproj.transform(native, requested, b_lon, b_lat)
    b_lon = b_lon[0, :]
    b_lat = b_lat[:, 0]
    p_lon = p_lon[0, :]
    p_lat = p_lat[:, 0]

    print 'p_lon = {}..{}'.format(p_lon.min(), p_lon.max())
    print 'p_lat = {}..{}'.format(p_lat.min(), p_lat.max())
    print '-'*50

    b_lat = (b_lat - b_lat.min()) / ((b_lat - b_lat.min()).max())
    b_lon = (b_lon - b_lon.min()) / ((b_lon - b_lon.min()).max())

    print 'scaled b_lon = {}..{}'.format(b_lon.min(), b_lon.max())
    print 'scaled b_lat = {}..{}'.format(b_lat.min(), b_lat.max())
    print '-'*50

    p_lat = (p_lat - p_lat.min()) / ((p_lat - p_lat.min()).max())
    p_lon = (p_lon - p_lon.min()) / ((p_lon - p_lon.min()).max())

    print 'scaled p_lon = {}..{}'.format(p_lon.min(), p_lon.max())
    print 'scaled p_lat = {}..{}'.format(p_lat.min(), p_lat.max())
    print '-'*50

    #b_lon, b_lat = np.meshgrid(b_lon, b_lat)
    i2d = interp2d(b_lon, b_lat, w)
    w = i2d(p_lon, p_lat)

    return w

def render(layer, width=100, height=100, bbox=None, crs='EPSG:4326'):

    proj_to = pyproj.Proj('+init={}'.format(crs))

    print "Reading data..."
    nc = Dataset(layer.data_file_glob, 'r')
    w = np.squeeze(nc[layer.var_name][0, :, :])
    lon = np.squeeze(nc['longitude'][:])
    lat = np.squeeze(nc['latitude'][:])
    nc.close()
    print "   ...done"

    lon, lat = np.meshgrid(lon,lat)
    wgs84 = pyproj.Proj('+init=EPSG:4326')
    # p_lon, p_lat = pyproj.transform(wgs84, proj_to, lon, lat)



    #nans, x = nan_helper(w)
    #w[nans] = np.interp(x(nans), x(~nans), w[~nans])
    mask = np.zeros_like(w)
    mask[~np.isnan(w)] = 1
    w[np.isnan(w)] = np.nanmean(w)

    lon = lon[0, :]
    lat = lat[:, 0]

    b_lon_min = bbox[0]
    b_lat_min = bbox[1]
    b_lon_max = bbox[2]
    b_lat_max = bbox[3]

    nx = len(lon)
    ny = len(lat)

    #Interpolate onto bounding box
    b_lon = np.linspace(b_lon_min, b_lon_max, num=nx)
    b_lat = np.linspace(b_lat_min, b_lat_max, num=ny)
    i2d = interp2d(lon, lat, w)
    w = i2d(b_lon, b_lat)

    w = reproject(b_lon, b_lat, w, wgs84, proj_to)
    
    fig = plt.figure(frameon=False)
    fig.set_size_inches(width/100, height/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    if layer.refine_data:
         w = refine_data(b_lon, b_lat, w, layer.refine_data)

    print "Rendering image..."
    layer.render_fn(ax=ax, data=w, bbox=bbox, layer=layer)
    print "   ...done"

    memdata = io.BytesIO()
    plt.savefig(memdata, format='png', dpi=100, transparent=True)
    image = memdata.getvalue()

    print "request complete"
    return image



def nan_helper(y):
	return np.isnan(y), lambda z: z.nonzero()[0]

def get_capabilities():
    with open('capabilities.xml', 'r') as f:
	   return f.read()


class MainHandler(tornado.web.RequestHandler):
    def get(self):


        print self.request.uri

        request = self.get_argument('REQUEST')
        if request == 'GetCapabilities':
            capes = get_capabilities()
            self.set_header('Content-type', 'text/xml')
            self.write(capes)
            return

        if request == 'GetMap':        
            bbox = [float(i) for i in self.get_argument('BBOX').split(',')]
            print 'bbox = {}'.format(bbox)
            width = int(self.get_argument('WIDTH', default=100))
            height = int(self.get_argument('HEIGHT', default=100))
            crs = tornado.escape.url_unescape(self.get_argument('CRS', default='EPSG:4326'))

            image = render(test_layer, bbox=bbox, width=width, height=height, crs=crs)

            self.set_header('Content-type', 'image/png')
            self.write(image)

            
            




def make_app():
    return tornado.web.Application([
        (r'/', MainHandler),
        (r'', MainHandler),
    ])


if __name__ == '__main__':
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
