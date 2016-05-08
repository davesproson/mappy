from __future__ import division
import tornado.ioloop
import tornado.web
import tornado.httputil

import numpy as np
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from netCDF4 import Dataset

import io
import os
import pyproj

from scipy import interpolate
from scipy.interpolate import interp2d

from mpl_toolkits.basemap import maskoceans

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

class Layer(object):
    def __init__(self, data_file_glob=None, crop=False, crop_inverse=False,
                    crop_file=None, colormap=None, refine_data=4,
                    gshhs_resolution=None, var_name=None):

        self.data_file_glob = data_file_glob
        self.crop = crop
        self.var_name = var_name
        self.crop_inverse = crop_inverse
        self.crop_file = crop_file
        self.colormap = colormap
        self.refine_data = refine_data
        self.gshhs_resolution = gshhs_resolution
        self.plotting_function = test_plot_1

        if crop and crop_inverse:
            raise ValueError('crop and crop_inverse cannot both be True')


test_layer = Layer(data_file_glob='wrf.ns.24km.2016050700.100.nc', 
                   crop=True, refine_data=8, gshhs_resolution='i',
                   var_name='mean_sea_level_pressure')

layers = [test_layer]



def render(x, y, f, width=100, height=100, lon=None, lat=None):
    """Return a Matplotlib `fig` and `ax` handles for a Google-Earth Image."""

    print "Rendering image"
    bbox = [lon.min(), lat.min(), lon.max(), lat.max()]

    refine=8
    if(refine):
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
        lon, lat = np.meshgrid(lon_hi, lat_hi)
    else:
        nx = len(lon)
        ny = len(lat)
        lon, lat = np.meshgrid(lon, lat)

    print "loading shapefile"
    # r = shapefile.Reader("ne_10m_coastline")
    r = shapefile.Reader(os.path.join('gshhg', 'GSHHS_shp', 'i', 'GSHHS_i_L1'))

    def PIL2np(img):
        return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)


    #================= START RASTER CLIP TEST ===========================
    if True: # Masking
        xdist = bbox[2] - bbox[0]
        ydist = bbox[3] - bbox[1]

        print xdist, ydist
        # Image width & height
        iwidth = nx
        iheight = ny
        xratio = iwidth/xdist
        yratio = iheight/ydist
        pixels = []

        cnt = 0
        img = Image.new("RGB", (iwidth, iheight), "white")
        draw = ImageDraw.Draw(img)

        print dir(r)
        print r.fields

        for shape in r.shapes():
        # for i in range(1):
            # shape = r.shapes()[40]
            # print dir(shape)
            pixels = []
            for x,y in shape.points:

                px = int(iwidth - ((bbox[2] - x) * xratio))
                py = int((bbox[3] - y) * yratio)
                pixels.append((px,py))
            # pixels.append(pixels[0])

            draw.polygon(pixels, outline="rgb(0,0,0)", fill="rgb(0,0,0)")

        mdata = np.mean(PIL2np(img), axis=2)

        print "mdata size: {}".format(np.shape(mdata))
        print mdata

        f[np.flipud(mdata)>10] = np.nan

    #--------------- END RASTER CLIP TEST -----------------------------

        
    #lon_hi, lat_hi = np.meshgrid(lon_hi, lat_hi)
    # print 'masking'
    #a = maskoceans(lon, lat, f, inlands=True, resolution='i', grid=1.25)
    # print 'inverting mask'
    # f = np.ma.masked_array(f, np.logical_not(a.mask))


    fig = plt.figure(frameon=False)
    fig.set_size_inches(width/100, height/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)  

    print 'rendering image'
    print np.shape(f)

           
    ax.imshow(f,  interpolation='nearest', origin='lower', aspect='auto')
    #ax.contour(f/100,levels=np.arange(920,1030,4), colors='k', linewidths=3)

    





    memdata = io.BytesIO()

   
    plt.savefig(memdata, format='png', dpi=100, transparent=True)
    image = memdata.getvalue()
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

        
        bbox = [float(i) for i in self.get_argument('BBOX').split(',')]
        print 'bbox = {}'.format(bbox)
        width = int(self.get_argument('WIDTH', default=100))
        height = int(self.get_argument('HEIGHT', default=100))
        crs = self.get_argument('CRS', default='EPSG:4326') 

        wgs84 = pyproj.Proj('+init=EPSG:4326')

        print crs
        proj_to = pyproj.Proj('+init={}'.format(crs))

        f = 'wrf.ns.24km.2016050700.100.nc'
        nc = Dataset(f)
        w = np.squeeze(nc['mean_sea_level_pressure'][0, :, :])
        lon = np.squeeze(nc['longitude'][:])
        lat = np.squeeze(nc['latitude'][:])
        nc.close()



        lon, lat = np.meshgrid(lon,lat)
        p_lon, p_lat = pyproj.transform(wgs84, proj_to, lon, lat)



        #nans, x = nan_helper(w)
        #w[nans] = np.interp(x(nans), x(~nans), w[~nans])
        mask = np.zeros_like(w)
        mask[~np.isnan(w)] = 1
        w[np.isnan(w)] = np.nanmean(w)

        lon = lon[0, :]
        lat = lat[:, 0]

        print 'lon = {}..{}'.format(lon.min(), lon.max())
        print 'lat = {}..{}'.format(lat.min(), lat.max())

        p_lon = p_lon[0, :]
        p_lat = p_lat[:, 0]

        #w = np.ma.masked_where(np.isnan(w), w)


        lat = (lat - lat.min()) / ((lat - lat.min()).max())
        lon = (lon - lon.min()) / ((lon - lon.min()).max())

        b_lon_min = bbox[0]
        b_lat_min = bbox[1]
        b_lon_max = bbox[2]
        b_lat_max = bbox[3]

        nx = len(lon)
        ny = len(lat)
        
        print "lat range = {} - {}".format(b_lat_min, b_lat_max)
        print "lon range = {} - {}".format(b_lon_min, b_lon_max)
        print '-'*50

        b_lon = np.linspace(b_lon_min, b_lon_max, num=nx)
        b_lat = np.linspace(b_lat_min, b_lat_max, num=ny)

        #p_lat = (p_lat - p_lat.min()) / ((p_lat - p_lat.min()).max())
        #p_lon = (p_lon - p_lon.min()) / ((p_lon - p_lon.min()).max())

        print "transformed lat range = {} - {}".format(p_lat.min(), p_lat.max())
        print "transformed lon range = {} - {}".format(p_lon.min(), p_lon.max())
        print '-'*50

        p_lat = (b_lat - p_lat.min()) / (p_lat.max() - p_lat.min())
        p_lon = (b_lon - p_lon.min()) / (p_lon.max() - p_lon.min())

        #print lat
        print "scaled lat range = {} - {}".format(p_lat.min(), p_lat.max())
        print "scaled lon range = {} - {}".format(p_lon.min(), p_lon.max())
        print '-'*50

        print "Building interpolant..."
        f = interp2d(lon, lat, w)

        print "Interpolating data..."
        #p_lon, p_lat = np.meshgrid(p_lon, p_lat)
        w_p = f(p_lon, p_lat)


        #w_p[mask==0] = np.nan


        image = render(p_lon, p_lat, w_p, width=width, height=height, lat=b_lat, lon=b_lon)

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
