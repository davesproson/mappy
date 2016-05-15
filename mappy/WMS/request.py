import pyproj

__all__ = ['WMSGetMapRequest']

class WMSGetMapRequest(object):
    
    def __init__(self, **kwargs):

        self.required_args = ['width', 'height', 'crs', 'bbox']

        a = {key.lower(): value[-1] for key, value in kwargs.items()}
        self.__dict__.update(a)


        print a

        self.validate()
        self.normalize_bbox()

    def validate(self):
        self.width = int(self.width)
        self.height = int(self.height)
        self.bbox = [float(i) for i in self.bbox.split(',')]
        try:
            self.crs = self.crs
            self.srs = self.crs
        except:
            self.srs = self.srs
            self.crs = self.srs

        self.version = self.version
        self.format = self.format

    def normalize_bbox(self):

        p = pyproj.Proj('+init=EPSG:4326')
        print "***SRS={}".format(p.srs)

        if self.version == '1.3.0':
            self.bbox = [self.bbox[1], self.bbox[0],
                         self.bbox[3], self.bbox[2]]

            wlon, wlat = pyproj.transform(
                                pyproj.Proj('+init={}'.format(self.crs)),
                                pyproj.Proj('+init=EPSG:4326'),
                                [self.bbox[0], self.bbox[2]],
                                [self.bbox[1], self.bbox[3]])

            self.wgs84_bbox = [wlon[0], wlat[0], wlon[1], wlat[1]]


        
        
