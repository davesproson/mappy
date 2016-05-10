__all__ = ['WMSGetMapRequest']

class WMSGetMapRequest(object):
    
    def __init__(self, **kwargs):

        self.required_args = ['width', 'height', 'crs', 'bbox']

        a = {key.lower(): value[-1] for key, value in kwargs.items()}
        self.__dict__.update(a)


        print a

        self.validate()

    def validate(self):
        self.width = int(self.width)
        self.height = int(self.height)
        self.bbox = [float(i) for i in self.bbox.split(',')]
        self.crs = self.crs
        self.format = self.format


        
        
