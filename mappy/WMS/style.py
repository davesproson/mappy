from ConfigParser import SafeConfigParser
import json

class Style(object):
    
    def __init__(self, render_function=None, render_args=dict(),
                    mask=None, refine=0, name=None, title=None,
                    abstract=None):

        self.render_function = render_function
        self.render_args = render_args
        self.mask = mask
        self.refine = refine

        if self.refine < 0:
            raise ValueError('Data refinement must be a +ve integer')

    def __repr__(self):
        return '<Style: {!r} ({!r})>'.format(self.name, self.render_function)

    def __str__(self):
        return 'WMS Style[{}]: {}'.format(self.name, self.abstract)




class StyleReader(object):

    def __init__(self, ini_file):
        
        self.ini_file = ini_file
        self.styles = []

        self.config = SafeConfigParser()
        self.config.read(self.ini_file)

        for section in self.config.sections():
            style = self._read_style(section)
            if style:
                self.styles.append(style)


    def __repr__(self):
        return 'StyleReader({!r})'.format(self.ini_file)


    def _read_style(self, section):
        default_style = Style()
        default_style.render_args = {}

        # much of this could be done via __dict__.update()
        default_style.name = section
        for name, value in self.config.items(section):
            print "{}: {} = {}".format(section, name, value)
            name = name.lower()

            if name == 'title':
                default_style.title = value
                continue
            if name == 'abstract':
                default_style.abstract = value
                continue
            if name == 'mask':
                default_style.mask = None
                if value:
                    if (value.lower() == 'land' 
                            or value.lower() == 'ocean'):
                        default_style.mask = value
                continue
            if name == 'render_args':
                args = json.loads(self.config.get(section, name).strip())
                if args:
                    for arg in args:
                        a = arg.split('=')
                        if len(a) == 2:
                            default_style.render_args.update({a[0]: a[1]})
                continue
            if name == 'render_function':
                default_style.render_function = value

        return default_style

        
