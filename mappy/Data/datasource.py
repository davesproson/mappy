import abc
import datetime
import glob

from netCDF4 import Dataset, num2date
from numpy import squeeze

__all__ = ['DataCollection']

class ForecastTime(object):
    def __init__(self, ref_time=None, valid_time=None, filename=None,
                       index=None):
        self.ref_time = ref_time
        self.valid_time = valid_time
        self.filename = filename
        self.index = index

    def __str__(self):
        return "FC: {} - {} @ {}[{}]".format(self.ref_time,
                    self.valid_time, self.filename, self.index)


class DataCollection(object):
    
    def __init__(self, file_glob=None, lat_var=None, lon_var=None,
                       elevation_var=None, time_var=None, data_type='netcdf'):

        self._glob = file_glob
        self._data_type = data_type
        self._lat_var = lat_var
        self._lon_var = lon_var
        self._elevation_var = elevation_var
        self._time_var = time_var

        self._reader = DataReaderMap[self._data_type](
                        lon_var=self._lon_var, lat_var=self._lat_var,
                        time_var=self._time_var)


        self._files = [f for f in glob.glob(self._glob)]

        self._time_map = []
        for f in self._files:
            self._time_map += [
                ForecastTime(ref_time=None, valid_time=t,
                             filename=f, index=i)
                    for t, i in self._reader.get_times(f)]

    def get_available_times(self):
        return [f.valid_time for f in self._time_map]

    def get_data_layer(self, var_name=None, time=None):
        for i in self._time_map:
            if i.valid_time == time:  
                return self._reader.get_data_layer(
                        var_name=var_name, time_level=i.index,
                        filename=i.filename)



class DataReader(object):
    
    def __init__(self, lon_var=None, lat_var=None, elevation_var=None,
                        time_var=None):

        self._lat_var = lat_var
        self._lon_var = lon_var
        self._time_var = time_var
        self._elevation_var = elevation_var

    @abc.abstractmethod
    def get_times(self, filename=None):
        '''Return all of the times in a data file'''
        return

    @abc.abstractmethod
    def get_data_level(self, filename=None, elevation=None,
                        time=None):
        return

class NetCDFDataReader(DataReader):
    
    def get_times(self, filename=None):
        nc = Dataset(filename, 'r')
        time_unit = nc.variables[self._time_var].units
        time_data = nc[self._time_var][:]
        time_data = num2date(time_data, time_unit)
        nc.close()

        return [(t, i) for i, t in zip(range(len(time_data)), time_data)]

    def get_data_layer(self, var_name=None, filename=None, time_level=None,
                        elevation_level=None):

        nc = Dataset(filename, 'r')
        if time_level and elevation_level:
            data = nc[var_name][time_level, elevation_level, :, :]
        elif time_level and not elevation_level:
            data = nc[var_name][time_level, :, :]
        elif elevation_level and not time_level:
            data = nc[var_name][elevation_level, :, :]
        else:
            data = nc[var_name][:, :]

        nc.close()
        print data
        return squeeze(data)


DataReaderMap = {
    'netcdf': NetCDFDataReader}
