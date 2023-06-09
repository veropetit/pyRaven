import json
import numpy as np
 
class MyJSONEncoder(json.JSONEncoder): 
    '''Utility class to convert numpy arrays to list when using the json encoding
    
    From https://python-forum.io/thread-35245.html
    '''   
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class parameters(dict):
    '''
    Class to add functionalities to the `param` dictionaries. 

    Usage: to param = parameters(param) to use the `write` and `pprint` functions. 

    ADD LINK to info page about the param dictionaries
    '''
    def write(self, fname):
        '''
        Method to write the parameter dictionary to a json file

        :param self: a `parameters` dictionary that can contain numpy arrays (they will be converted to list)
        :param fname: (string) the name of the .json file to write
        '''
        with open(fname, "w") as f:
            json.dump(self, f, separators=(',', ':'), sort_keys=True, indent=4, cls=MyJSONEncoder)

    def pprint(self):
        '''
        Method to pretty-print a `parameters` dictionary in a readable manner
        '''
        print(json.dumps(self, indent=2,default=str))

def read_parameters(fname):
    '''
    Function to read a parameters dictionary that has been written to file
    as a .json file. 
    IMPORTANT NOTE: if some of the keys originally contained a numpy array prior to being written to file,
    they will now be in the form of python lists. It is important to transform them back into numpy array if you want to use them as such. 

    TODO: decide if we want to return a plain dictionary or a parameters class dictionary?

    :param fname: the name of the .json file
    '''
    with open(fname, "r") as f:
        param = parameters(json.load(f))
    return(parameters(param))

def get_def_param_fitI():
    '''
    Function that returns a `parameters` dictionary with default values 
    for parameters that are necessary for the intensity profile fitting
    '''

    genparam = {
    'lambda0':5000,    # the central wavelength of the transition
    'vsini':50.0,         # the projected rotational velocity
    'vdop':10.0,          # the thermal broadening
    'av':0.05,             # the damping coefficient of the Voigt profile
    'bnu':1.5,             # the slope of the source function with respect to vertical optical depth
    'logkappa':0.98,          # the line strength parameter
    'ndop':int(10)       # the number of sample point per doppler width for the wavelength array
    }
    return(parameters({'general' : genparam}))

def get_def_param_weak():
    '''
    Function that returns a `parameters` dictionary with default values 
    for parameters that are necessary for a typical weak field calculation
    '''

    genparam = {
    'lambda0':5000,    # the central wavelength of the transition
    'vsini':50.0,         # the projected rotational velocity
    'vdop':10.0,          # the thermal broadening
    'av':0.05,             # the damping coefficient of the Voigt profile
    'bnu':1.5,             # the slope of the source function with respect to vertical optical depth
    'logkappa':0.98,          # the line strength parameter
    'ndop':int(10),       # the number of sample point per doppler width for the wavelength array
    'Bpole':1000,       # dipolar field strength in Gauss
    'incl':90,          # inclination of the rotation axis to the LOS
    'beta':90,          # obliquity of the field to the rotation axis
    'phase':0           # rotational phase
    }
    weakparam = {
        'geff':1.0
    }

    return(parameters({'general' : genparam,'weak' : weakparam}))