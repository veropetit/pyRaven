import numpy as np
import matplotlib.pyplot as plt
import h5py
import specpolFlow as pol


# class for a collection of LSD profiles (let's call it a LSDprofs object), with meta information for the star, and also for each individual LSD profiles.
## meta information for the class
# star name
# number of LSD profiles
## meta information for each LSD profile in the collection:
# filename
# vrad
# Ic
## the LSD profiles themselves (an array of LSD profile objects, so that I can iterate over the profiles?)

class LSDprofs:
  """
  Holds a list of LSD profiles, including metainfo about each star and profile.

  star_name: Name of the star
  nobs: Number of observations
  fname: List of the file names for each lsd profile.
  vrad: List of the vrads for each lsd profile.
  ic: List of the Ics for each lsd profile.
  lsds: List of the lsd profiles.
  """
  def __init__(self, star_name, nobs, fname, vrad, ic, lsds = [], **kwargs):
    """
    Initialization method for LSDprofs.

    :param self: object being initialized
    :param star_name: star name
    :param nobs: number of observations
    :param fname: names of the lsd data files
    :param vrad: list of vrads of the lsd profiles
    :param ic: list of Ics for the lsd profiles
    :param lsds: list of lsd profiles
    """
    self.star_name = star_name
    self.nobs = nobs
    self.fname = fname
    self.vrad = vrad
    self.Ic = ic

    #self.lsds = []
    #as i've just learned, empty lists are considered false!
    if (not lsds):
      for name in fname:
        self.lsds = []
        new_lsd = pol.iolsd.read_lsd(name)
        self.lsds.append(new_lsd)
    else:
      self.lsds = lsds

# class for a 'DataPacket'
## meta information
# fitrange
# vsini
## Original: a LSDprofs object
## Scaled: a LSDprofs object
## Cutfit: a LSDprofs object

class DataPacket:
  """
  Packet of data, convertable back and forth to and h5 file.

  fitrange:
  vsini:
  wint_data:
  wpol_data:
  wint_rav:
  wpol_rav:
  original:
  scaled:
  cutfit:
  """
  def __init__(self, fitrange, vsini,
               wint_data, wpol_data, wint_rav, wpol_rav,
               original, scaled, cutfit):
    """
    Initialization method for DataPacket
    :param self: The object being initialized
    :param fitrange: fitrange for the packet
    :param vsini:
    :param wint_data:
    :param wpol_data:
    :param wint_rav:
    :param wpol_rav:
    :param original:
    :param scaled:
    :param cutfit:
    """
    self.fitrange = fitrange
    self.vsini = vsini
    self.wint_data = wint_data
    self.wpol_data = wpol_data
    self.wint_rav = wint_rav
    self.wpol_rav = wpol_rav

    self.original = original
    self.scaled = scaled
    self.cutfit = cutfit

## I need a write method to store a DataPacket object (including the data themselved and the meta data as attributed) into a h5 file.

## From Vero to Robin: maybe this function would be bettwr as a method of the DataPacket class?
## e.g. Packet.write(file)?
def write_packet(packet, fname):
  """
  Method that writes a data packet to an h5 file.

  :param packet: DataPacket object being stored in the file
  :param fname: name of the file to be written

  :rtype: none
  """
  with h5py.File(fname,'w') as f:
    f.create_dataset('fitrange', data = packet.fitrange)
    f.create_dataset('vsini', data = packet.vsini)
    f.create_dataset('wint_data', data = packet.wint_data)
    f.create_dataset('wpol_data', data = packet.wpol_data)
    f.create_dataset('wint_rav', data = packet.wint_rav)
    f.create_dataset('wpol_pol', data = packet.wpol_rav)

    write_lsds(f, packet.original, 'original')
    write_lsds(f, packet.scaled, 'scaled')
    write_lsds(f, packet.cutfit, 'cutfit')

def write_lsds(f, lsds, name):
  """
  Helper function that writes the contents of an LSDProfs object to an h5 file

  :param f: h5 file being written
  :param lsds: LSDprofs object being stored in the file
  :param name: name of the LSDprofs object

  :rtype: none
  """
  for i in range(0, lsds.nobs):
    write_lsd(f, lsds.lsds[i], name + '_' + str(i))
  f.create_dataset(name+'_nobs', data=lsds.nobs)
  f.create_dataset(name+'_star_name', data=lsds.star_name)
  f.create_dataset(name+'_fname', data=lsds.fname)
  f.create_dataset(name+'_vrad', data=lsds.vrad)
  f.create_dataset(name+'_Ic', data=lsds.Ic)

def write_lsd(f, lsd, name):
  """
  Helper function that writes the contents of an lsd_prof to an h5 file

  :param f: h5 file being written
  :param lsd: lsd_prof being stored in the file
  :param name: name of the LSDprofs object and the index of the lsd_prof in the LSDprofs' list.

  :rtype: none
  """
  f.create_dataset(name + '_vel', data = lsd.vel)
  f.create_dataset(name + '_specI', data = lsd.specI)
  f.create_dataset(name + '_specSigI', data = lsd.specSigI)
  f.create_dataset(name + '_specV', data = lsd.specV)
  f.create_dataset(name + '_specSigV', data = lsd.specSigV)
  f.create_dataset(name + '_specN1', data = lsd.specN1)
  f.create_dataset(name + '_specSigN1', data = lsd.specSigN1)
  f.create_dataset(name + '_specN2', data = lsd.specN2)
  f.create_dataset(name + '_specSigN2', data = lsd.specSigN2)
  f.create_dataset(name + '_header', data = lsd.header)

## I will also need a read method to get a h5 file back into a DataPacket object.

def read_packet(fname):
  """
  Function that reads an h5 file and produces a DataPacket object from the data.

  :param fname: name of the h5 file being read

  :rtype DataPacket:
  """
  with h5py.File(fname) as f:
    fitrange = (f['fitrange'])[()]
    vsini = (f['vsini'])[()]
    wint_data = (f['wint_data'])[()]
    wpol_data = (f['wpol_data'])[()]
    wint_rav = (f['wint_rav'])[()]
    wpol_rav = (f['wpol_data'])[()]
    original = read_lsds(f, 'original')
    scaled = read_lsds(f, 'scaled')
    cutfit = read_lsds(f, 'cutfit')
    return DataPacket(fitrange, vsini,
                      wint_data, wpol_data, wint_rav, wpol_rav,
                      original, scaled, cutfit)
    
def read_lsds(f, name):
  """
  Helper function that reads an LSDprofs object from an h5 file

  :param f: the h5 file with the data
  :param name: name of the LSDprof object

  :rtype LSDprof:
  """
  nobs = (f[name+'_nobs'])[()]
  star_name = (f[name+'_star_name'])[()]
  fname = (f[name+'_fname'])[()]
  vrad = (f[name+'_vrad'])[()]
  Ic = (f[name+'_Ic'])[()]
  lsds = []
  for i in range(0, nobs):
    lsds.append(read_lsd(f, name + '_' + str(i)))
  lsd_profs = LSDprofs(star_name, nobs, fname, vrad, Ic, lsds)
  return lsd_profs

def read_lsd(f, name):
  """
  Helper function that reads an lsd_prof from an h5 file

  :param f: the h5 file with the data
  :param name: name of the LSDprofs object the data is a part of, with the index in the list

  :rtype lsd_prof:
  """
  vel = np.array(f[name+'_vel'])
  specI = np.array(f[name+'_specI'])
  specSigI = np.array(f[name+'_specSigI'])
  specV = np.array(f[name+'_specV'])
  specSigV = np.array(f[name+'_specSigV'])
  specN1 = np.array(f[name+'_specN1'])
  specSigN1 = np.array(f[name+'_specSigN1'])
  specN2 = np.array(f[name+'_specN2'])
  specSigN2 = np.array(f[name+'_specSigN2'])
  header = (f[name+'_header'])[()]
  lsd = pol.iolsd.lsd_prof(vel, specI, specSigI, specV, specSigV, specN1, specSigN1, specN2=specN2, specSigN2=specSigN2, header=header)
  return lsd

