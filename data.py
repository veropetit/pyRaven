import numpy as np
import matplotlib.pyplot as plt
import h5py
import specpolFlow as pol
import copy
import matplotlib.pyplot as plt

# written by Robin Moore, Veronique Petit, and Patrick Stanley

# class for a collection of LSD profiles 
# with meta information for the star, 
# and also for each individual LSD profiles.
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
  Holds a list of LSD profiles

  :param lsds: List of lsd profiles objects
  """
  def __init__(self, lsds):
    """
    Initialization method for LSDprofs.

    :param self: object being initialized
    :param lsds: list of lsd profiles objects
    """
    self.lsds = lsds

  def write(self, f, obs_names):
    """
    Helper function that writes the contents of an LSDProfs object to an h5 file

    :param f: h5 file being written
    :param lsds: LSDprofs object being stored in the file
    :param obs_names: (List of string) the IDs to be used for each observations

    :rtype: none
    """

    def write_lsd(lsd, f):
      """
      Helper function that writes the contents of an lsd_prof to an h5 file

      :param f: h5 file being written
      :param lsd: lsd_prof being stored in the file

      :rtype: none
      """
      f.create_dataset('vel', data = lsd.vel)
      f.create_dataset('specI', data = lsd.specI)
      f.create_dataset('specSigI', data = lsd.specSigI)
      f.create_dataset('specV', data = lsd.specV)
      f.create_dataset('specSigV', data = lsd.specSigV)
      f.create_dataset('specN1', data = lsd.specN1)
      f.create_dataset('specSigN1', data = lsd.specSigN1)
      f.create_dataset('specN2', data = lsd.specN2)
      f.create_dataset('specSigN2', data = lsd.specSigN2)
      f.create_dataset('header', data = lsd.header)

    for i in range(0, len(self.lsds)):
      write_lsd(self.lsds[i], f.create_group(obs_names[i]))

  def scale(self, vrad, Ic, wint_data, wpol_data, wint_rav, wpol_rav):

    lsd_scaled = copy.deepcopy(self)
    for i in range(0,len(lsd_scaled.lsds)):
      # Radial velocity correction
      lsd_scaled.lsds[i] = lsd_scaled.lsds[i].vshift(vrad[i])
      # Renormalization to the continuum. 
      lsd_scaled.lsds[i] = lsd_scaled.lsds[i].norm(Ic[i])
      # scaling to the desired LSD weigth for raven computation
      lsd_scaled.lsds[i] = lsd_scaled.lsds[i].set_weights(wint_data[i], wpol_data[i], wint_rav, wpol_rav)

    return(lsd_scaled)

  def cut(self, fitrange):
    lsd_cut = copy.deepcopy(self)
    for i in range(0,len(lsd_cut.lsds)):
      inline = np.logical_and(lsd_cut.lsds[i].vel>=-1*fitrange, lsd_cut.lsds[i].vel<=fitrange)
      lsd_cut.lsds[i] = lsd_cut.lsds[i][inline]
    return(lsd_cut)

  def plotI(self, ax, label=[]):
    for i in range(0,len(self.lsds)):
      if not(label):
        ax.plot(self.lsds[i].vel, self.lsds[i].specI)
      else:
        ax.plot(self.lsds[i].vel, self.lsds[i].specI,label=label[i])
    ax.set_xlabel('Velocity (km/s)')
    ax.set_ylabel('I / Ic')
    return(ax)

  def plot(self, fig=None, ax=None, labels=None):
      if fig is None:
        fig, ax = plt.subplots(4,1)

      for item in self.lsds:
        ax[3].plot(item.vel, item.specI)
        ax[2].plot(item.vel, item.specN1)
        ax[1].plot(item.vel, item.specN2)
        ax[0].plot(item.vel, item.specV)
          
      lim = np.max([np.abs(ax[0].get_ylim()), np.abs(ax[1].get_ylim()), np.abs(ax[2].get_ylim())] )
      for item in ax.flatten()[0:2]:
        item.set_ylim(-1*lim, lim)

      ax[3].set_ylabel('I/Ic')
      ax[2].set_ylabel('N1/Ic')
      ax[1].set_ylabel('N2/Ic')
      ax[0].set_ylabel('V/Ic')

      plt.tight_layout()
      
      return(fig, ax)


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
  def __init__(self, star_name, nobs, obs_names,
              vrad, Ic, wint_data, wpol_data, wint_rav, wpol_rav,
              fitrange, vsini,
              original, scaled, cutfit):
    """
    Initialization method for DataPacket
    :param self: The object being initialized
    :param fitrange: fitrange for the packet
    :param vsini:
    :param wint_rav:
    :param wpol_rav:
    :param original:
    :param scaled:
    :param cutfit:
    """

    self.star_name = star_name
    self.nobs = nobs
    self.obs_names = obs_names
    self.vrad = vrad
    self.Ic = Ic
    self.wint_data = wint_data
    self.wpol_data = wpol_data
    self.wint_rav = wint_rav
    self.wpol_rav = wpol_rav
    self.fitrange = fitrange
    self.vsini = vsini

    self.original = original
    self.scaled = scaled
    self.cutfit = cutfit

  def write(self, fname):
    """
    Method that writes a data packet to an h5 file.

    :param packet: DataPacket object being stored in the file
    :param fname: name of the file to be written

    :rtype: none
    """

    with h5py.File(fname,'w') as f:
      f.create_dataset('star_name', data = self.star_name)
      f.create_dataset('nobs', data = self.nobs)
      f.create_dataset('obs_names', data = self.obs_names)
      f.create_dataset('vrad', data = self.vrad)
      f.create_dataset('Ic', data = self.Ic)
      f.create_dataset('wint_data', data=self.wint_data)
      f.create_dataset('wpol_data', data=self.wpol_data)
      f.create_dataset('wint_rav', data=self.wint_data)
      f.create_dataset('wpol_rav', data=self.wpol_data)

      f.create_dataset('fitrange', data = self.fitrange)
      f.create_dataset('vsini', data = self.vsini)

      f_original = f.create_group('original')
      self.original.write(f_original, self.obs_names)
      
      f_scaled = f.create_group('scaled') 
      self.scaled.write(f_scaled, self.obs_names)

      f_cutfit = f.create_group('cutfit')
      self.cutfit.write(f_cutfit, self.obs_names)

  def plotI(self):
    # Written by Patrick Stanley

    fig, ax = plt.subplots(1,3, figsize=(30,10))

    #Set titles
    ax[0].set_title('Original')
    ax[1].set_title('Velocity corrected and normalized')
    ax[2].set_title('Data Cut to Fitting region')
    
    #Loops over each observation
    for i in range(0, self.nobs):
      vrad=self.vrad[i]
      Ic=self.Ic[i]
      lsd=self.original.lsds[i]
      #color = next(ax[0]._get_lines.prop_cycler)['color']
      color = ax[0]._get_lines.get_next_color()

      ax[0].plot(lsd.vel, lsd.specI, c=color,
                label='Obs: {}, vrad: {}, Ic: {}'.format(i+1, vrad, Ic))
      ax[0].axhline(y=float(Ic), c=color, ls='--') #horizonatal line at Ic
      ax[0].axvline(x=vrad, ls='--', c=color) #vertical line at vrad

      #repeat for the scaled plot
      lsd_scaled=self.scaled.lsds[i]
      ax[1].plot(lsd_scaled.vel, lsd_scaled.specI, c=color)

      #repeat for the cutfit plot
      lsd_cutfit=self.cutfit.lsds[i]
      ax[2].plot(lsd_cutfit.vel, lsd_cutfit.specI, c=color)

    ax[1].axhline(y=1.0, c='k', ls='--')
    ax[1].axvline(x=0.0, ls='--', c='k')
    ax[2].axhline(y=1.0, c='k', ls='--')
    ax[2].axvline(x=0.0, ls='--', c='k')

    ax[1].axhline(y=1.0, ls='--', c='k') #horizonal line at 1
    #vertical lines at +/- fitrange
    ax[1].axvline(x=float(self.fitrange), ls='--', c='k',label='Fitrange: {}'.format(self.fitrange))
    ax[1].axvline(x=-1*float(self.fitrange), ls='--', c='k')
    #vertical lines at +/- vsini
    ax[1].axvline(x=float(self.vsini), ls='-.', c='k',label='vsini: {}'.format(self.vsini))
    ax[1].axvline(x=-1*float(self.vsini), ls='-.', c='k')

    #repeat for cutfit graph
    ax[2].axhline(y=1.0, ls='--', c='k')
    ax[2].axvline(x=float(self.fitrange), ls='--', c='k',label='Fitrange: {}'.format(self.fitrange))
    ax[2].axvline(x=-1*float(self.fitrange), ls='--', c='k')
    ax[2].axvline(x=float(self.vsini), ls='-.', c='k',label='vsini: {}'.format(self.vsini))
    ax[2].axvline(x=-1*float(self.vsini), ls='-.', c='k')

    #make labels and vertical line at x=0
    for item in ax:
      item.set_xlabel('Velocity (km/s)')
      item.set_ylabel('I/Ic')

    #show legends
    ax[0].legend(loc=0)
    ax[1].legend(loc=0)

    return(fig, ax)


def read_lsds_from_sfiles(fnames):
  '''
  Function to create a LSDprof object from a list of filenames for LSD profiles in the .s format
  '''
  lsds = []
  for fname in fnames:
    lsds.append(pol.read_lsd(fname))
  return(LSDprofs(lsds))


def create_Packet(star_name, nobs, obs_names,
            vrad, Ic, wint_data, wpol_data, wint_rav, wpol_rav,
            fitrange, vsini,
            original):
  '''
  Create a new DataPacket from meta information and a LSDprof object with the original LSD profiles. 
  The creator function will calculate the scaled (radial velocity correction, normalization, LSD weigth scaling) version of the LSD profiles
  based on the meta information provided. 

  :param star_name: (string) the name (or ID) for the star
  :param nobs: (int) the number of LSD profiles to be analyzed together
  :param obs_names: (list of strings) ID for the observations to be used in e.g. graphs (e.g. filename, obs number, etc)
  :param vrad: (1D array) radial velocity for the LSD profiles to put them in the stellar reference frame (if the LSD profiles are already in the star reference frame, just put an array of zeros)
  :param Ic: (1D array) continuum normalization values for the LSD profiles (if the LSD profiles are already normalized, just input an array of ones)
  :param wint_data: (1D array) the intensity weigth that was used to calculate the LSD profiles
  :param wpol_data: (1D array) the polarization weigth that was used to calculate the LSD profiles
  :param wint_rav: (float) the common LSD intensity weigth that will be used for all of the observations
  :param wpol_rav: (float) the common LSD intensity weigth that will be used for all of the observations
  :param fitrange: (float) the velocity range around the line center that will be use for the fitting
  :param vsini: (float) the vsini of the star (RIGHT NOW, ONLY FOR DISPLAY PURPOSES)
  :param original: (LSDprofs object) the original LSD profiles
  '''
  
  scaled = original.scale(vrad, Ic, wint_data, wpol_data, wint_rav, wpol_rav)
  cutfit = scaled.cut(fitrange)

  Packet = DataPacket(star_name, nobs, obs_names,
            vrad, Ic, wint_data, wpol_data, wint_rav, wpol_rav,
            fitrange, vsini,
            original, scaled, cutfit)

  return(Packet)

def read_packet(fname):
  """
  Function that reads an h5 file and produces a DataPacket object from the data.

  :param fname: name of the h5 file being read

  :rtype DataPacket:
  """
  with h5py.File(fname) as f:
    nobs = (f['nobs'])[()]
    star_name = (f['star_name'])[()]
    obs_names = (f['obs_names'])[()]
    vrad = (f['vrad'])[()]
    Ic = (f['Ic'])[()]
    wint_data = (f['wint_data'])[()]
    wpol_data = (f['wpol_data'])[()]
    wint_rav = (f['wint_rav'])[()]
    wpol_rav = (f['wpol_rav'])[()]
    fitrange = (f['fitrange'])[()]
    vsini = (f['vsini'])[()]
    original = read_lsds(f['original'], obs_names)
    scaled = read_lsds(f['scaled'], obs_names)
    cutfit = read_lsds(f['cutfit'], obs_names)
    return DataPacket(star_name, nobs, obs_names,
              vrad, Ic, wint_data, wpol_data, wint_rav, wpol_rav,
              fitrange, vsini,
              original, scaled, cutfit)
    
def read_lsds(f, obs_names):
  """
  Helper function that reads an LSDprofs object from an h5 file

  :param f: the h5 file with the data
  :param name: name of the LSDprof object (e.g. original, scaled, or cutfit)
  :param nobs: the number of LSD profiles in the LSDprof object. 

  :rtype LSDprof:
  """
  lsds = []
  for obsname in obs_names:
    lsds.append(read_lsd(f[obsname]))
  lsd_profs = LSDprofs(lsds)
  return lsd_profs

def read_lsd(f):
  """
  Helper function that reads an lsd_prof from an h5 file

  :param f: the h5 file with the data
  :param name: name of the LSDprofs object the data is a part of, with the index in the list

  :rtype lsd_prof:
  """
  vel = np.array(f['vel'])
  specI = np.array(f['specI'])
  specSigI = np.array(f['specSigI'])
  specV = np.array(f['specV'])
  specSigV = np.array(f['specSigV'])
  specN1 = np.array(f['specN1'])
  specSigN1 = np.array(f['specSigN1'])
  specN2 = np.array(f['specN2'])
  specSigN2 = np.array(f['specSigN2'])
  header = (f['header'])[()]
  lsd = pol.LSD(vel, specI, specSigI, specV, specSigV, specN1, specSigN1, specN2=specN2, specSigN2=specSigN2, header=header)
  return lsd

