print("RUNNING NOW: generate_mock_data_EXAMPLE_py3.py")
print("CODE VERSION: 2021-October-22")

#==============================
# Import packages
#==============================

# python:
import numpy
import os
import sys

# galpy:
from galpy.df import quasiisothermaldf
from galpy.util import save_pickles

# RoadMapping:
sys.path.append('/u/twilm/research/RoadMapping/py')
from setup_pot_and_sf import setup_Potential_and_ActionAngle_object
from SF_Wedge import SF_Wedge

#==============================
# Global constants
#==============================

# Constants for galpy units:
_REFR0 = 8.   # length unit in kpc
_REFV0 = 220. # velocity unit in km/s

# Number of cores to use in multiprocessing:
_MULTI = 40

#==============================
# Setup Potential pot
#==============================

# Potential Type parameter:
pottype   = 7 
# MWPotential(2014)-like potential:
# Miyamoto-Nagai disk, NFW halo, Hernquist bulge 
# using Staeckel actions
# (see setup_pot_and_sf.py and write_RoadMapping_parameters.py)

# Potential parameters:
R0_kpc      = 8.    # Solar radius
vc_kms      = 240.  # Circular velocity at the Solar radius
a_disk_kpc  = 2.5   # stellar disk scale length
b_disk_kpc  = 0.3   # stellar disk scale height
f_halo      = 0.3   # halo contribution to the disk's v_c^2
a_halo_kpc  = 18.   # dark matter halo scale length
potPar_phys = numpy.array([R0_kpc,vc_kms,a_disk_kpc,b_disk_kpc,f_halo,a_halo_kpc])

print("* Galpy scale parameters in this potential: *")
ro = R0_kpc/_REFR0
vo = vc_kms/_REFV0
print("ro = "+str(ro)+", vo = "+str(vo))

# Setup potential and ActionAngleStaeckel object:
pot, aAS = setup_Potential_and_ActionAngle_object(
                    pottype,
                    potPar_phys
                    )

#==============================
# Setup Stellar Distribution Function df
#==============================

# DF Type parameter:
dftype = 0 #quasiisothermal df (qDF) (Binney & McMillan 2011)
# (see write_RoadMapping_parameters.py)

# Parameters of the qDF:
hr_kpc  = 2.5 # radial scale length
sr0_kms = 33. # radial velocity dispersion at the solar radius
sz0_kms = 25. # vertical velocity dispersion at the solar radius
hsr_kpc = 8.  # radial-velocity-dispersion scale length
hsz_kpc = 7.  # vertial-velocity-dispersion scale length

# Setup qDF in galpy units:
hr  = hr_kpc /_REFR0/ro
sr  = sr0_kms/_REFV0/vo
sz  = sz0_kms/_REFV0/vo
hsr = hsr_kpc/_REFR0/ro
hsz = hsz_kpc/_REFR0/ro
df = quasiisothermaldf(
        hr,sr,sz,hsr,hsz,
        pot=pot,aA=aAS,
        cutcounter=True, 
        ro=ro
        )
# see https://docs.galpy.org/en/v1.6.0/reference/dfquasiisothermal.html for more details

#==============================
# Setup Selection Function sf
#==============================

# SF Type parameter 
sftype = 1 #wedge
# (see write_RoadMapping_parameters.py)

# Parameters of of wedge-shaped survey volume:
Rmin_kpc, Rmax_kpc = 5.,10.      # minimum and maximum galactocentric radius [kpc]
zmin_kpc, zmax_kpc = -1.,1.      # minimum and maximum height above the plane [kpc]
pmin_deg, pmax_deg = -180., 180. # minimum and maximum azimuth angle [degrees]

# Setup wedge-shaped sf in galpy units:
sf = SF_Wedge(Rmin_kpc/_REFR0/ro,
              Rmax_kpc/_REFR0/ro,
              zmin_kpc/_REFR0/ro,
              zmax_kpc/_REFR0/ro,
              pmin_deg,
              pmax_deg,
              df=df
              )

#==============================
# Generate Mock Data
#==============================

_NSTARS = 500 # number of mock stars to generate from the spatial stellar density

print("* Start sampling the DF spatially *")

# First sample the DF spatially (in the given pot and sf)
Rs, zs, phis = sf.spatialSampleDF(   #returns (R,z,phi) in galpy units
                       nmock=_NSTARS,        
                       nrs=20, nzs=20,   # number of points in the interpolation grid of the stellar_density(R,z)
                       ngl_vel=28,       # Gauss-Legendre order of integration of the DF over the velocities
                       n_sigma=5.5,      # integration range in vR and vz direction: +/- n_sigma x velocity dispersion
                       vT_galpy_max=1.5, # integration range in vT direction: from 0 to vT_galpy_max x _REFV0 * vo
                       quiet=False,      # write output to screen
                       _multi=_MULTI
                       )

print("* Start sampling velocities *")

# Secondly, sample velocities at these locations:
vRs,vTs,vzs = sf.velocitySampleDF( #returns velocities in galpy units
                        Rs,zs,
                        _multi=_MULTI
                        )   

# Back to physical units:
Rs_kpc   = Rs*_REFR0*ro          #[kpc]
zs_kpc   = zs*_REFR0*ro          #[kpc]
phis_deg = phis                  #[deg]
vRs_kms  = vRs*_REFV0*vo         #[km/s]
vzs_kms  = vzs*_REFV0*vo         #[kms]
vTs_kms  = vTs*_REFV0*vo         #[km/s]

#==============================
# Save data to file
#==============================

print("* Save data *")

mockdatapath = '../data/' # path where the data lives
datasetname  = 'mock_data_exampleB_py3' # name of the data set

directory = mockdatapath+'/'+datasetname+'/'
if not os.path.exists(directory):
    os.makedirs(directory)
    
datafilename = mockdatapath+'/'+datasetname+'/'+datasetname+'_mockdata.sav'

save_pickles(datafilename,\
            Rs_kpc,   vRs_kms,
            phis_deg, vTs_kms,
            zs_kpc,   vzs_kms
            )

print('********************** THIS PROGRAM ENDED PROPERLY **********************')