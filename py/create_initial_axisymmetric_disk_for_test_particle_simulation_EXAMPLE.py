print("RUNNING NOW: create_initial_axisymmetric_disk_for_test_particle_simulation_EXAMPLE.py")
print("CODE VERSION: 2021-November-08")

#==============================
# Import packages
#==============================

# python:
from astropy.table import Table
import math
import numpy
import sys
import time

# galpy:
from galpy.df import quasiisothermaldf
from galpy.util import bovy_conversion

# RoadMapping:
sys.path.append('/u/twilm/research/RoadMapping/py')
from setup_pot_and_sf import setup_Potential_and_ActionAngle_object
from SF_Wedge import SF_Wedge

# other:
from calculate_actions_frequencies_angles import calculate_actions_frequencies_angles

#==============================
# Global constants
#==============================

# Constants for galpy units:
_REFR0 = 8.   # length unit in kpc
_REFV0 = 220. # velocity unit in km/s

# Number of cores to use in multiprocessing:
_MULTI = 40

# How many stars do we want:
_NSTARS = 100000

# Continue with existing file (=True) or generate new data set (=False):
_CONTINUE = False

#==========================
# Model Parameters
#==========================

#_____data_____
datatype = 1 #no measurement errors

#_____quasiisothermal df______
dftype  = 0
hr_kpc  = 2.5
sr0_kms = 33. # = 37./1.1 following Bovy & Rix 2013
sz0_kms = 25. # = 20./0.8 following Bovy & Rix 2013
hsr_kpc = 8.
hsz_kpc = 7.

#_____selection function______
sftype = 1 #wedge
Rmin_kpc = 3.
Rmax_kpc = 20.
zmin_kpc = -8.
zmax_kpc = +8.
phimin_deg = -180.
phimax_deg = +180.
sfPar_phys = numpy.array([Rmin_kpc,Rmax_kpc,zmin_kpc,zmax_kpc,phimin_deg,phimax_deg])

#_____grav. potential_____
pottype   = 4     # MWPotential2014 by Bovy 2015 + Staeckel actions
R0_kpc    = 8.    # Solar radius
vc_kms    = 220.  # Circular velocity at the Solar radius
potPar_phys = numpy.array([R0_kpc,vc_kms])

#_____numerical parameters_____
_N_SPAT = 25
_N_SPAT = 25
_NGL_VEL = 28
_N_SIGMA = 5.5

#==========================
# Set up galaxy
#==========================

#_____grav. potential_____
pot, aAS = setup_Potential_and_ActionAngle_object(
                    pottype,
                    potPar_phys
                    )
print("* Galpy scale parameters in this potential: *")
ro = R0_kpc/_REFR0
vo = vc_kms/_REFV0
print("ro = "+str(ro)+", vo = "+str(vo))


#_____qdf_____
hr  = hr_kpc /_REFR0/ro
sr  = sr0_kms/_REFV0/vo
sz  = sz0_kms/_REFV0/vo
hsr = hsr_kpc/_REFR0/ro
hsz = hsz_kpc/_REFR0/ro
qdf = quasiisothermaldf(
        hr,sr,sz,hsr,hsz,
        pot=pot,aA=aAS,
        cutcounter=True, 
        ro=ro
        )
        
#_____selection function_____
sf = SF_Wedge(
        sfPar_phys[0]/_REFR0/ro, #Rmin
        sfPar_phys[1]/_REFR0/ro, #Rmax
        sfPar_phys[2]/_REFR0/ro, #zmin
        sfPar_phys[3]/_REFR0/ro, #zmax
        sfPar_phys[4], #phimin
        sfPar_phys[5], #phimax
        df=qdf)

#==========================
# Prepare output
#==========================

# Where to store the data:
output_filename = '../data/test_particle_simulation/initial_axisymmetric_mockdata_for_test_particle_simulation_EXAMPLE.fits'

# Which coordinates will be stored in the data:
list_of_fields = ('R_kpc_start','z_kpc_start','phi_rad_start', 'x_kpc_start','y_kpc_start', \
                  'vR_kms_start','vz_kms_start','vT_kms_start',\
                  'JR_kpckms_start','Lz_kpckms_start','Jz_kpckms_start',\
                  'OmegaR_kmskpc_start','OmegaT_kmskpc_start','Omegaz_kmskpc_start',\
                  'wR_rad_start','wT_rad_start','wz_rad_start')
list_of_fields = numpy.array(list_of_fields,dtype=str)
n_fields_start = len(list_of_fields)

# Take into account existing mock data (or not):
if _CONTINUE: 

    print("* Read in existing mock data and continue *")

    # read existing output:
    hdul_start = fits.open(output_filename)
    tbdata_start = hdul_start[1].data
    hdul_start.close()

    # initialize empty array and fill with existing output:
    out_start = numpy.zeros((n_fields_start,_NSTARS)) + numpy.nan
    for kk in range(n_fields_start):
        out_start[kk,:]= tbdata_start.field(list_of_fields[kk])

    # Sum the occurences of NaN *after* each element.
    # If all of them are NaN, use this element as new starting index.
    sum_of_nans     = numpy.cumsum(numpy.isnan(out_start[0,:])[::-1])[::-1]
    sum_of_elements = numpy.cumsum(numpy.ones_like(out_start[0,:]))[::-1]
    index_start = numpy.where(sum_of_nans == sum_of_elements)[0][0]

else:   
    
    # initialize empty array and start index:
    out_start = numpy.zeros((n_fields_start,_NSTARS)) + numpy.nan
    index_start = 0
    
#==========================
# Sample mock data
#==========================
    
# how many stars do we need:
N_more_stars = _NSTARS - index_start

print("* Sample the DF in the SF spatially *")
Rs, zs, phis = sf.spatialSampleDF(
                    nmock  =N_more_stars,
                    nrs    =_N_SPAT,
                    nzs    =_N_SPAT,
                    ngl_vel=_NGL_VEL,
                    n_sigma=_N_SIGMA,
                    _multi =_MULTI,
                    quiet  =False
                    ) #galpy units                                                                                         

print("* Sample velocities for all mock star positions *")
vRs, vTs, vzs = sf.velocitySampleDF(Rs, zs, _multi=_MULTI)   #galpy units

# Transformation to physical units:
R_kpc_start   = Rs*_REFR0*ro      #[kpc]
z_kpc_start   = zs*_REFR0*ro      #[kpc]
phi_rad_start = phis/180.*math.pi #[rad]
vR_kms_start  = vRs*_REFV0*vo     #[km/s]
vz_kms_start  = vzs*_REFV0*vo     #[kms]
vT_kms_start  = vTs*_REFV0*vo     #[km/s]
x_kpc_start   = R_kpc_start * numpy.cos(phi_rad_start)
y_kpc_start   = R_kpc_start * numpy.sin(phi_rad_start)

#==============================
# Calculate Actions
#==============================

print("* Calculate actions, frequencies, and angles *")
JR_kpckms_start,Lz_kpckms_start,Jz_kpckms_start,\
OmegaR_kmskpc_start,OmegaT_kmskpc_start,Omegaz_kmskpc_start,\
wR_rad_start,wT_rad_start,wz_rad_start \
            = calculate_actions_frequencies_angles(
                aAS,
                _REFR0*ro,_REFV0*vo,
                R_kpc_start ,z_kpc_start ,phi_rad_start,
                vR_kms_start,vz_kms_start,vT_kms_start
                )

#==========================
# Save data
#==========================

#_____store data in array_____
out_start[ 0,index_start::] = R_kpc_start
out_start[ 1,index_start::] = z_kpc_start
out_start[ 2,index_start::] = phi_rad_start
out_start[ 3,index_start::] = x_kpc_start
out_start[ 4,index_start::] = y_kpc_start
out_start[ 5,index_start::] = vR_kms_start
out_start[ 6,index_start::] = vz_kms_start
out_start[ 7,index_start::] = vT_kms_start
out_start[ 8,index_start::] = JR_kpckms_start
out_start[ 9,index_start::] = Lz_kpckms_start
out_start[10,index_start::] = Jz_kpckms_start
out_start[11,index_start::] = OmegaR_kmskpc_start
out_start[12,index_start::] = OmegaT_kmskpc_start
out_start[13,index_start::] = Omegaz_kmskpc_start
out_start[14,index_start::] = wR_rad_start
out_start[15,index_start::] = wT_rad_start
out_start[16,index_start::] = wz_rad_start


#_____store output to file_____
t = Table(out_start.T, names=list_of_fields)
t.write(output_filename, format='fits', overwrite=True)

print('********************** THIS PROGRAM ENDED PROPERLY **********************')