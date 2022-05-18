print("RUNNING NOW: integrate_mock_data_in_quadrupole_bar_potential_EXAMPLE.py")
print("CODE VERSION: 2022-May-18")

#==============================
# Import packages
#==============================

# python:
from astropy.io import fits
from astropy.table import Table
import math
import multiprocessing
import numpy
from pathlib import Path
from scipy.fftpack import fft, fftfreq
import sys

# galpy:
from galpy.orbit import Orbit
from galpy.potential import DehnenBarPotential
from galpy.util import bovy_conversion, multi

# RoadMapping:
sys.path.append('/u/twilm/research/RoadMapping/py')
from setup_pot_and_sf import setup_Potential_and_ActionAngle_object

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

# Continue or overwrite existing simulation:
_CONTINUE = True

#==============================
# Bar potential parameters
#==============================

# bar pattern speed [units of km/s/kpc]:
bar_omega_kmskpc = 40.
name_speed = '40kmskpc'

# bar starts to grow at [units of bar periods]:
t_form = 0.

# bar is fully grown at [units of bar periods]:
t_steady = 5.

# total orbit integration time [units of bar periods]:
t_final = 50.

# output times [units of bar periods]:
t_out_period       = numpy.array([10., 20., 30., 40., 50.])
output_times_names = numpy.array(['10','20','30','40','50'],dtype=str)

# bar angle wrt Sun (which is at phi=0) at t=t_out_period:
bar_angle_deg = 25.

# bar length --> such that R_CR/R_bar = 1.2:
Rbar_kpc = 4.5 

# bar component(s), i.e. m = ...:
Fourier_m = 2. # pure quadrupole! (or use for example: numpy.array([2.,4.]))

# density slopes of the bar component(s):
slope_p = -3. # (or use for example: numpy.array([-3.,-5.]))
                      
# bar strength at R=galpy_scale_length of the bar component(s):
alpha_bar = 0.01 # weak bar (or use for example: numpy.array([0.01,-0.0015]))
# adding a negative alpha_(m=4) component creates a "boxy" bar, positive a "pointy" bar
name_shape = 'm2a01' # (or use for example: 'm2a01_m4a0015boxy')

#==============================
# Setup axisymmetric potential
#==============================

# Potential type: 
# MWPotential2014 by Bovy (2015)
pottype = 4

# Potential parameters for MWPotential2014:
R0_kpc      = 8.    # Solar radius
vc_kms      = 220.  # Circular velocity at the Solar radius
potPar_phys = numpy.array([R0_kpc,vc_kms])

print("* Galpy scale parameters in this potential: *")
ro = R0_kpc/_REFR0
vo = vc_kms/_REFV0
print("ro = "+str(ro)+", vo = "+str(vo))

# Setup potential and ActionAngleStaeckel object:
pot_axi, aAS_axi = setup_Potential_and_ActionAngle_object(
    pottype,potPar_phys)

# Galpy scale units:
galpy_scale_length   = _REFR0 * ro
galpy_scale_velocity = _REFV0 * vo

#==============================
# Setup bar potential
#==============================

#parameters to galpy units:
bar_omega_galpy = bar_omega_kmskpc/bovy_conversion.freq_in_kmskpc(galpy_scale_velocity, galpy_scale_length) # Pattern speed fo the bar
bar_angle_rad   = bar_angle_deg/180.*numpy.pi # Bar Angle
Rbar_galpy      = Rbar_kpc/galpy_scale_length             # Bar radius
Af_bar          = -alpha_bar/slope_p*(Rbar_kpc/galpy_scale_length)**slope_p # bar strength
    
if not isinstance(Fourier_m,numpy.ndarray) and (Fourier_m == 2):
    
    # For a 1-component, m=2 (i.e. pure quadrupole) bar, use the standard DehnenBarPotential:
    bar = DehnenBarPotential(omegab=bar_omega_galpy,
                             rb=Rbar_galpy,
                             Af=Af_bar,
                             tform=t_form,
                             tsteady=t_steady,
                             barphi=bar_angle_rad
                            )
    
else:
    
    # For a N-component bar, with each component having a different m, use Wilma's GeneralDehnenBarPotential:
    raise Exception("Error: Wilma's GeneralDehnenBarPotential has not yet been merged into galpy. "+
                    "So far, only a single-component quadrupole bar can be used.")
    
    if not isinstance(slope_p,numpy.ndarray) and len(slope_p) == len(Fourier_m): 
        raise Exception("Error: slope_p needs to be an array of len(Fourier_m).")
    if not isinstance(Af,numpy.ndarray) and len(Af) == len(Fourier_m): 
        raise Exception("Error: Af_bar needs to be an array of len(Fourier_m).")   
    bar = []
    for ii in range(len(Fourier_m)):
        bar.extend([GeneralDehnenBarPotential(omegab=bar_omega_galpy,
                                      m=Fourier_m[ii],p=slope_p[ii],
                                      rb=Rbar_galpy,Af=Af_bar[ii],
                                      tform=t_form,tsteady=t_steady,barphi=bar_angle_rad)])

# Add bar to axisymmetric potential:
pot_bar = pot_axi + bar

#==============================
# Simulation parameters
#==============================
                      
# Input for axisymmetric disk:
input_filename = '../data/test_particle_simulation/initial_axisymmetric_mockdata_for_test_particle_simulation_EXAMPLE.fits'
name_input = 'EXAMPLE'

# Name of model and test/version number:
test_number = 'v0'
test_name = name_input + '_' + name_speed + '_' + name_shape
print('Name of this test particle simulation: ',test_name)

# Output for perturbed disk:
_action_angles           = True # Do we want to calculate action-angle-frequency estimates?
_fundamental_frequencies = True # Do we want to calculate "true" orbital frequencies from a fast Fourier transformation?
output_filename = '../data/test_particle_simulation/output_barred_test_particle_simulation_' + test_name + '_' + test_number + '.fits'

#==========================
# Read axisymmetric mock data
#==========================
print("* read-in axisymmetric mock data * ")
hdul = fits.open(input_filename)
tbdata = hdul[1].data

R_kpc_start = tbdata.field('R_kpc_start')
z_kpc_start = tbdata.field('z_kpc_start')
phi_rad_start = tbdata.field('phi_rad_start')
vR_kms_start = tbdata.field('vR_kms_start')
vz_kms_start = tbdata.field('vz_kms_start')
vT_kms_start = tbdata.field('vT_kms_start')

hdul.close()

_NSTARS = len(R_kpc_start)

#==========================
# Prepare simulation
#==========================

#_____prepare output times_____

# bar period in units of Gyr:
period_Gyr = 2.*math.pi/bar_omega_galpy*bovy_conversion.time_in_Gyr(galpy_scale_velocity, galpy_scale_length)

# number of orbit output steps:
n_steps       = 5001

# integration duration in units of bar periods:
start_period  = t_form
steady_period = t_steady
stop_period   = t_final
ts_period     = numpy.linspace(start_period,stop_period,n_steps)

# integration duration in units of Gyr:
start_Gyr     = start_period*period_Gyr
stop_Gyr      = stop_period *period_Gyr

# integration output steps in galpy units:
steady_galpy = steady_period*period_Gyr                  /bovy_conversion.time_in_Gyr(galpy_scale_velocity, galpy_scale_length)
ts_galpy     = numpy.linspace(start_Gyr,stop_Gyr,n_steps)/bovy_conversion.time_in_Gyr(galpy_scale_velocity, galpy_scale_length)

# which output steps correspond to the times we want to save to file:
index_output_times = numpy.isin(ts_period,t_out_period,assume_unique=True)

# at which output times is the bar steady?
index_steady = (ts_galpy >= steady_galpy)


#_____output field names_____
field_list = []

string_array_temp = numpy.array(['R_kpc_T','phi_rad_T','z_kpc_T','x_kpc_T','y_kpc_T',
                            'vR_kms_T','vT_kms_T','vz_kms_T'
                           ],dtype=str)
for string_temp in string_array_temp:
    for tx_str in output_times_names:
        field_list.append(string_temp+tx_str)
        
if _action_angles:
    string_array_temp = numpy.array(['JR_kpckms_T','Lz_kpckms_T','Jz_kpckms_T',
                                   'OmegaR_kmskpc_T','OmegaT_kmskpc_T','Omegaz_kmskpc_T',
                                   'wR_rad_T','wT_rad_T','wz_rad_T'
                                    ],dtype=str)
    for string_temp in string_array_temp:
        for tx_str in output_times_names:
            field_list.append(string_temp+tx_str)
        
if _fundamental_frequencies:
    for string_temp in numpy.array(['fund_OmegaR_kmskpc','fund_OmegaT_kmskpc','fund_Omegaz_kmskpc']):
        field_list.append(string_temp)

field_array = numpy.array(field_list,dtype=str)
n_fields_end = len(field_array)
print("Number of output fields: n_fields_end = ",n_fields_end)
print("Names of output fields: field_array = ",field_array)

#_____Continue with a existing simulation_____
my_file = Path(output_filename)
if (not _CONTINUE) or (not my_file.exists()):   
    
     i_start_0 = 0
     out_end = numpy.zeros((_NSTARS,n_fields_end)) + numpy.nan
    
else:
        
    print("* Read-in existing simulation and continue *")

    # read existing output:
    hdul_end = fits.open(output_filename)
    tbdata_end = hdul_end[1].data
    hdul_end.close()

    # initialize empty array and fill with existing output:
    out_end = numpy.zeros((_NSTARS,n_fields_end)) + numpy.nan
    for kk in range(n_fields_end):
        out_end[:,kk]= tbdata_end.field(field_array[kk])

    # Sum the occurences of NaN *after* each element.
    # If all of them are NaN, use this element as new starting index.
    sum_of_nans     = numpy.cumsum(numpy.isnan(out_end[:,0])[::-1])[::-1]
    sum_of_elements = numpy.cumsum(numpy.ones_like(out_end[:,0]))[::-1]
    i_start_0       = numpy.where(sum_of_nans == sum_of_elements)[0][0]


#_____prepare iteration indices_____
if (_MULTI is None) or (_MULTI < 2):
    pass
elif _MULTI > 0:
    number_of_stars_per_save     = _MULTI*10
    number_of_intermediate_saves = int(numpy.floor(float(_NSTARS)/float(number_of_stars_per_save)))
    i_start = numpy.arange(i_start_0,_NSTARS,number_of_stars_per_save,dtype=int)
    i_end   = i_start + number_of_stars_per_save
    i_end[-1] = _NSTARS
    
#==========================
# Function to integrate orbit for one star
#==========================

def integrate_orbit_prepare_output(pot_bar,\
                                   R_kpc_start,vR_kms_start,vT_kms_start,z_kpc_start,vz_kms_start,phi_rad_start):
    
    #********** ORBIT INTEGRATION **********

    #initialize orbit:
    R    = R_kpc_start /galpy_scale_length
    vR   = vR_kms_start/galpy_scale_velocity
    vT   = vT_kms_start/galpy_scale_velocity
    z    = z_kpc_start /galpy_scale_length
    vz   = vz_kms_start/galpy_scale_velocity
    phi  = phi_rad_start
    vxvv = [R,vR,vT,z,vz,phi]

    #integrate orbit in bar potential:
    o = Orbit(vxvv=vxvv)
    o.integrate(ts_galpy,pot_bar,method='rk4_c')
    
    #spatial coordinates:
    R_kpc_p   = o.R(ts_galpy)*galpy_scale_length
    z_kpc_p   = o.z(ts_galpy)*galpy_scale_length
    phi_rad_p = o.phi(ts_galpy)
    cos_phi_rad_p = numpy.cos(phi_rad_p)
    x_kpc_p   = o.x(ts_galpy)*galpy_scale_length
    y_kpc_p   = o.y(ts_galpy)*galpy_scale_length
    
    #velocities:
    vR_kms_p  = o.vR(ts_galpy)*galpy_scale_velocity
    vT_kms_p  = o.vT(ts_galpy)*galpy_scale_velocity
    vz_kms_p  = o.vz(ts_galpy)*galpy_scale_velocity
    
    #********** CALCULATE ACTIONS **********
    
    if _action_angles:
    
        JR_kpckms_p,Lz_kpckms_p,Jz_kpckms_p,\
        OmegaR_kmskpc_p,OmegaT_kmskpc_p,Omegaz_kmskpc_p,\
        wR_rad_p,wT_rad_p,wz_rad_p \
            = calculate_actions_frequencies_angles(
                aAS_axi,
                galpy_scale_length,galpy_scale_velocity,
                R_kpc_p,z_kpc_p,phi_rad_p,vR_kms_p,vz_kms_p,vT_kms_p,
                quiet=True
                )

    #********** OUTPUT: INTEGRATION END POINTS **********
    
    R_kpc_end = R_kpc_p[index_output_times]
    phi_rad_end = phi_rad_p[index_output_times]
    z_kpc_end = z_kpc_p[index_output_times]
    x_kpc_end = x_kpc_p[index_output_times]
    y_kpc_end = y_kpc_p[index_output_times]
    
    
    vR_kms_end = vR_kms_p[index_output_times]
    vT_kms_end = vT_kms_p[index_output_times]
    vz_kms_end = vz_kms_p[index_output_times]
    JR_kpckms_end = JR_kpckms_p[index_output_times]
    Lz_kpckms_end = Lz_kpckms_p[index_output_times]
    Jz_kpckms_end = Jz_kpckms_p[index_output_times]
    OmegaR_kmskpc_end = OmegaR_kmskpc_p[index_output_times]
    OmegaT_kmskpc_end = OmegaT_kmskpc_p[index_output_times]
    Omegaz_kmskpc_end = Omegaz_kmskpc_p[index_output_times]
    wR_rad_end = wR_rad_p[index_output_times]
    wT_rad_end = wT_rad_p[index_output_times]
    wz_rad_end = wz_rad_p[index_output_times]
       
    
    #********** OUTPUT: REAL FREQUENCIES **********
    
    if _fundamental_frequencies:
        
        #Orbit in steady bar potential only:
        R_kpc_s       = R_kpc_p[index_steady]
        z_kpc_s       = z_kpc_p[index_steady]
        phi_rad_s     = phi_rad_p[index_steady]
        cos_phi_rad_s = cos_phi_rad_p[index_steady]
    
        #Fast Fourier Transform:
        fourier_R   = fft(R_kpc_s)
        fourier_phi = fft(cos_phi_rad_s)
        fourier_z   = fft(z_kpc_s)    
        n  = R_kpc_s.size
        dt = ts_galpy[1]-ts_galpy[0] # time step [galpy units]    
        freq = fftfreq(n,d=dt)       # Fourier Transform sample frequencies
        start = 1                    # don't take frequency = 0
        if   n%2 == 0: end = n//2-1  # take only positive frequencies
        elif n%2 == 1: end = (n-1)//2
        om_kmskpc = 2.*math.pi*freq[start:end]*bovy_conversion.freq_in_kmskpc(galpy_scale_velocity, galpy_scale_length)
        amp_R_kpc = numpy.abs(fourier_R  [start:end])/n
        amp_phi   = numpy.abs(fourier_phi[start:end])/n
        amp_z_kpc = numpy.abs(fourier_z  [start:end])/n

        #pick fundamental frequencies:
        fund_OmegaR_kmskpc = om_kmskpc[numpy.argmax(amp_R_kpc)]
        fund_OmegaT_kmskpc = om_kmskpc[numpy.argmax(amp_phi)]
        fund_Omegaz_kmskpc = om_kmskpc[numpy.argmax(amp_z_kpc)]
    
    
    #********** RETURN  RESULTS ***********
    
    out = numpy.hstack((R_kpc_end,phi_rad_end,z_kpc_end, \
                        x_kpc_end,y_kpc_end, \
                        vR_kms_end,vT_kms_end,vz_kms_end))
    if _action_angles:
        temp = numpy.hstack((JR_kpckms_end,Lz_kpckms_end,Jz_kpckms_end, \
               OmegaR_kmskpc_end,OmegaT_kmskpc_end,Omegaz_kmskpc_end, \
               wR_rad_end,wT_rad_end,wz_rad_end))
        out = numpy.hstack((out,temp))
    if _fundamental_frequencies:
        out = numpy.hstack((out,[fund_OmegaR_kmskpc,fund_OmegaT_kmskpc,fund_Omegaz_kmskpc]))
    return out
    
#==========================
# Integrate orbits
#==========================

print("* Integrate orbits *")
                                       
if (_MULTI is None) or (_MULTI < 2):
    
    print('Integrate now star...')                               
    for ii in numpy.arange(i_start_0,_NSTARS,1):
        print('no. ',ii)
        
        #_____Calculate on single core:_____
        out_end[ii,:] = integrate_orbit_prepare_output(
                pot_bar,\
                R_kpc_start[ii],vR_kms_start[ii],vT_kms_start[ii],z_kpc_start[ii],vz_kms_start[ii],phi_rad_start[ii]
                )
        
        #_____Save to file:_____
        if (ii+1) % 100 == 0:
            t = Table(out_end, names=field_array)
            t.write(output_filename, format='fits', overwrite=True)
        
elif _MULTI > 1:

    for ii in range(len(i_start)):
        print('Integrate now current batch ',ii,'/',len(i_start),'\t\t; in total done: ',i_start[ii],'/',_NSTARS,' stars')
        sys.stdout.flush()
        ii_s = numpy.arange(i_start[ii],i_end[ii],1,dtype=int)
        
        #_____Calculate on multiple cores:_____
        out = multi.parallel_map(
            lambda x: integrate_orbit_prepare_output(
                pot_bar,\
                R_kpc_start[x],vR_kms_start[x],vT_kms_start[x],z_kpc_start[x],vz_kms_start[x],phi_rad_start[x]),
            ii_s,
            numcores=numpy.amin([multiprocessing.cpu_count(),_MULTI])
            )

        for x in range(len(ii_s)):
                out_end[ii_s[x],:] = out[x]
                
        #_____Save to file:_____
        t = Table(out_end, names=field_array)
        t.write(output_filename, format='fits', overwrite=True)
        
# Save to file one last time:
t = Table(out_end, names=field_array)
t.write(output_filename, format='fits', overwrite=True)
        
print("#####################################")
print("##### THE CALCULATIONS ARE DONE! ####")
print("#####################################")