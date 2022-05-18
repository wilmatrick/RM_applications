# CODE VERSION: 2022-May-18

# import packages
import math
import numpy
import time
from galpy.util import bovy_conversion

def calculate_actions_frequencies_angles(aAS,galpy_scale_length,galpy_scale_velocity,
                                         R_kpc,z_kpc,phi_rad,vR_kms,vz_kms,vT_kms,quiet=False):
    
    start = time.time()
    
    #prepare empty data arrays:
    ndata = len(R_kpc)
    JR_kpckms = numpy.zeros_like(R_kpc) + numpy.nan
    Lz_kpckms = numpy.zeros_like(R_kpc) + numpy.nan
    Jz_kpckms = numpy.zeros_like(R_kpc) + numpy.nan
    wR_rad    = numpy.zeros_like(R_kpc) + numpy.nan
    wz_rad    = numpy.zeros_like(R_kpc) + numpy.nan
    wT_rad    = numpy.zeros_like(R_kpc) + numpy.nan
    OmegaR_kmskpc = numpy.zeros_like(R_kpc) + numpy.nan
    OmegaT_kmskpc = numpy.zeros_like(R_kpc) + numpy.nan
    Omegaz_kmskpc = numpy.zeros_like(R_kpc) + numpy.nan

    #prepare evaluation on one core:
    M = 50 # number of stars evaluated at once
    N = int(numpy.floor(float(ndata) / float(M))) #separate iteration steps
    K = ndata - N*M # number of stars not yet evaluated

    #start iteration over star batches:
    start_t = time.time()
    if N > 0:
        for x in range(N):

            jfa = aAS.actionsFreqsAngles(
                           R_kpc [x*M:(x+1)*M]/galpy_scale_length,
                           vR_kms[x*M:(x+1)*M]/galpy_scale_velocity,
                           vT_kms[x*M:(x+1)*M]/galpy_scale_velocity,
                           z_kpc [x*M:(x+1)*M]/galpy_scale_length,
                           vz_kms[x*M:(x+1)*M]/galpy_scale_velocity,
                           phi_rad[x*M:(x+1)*M])

            JR_kpckms[x*M:(x+1)*M] = jfa[0]*galpy_scale_length*galpy_scale_velocity
            Lz_kpckms[x*M:(x+1)*M] = jfa[1]*galpy_scale_length*galpy_scale_velocity
            Jz_kpckms[x*M:(x+1)*M] = jfa[2]*galpy_scale_length*galpy_scale_velocity
            OmegaR_kmskpc[x*M:(x+1)*M]  = jfa[3]*bovy_conversion.freq_in_kmskpc(galpy_scale_velocity, galpy_scale_length)
            OmegaT_kmskpc[x*M:(x+1)*M]  = jfa[4]*bovy_conversion.freq_in_kmskpc(galpy_scale_velocity, galpy_scale_length)
            Omegaz_kmskpc[x*M:(x+1)*M]  = jfa[5]*bovy_conversion.freq_in_kmskpc(galpy_scale_velocity, galpy_scale_length)
            wR_rad[x*M:(x+1)*M] = jfa[6]
            wT_rad[x*M:(x+1)*M] = jfa[7]
            wz_rad[x*M:(x+1)*M] = jfa[8]

            stop_t = time.time()
            if not quiet: print('Action batch ',x,'/',N,' needed ', round((stop_t-start_t),4),'sec for ',M,' stars')
            start_t = stop_t

    # now calculate the rest of the data:
    if K > 0:

        jfa = aAS.actionsFreqsAngles(
                       R_kpc [N*M::]/galpy_scale_length,
                       vR_kms[N*M::]/galpy_scale_velocity,
                       vT_kms[N*M::]/galpy_scale_velocity,
                       z_kpc [N*M::]/galpy_scale_length,
                       vz_kms[N*M::]/galpy_scale_velocity,
                       phi_rad[N*M::])

        JR_kpckms[N*M::] = jfa[0]*galpy_scale_length*galpy_scale_velocity
        Lz_kpckms[N*M::] = jfa[1]*galpy_scale_length*galpy_scale_velocity
        Jz_kpckms[N*M::] = jfa[2]*galpy_scale_length*galpy_scale_velocity
        OmegaR_kmskpc[N*M::] = jfa[3]*bovy_conversion.freq_in_kmskpc(galpy_scale_velocity, galpy_scale_length)
        OmegaT_kmskpc[N*M::] = jfa[4]*bovy_conversion.freq_in_kmskpc(galpy_scale_velocity, galpy_scale_length)
        Omegaz_kmskpc[N*M::] = jfa[5]*bovy_conversion.freq_in_kmskpc(galpy_scale_velocity, galpy_scale_length)
        wR_rad[N*M::] = jfa[6]
        wT_rad[N*M::] = jfa[7]
        wz_rad[N*M::] = jfa[8]
                            
    # Flag stars for which action calculation has failed:
    if numpy.any(JR_kpckms < 0.):
        if numpy.any(JR_kpckms/galpy_scale_length/galpy_scale_velocity == -9999.):
                            
            # Check for stars with standard galpy error flag and set to NaN:
            indices = numpy.argwhere(JR_kpckms/galpy_scale_length/galpy_scale_velocity == -9999.)
            for ii in indices:
                JR_kpckms[ii] = numpy.nan
                Lz_kpckms[ii] = numpy.nan
                Jz_kpckms[ii] = numpy.nan
                OmegaR_kmskpc[ii] = numpy.nan
                OmegaT_kmskpc[ii] = numpy.nan
                Omegaz_kmskpc[ii] = numpy.nan
                wR_rad[ii] = numpy.nan
                wT_rad[ii] = numpy.nan
                wz_rad[ii] = numpy.nan
            
        else:
            raise Exception('ERROR: There are stars with negative JRs! This should not happen! Check!')
                            
    # Rescale angles to lie within the range -pi and +pi:
    for wi in [wR_rad,wT_rad,wz_rad]:
        while numpy.any(wi >  math.pi): wi[wi >  math.pi] = wi[wi >  math.pi] - 2.*math.pi
        while numpy.any(wi < -math.pi): wi[wi < -math.pi] = wi[wi < -math.pi] + 2.*math.pi


    stop = time.time()
    if not quiet: print('Action calculation needed in total ', (stop-start)/60.,' min for ',ndata,' stars')
    
    return JR_kpckms,Lz_kpckms,Jz_kpckms,OmegaR_kmskpc,OmegaT_kmskpc,Omegaz_kmskpc,wR_rad,wT_rad,wz_rad
