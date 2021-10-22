print("RUNNING NOW: perform_RoadMapping_analysis_on_mockdata_EXAMPLE_py3.py")
print("CODE VERSION: 2021-October-20")

#==============================
# Import packages
#==============================

# python:
import numpy
import os
import sys
import time

# RoadMapping:
sys.path.append('/u/twilm/research/RoadMapping/py')
from adapt_fitting_range import adapt_fitting_range
from analyze_mockdata_RoadMapping import analyze_mockdata_RoadMapping
from estimate_initial_parameters import estimate_initial_df_parameters
from outlier_model import scale_df_phys_to_fit
from write_RoadMapping_parameters import write_RoadMapping_parameters

#==============================
# Model & analysis parameters
#==============================

#_____multiprocessing_____
_MULTI = 40     #number of cores to run the analysis on

#_____potential_____
_POTTYPE   = 71 # Potential type parameter
#                 Miyamoto-Nagai disk, NFW halo, (fixed) Hernquist bulge 
#                 using Staeckel-Grid actions (!!!)

R0_kpc      = 8.    # Solar radius
vc_kms      = 240.  # Circular velocity at the Solar radius
a_disk_kpc  = 2.5   # stellar disk scale length
b_disk_kpc  = 0.3   # stellar disk scale height
f_halo      = 0.3   # halo contribution to the disk's v_c^2
a_halo_kpc  = 18.   # dark matter halo scale length

# Known true potential parameters of the mock data:
potParTrue_phys = numpy.array([R0_kpc,vc_kms,a_disk_kpc,b_disk_kpc,f_halo,a_halo_kpc])            
potParFitBool   = numpy.array([False ,True  ,True      ,True      ,False ,False     ],dtype=bool) 
# denotes which parameters will be fitted (True) or kept fixed (False)

#_____distribution function_____
_DFTYPE  = 0  #quasiisothermal df (qDF) (Binney & McMillan 2011)

hr_kpc  = 2.5 # radial scale length
sr0_kms = 33. # radial velocity dispersion at the solar radius
sz0_kms = 25. # vertical velocity dispersion at the solar radius
hsr_kpc = 8.  # radial-velocity-dispersion scale length
hsz_kpc = 7.  # vertial-velocity-dispersion scale length

# Known true DF parameters of the mock data:
dfParTrue_phys = numpy.array([hr_kpc,sr0_kms,sz0_kms,hsr_kpc,hsz_kpc])            
dfParFitBool   = numpy.array([True  ,True   ,True   ,False  ,False   ],dtype=bool) 
# denotes which parameters will be fitted (True) or kept fixed (False)

#_____selection function_____
_SFTYPE = 1                          #wedge-shaped survey volume
Rmin_kpc,   Rmax_kpc = 5.,10.        # minimum and maximum galactocentric radius [kpc]
zmin_kpc,   zmax_kpc = -1.,1.        # minimum and maximum height above the plane [kpc]
phimin_deg, phimax_deg = -180., 180. # minimum and maximum azimuth angle [degrees]
sfPar_phys = numpy.array([Rmin_kpc,Rmax_kpc,zmin_kpc,zmax_kpc,phimin_deg,phimax_deg])

#_____data_____
_DATATYPE   = 1                       # perfect mock data
datasetname = 'mock_data_exampleB_py3' # name of the data set
_NSTARS     = 500                     # number of stars in the data set
mockdatapath = '../data/'             # path where the data lives

#_____RoadMapping analysis_____
testname      = 'test0'            # name of this specific RoadMapping analysis
_METHOD       = 'GRID_and_MCMC'    # method of analysis
_N_IT_TOT     = 10                 # max. number of iterations of grid analysis
_noMCMCsteps  = 200
_noMCMCburnin = 100

#_____priors_____
_PRIORTYPE = 0 # flat priors on potential and log(DF) parameters.

#==============================
# Write parameters to file to prepare the analysis
#==============================

#analysis parameter input file:
analysis_input_filename = mockdatapath+"/"+datasetname+"/"+datasetname+"_"+testname+"_analysis_parameters.txt"

if os.path.exists(analysis_input_filename) and _METHOD in ['GRID_and_MCMC','GRID']:
    # This part of the code is useful to have if your GRID analysis crashed earlier (e.g. through time limit of cluster). 
    # You don't need to redo the whole GRID fit then. Just update the analysis file.
    
    print("* Input parameter file exists already. Continue GRID analysis with existing parameters. *")
    

    #_____updated analysis parameter file_____
    potParFitNo = numpy.ones_like(potParFitBool,dtype=int)
    dfParFitNo  = numpy.ones_like(dfParFitBool,dtype=int)
    potParFitNo[potParFitBool] = 3
    dfParFitNo [dfParFitBool]  = 3
    write_RoadMapping_parameters(
            datasetname,
            testname        = testname,
            update          = True,
            mockdatapath    = mockdatapath,
            potParFitNo     = potParFitNo,
            dfParFitNo      = dfParFitNo
            )

elif _METHOD in ['GRID_and_MCMC','GRID']:
    
    print("* Input parameter file does not exist yet - create it now. *")

    #_____estimate model parameters_____
    # Here, we initialise the analysis with "estimated" parameters being slightly off from the true ones.
    # Parameters that are not fitted in the analysis are set to their known true values.
    potParEst_phys = 0.9 * potParTrue_phys 
    dfParEst_phys  = 1.1 *  dfParTrue_phys
    potParEst_phys[~potParFitBool] = potParTrue_phys[~potParFitBool]
    dfParEst_phys [~ dfParFitBool] = dfParTrue_phys [~ dfParFitBool]
    
    #_____initial potential parameter boundaries for fit_____
    potParMin_phys    = 0.8 * potParEst_phys
    potParMax_phys    = 1.2 * potParEst_phys
    potParMin_phys[~potParFitBool] = potParEst_phys[~potParFitBool]
    potParMax_phys[~potParFitBool] = potParEst_phys[~potParFitBool]
    
    #_____initial df parameter boundaries for fit_____
    # Note: scale_df_phys_to_fit() makes sure the fit is in log space.
    dfParEst_fit  = scale_df_phys_to_fit(_DFTYPE,dfParEst_phys)
    dfParMin_fit  = scale_df_phys_to_fit(_DFTYPE,0.8 * dfParEst_phys)
    dfParMax_fit  = scale_df_phys_to_fit(_DFTYPE,1.2 * dfParEst_phys)
    dfParMin_fit[~dfParFitBool] = dfParEst_fit[~dfParFitBool]
    dfParMax_fit[~dfParFitBool] = dfParEst_fit[~dfParFitBool]

    #_____write all analysis parameters to file_____
    write_RoadMapping_parameters(
                datasetname,
                testname        = testname,
                datatype        = _DATATYPE,
                pottype         = _POTTYPE,
                sftype          = _SFTYPE,
                dftype          = _DFTYPE,
                priortype       = _PRIORTYPE,
                noStars         = _NSTARS,
                potParTrue_phys = potParTrue_phys,
                potParEst_phys  = potParEst_phys,
                potParMin_phys  = potParMin_phys,
                potParMax_phys  = potParMax_phys,
                potParFitBool   = potParFitBool,
                dfParTrue_phys  = dfParTrue_phys,
                dfParEst_phys   = dfParEst_phys,
                dfParMin_fit    = dfParMin_fit,
                dfParMax_fit    = dfParMax_fit,
                dfParFitBool    = dfParFitBool,
                sfParTrue_phys  = sfPar_phys,
                mockdatapath    = mockdatapath,
                N_spatial       = 20,  # Numerical precision of Mtot (i.e. likelihood normalisation): spatial N-point quadrature
                N_velocity      = 28,  # Numerical precision of Mtot (i.e. likelihood normalisation): velocity N-point quadrature
                N_sigma         = 5.5, # Numerical precision of Mtot (i.e. likelihood normalisation): vR and vz integration range
                vT_galpy_max    = 1.5,  # Numerical precision of Mtot (i.e. likelihood normalisation): vT integration range
                MCMC_use_fidDF  = True, # Use the best fit "fiducial" DF from the GRID search to set the velocity integration range for calculating Mtot during the MCMC.
                noMCMCsteps     = _noMCMCsteps,
                noMCMCburnin    = _noMCMCburnin,
                aASG_accuracy   = [5.,70.,40.,50.], # Numerical accuracy for Staeckel-Grid action calculation from galpy
                use_default_Delta = True,           # use Delta = 0.45 for the underlying Staeckel potential in the Staeckel Fudge
                )

#=======================================================================

print("* Start RoadMapping analysis *")

time_start = time.time()

#===============================
#=====1. NESTED GRID SEARCH=====
#===============================

if (_METHOD == 'GRID_and_MCMC' or _METHOD == 'GRID' or _METHOD == 'only_GRID'):

    #____first analysis_____
    
    # This is where "THE MAGIC HAPPENS", i.e., where the actual likelihood is calculated 
    # on multiple cores and based on the analysis parameter file:
    
    analyze_mockdata_RoadMapping(datasetname,testname=testname,multicores=_MULTI,mockdatapath=mockdatapath,method='GRID')

    #_____iterate the nested-grid search until optimal fitting range is found ..._____
    
    # ... or until a maximum of _N_IT_TOT times is reached. 
    # As long as we're not yet sitting centered on the likelihood peak
    # (with our grid also having a similar width as the likelihood peak),
    # we calculate the likelihood on a 3^N grid (where N = number of free parameters), for speed reasons.
    # Only if _METHOD == 'only_GRID', the last iteration will calculate the likelihood on a fine grid with 
    # n_gridpoints_final^N points.
    
    fine_grid = 0
    n_it = 0
    while fine_grid == 0:
        n_it += 1
        if n_it < _N_IT_TOT:
            force_fine_grid = False
        else:
            force_fine_grid = True

        #_____adapt fitting range_____
        
        # This function fits Gaussians to the 3^N likelihood grid to determine 
        # a better fitting range for the next of the nested grid searches.
        # It decides if the nested grid has converged and overwrites the 
        # analysis parameter file with the updated grid location.
        
        fine_grid = adapt_fitting_range(
                        datasetname,
                        testname=testname,
                        n_sigma_range=3.,
                        n_gridpoints_final=12,
                        mockdatapath=mockdatapath,
                        force_fine_grid=force_fine_grid
                        )
        print("FINE GRID:",fine_grid)

        #____next iteration step of analysis_____
        if (_METHOD == 'only_GRID') or (fine_grid == 0):
            
            print(n_it,"th iteration")
            
            # More "MAGIC" is happening:
            analyze_mockdata_RoadMapping(datasetname,testname=testname,multicores=_MULTI,mockdatapath=mockdatapath,method='GRID')


#================================
#=====2. MCMC EXPLORATION========
#================================

# The MCMC will start at the likelihood peak that we found with the grid search.
# This step is only needed to get the likelihood (or actually the posterior distribution pdf) in a smoothly sampled format, 
# and not just in the still sparse grid from the previous step.

if _METHOD == 'GRID_and_MCMC' or _METHOD == 'MCMC':
    
    # The final piece of "MAGIC" is happening:
    analyze_mockdata_RoadMapping(datasetname,testname=testname,multicores=_MULTI,mockdatapath=mockdatapath,method='MCMC')

print("TOTAL TIME TAKEN FOR ANALYSIS: ",(time.time() - time_start)/60.," minutes")
