
# A Nipype version of doing seed-based connectivity analysis.
# Adapted from CPAC

import sys
import os
import re
import commands

from nipype.interfaces import afni as afni
from   nipype.interfaces.afni import preprocess
import nipype.pipeline.engine as pe
import nipype.algorithms.rapidart as ra
import nipype.interfaces.afni  as afni
import nipype.interfaces.fsl   as fsl
import nipype.interfaces.io    as nio
import nipype.interfaces.utility as util



ROI_ANALYSIS = True





def set_gauss(fwhm):
    """ Set Gaussian smoothing op-string for fslmaths """
    fwhm = float(fwhm)
    sigma = float(fwhm / 2.3548)
    op = "-kernel gauss %f -fmean -mas" % (sigma) # + "%s"
    # Converting FWHM into Sigma, see https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=fsl;54608a1b.1111
    op_string = op
    return op_string




def create_sca(name_sca,
               smooth_fwhm):

    """
    Map of the correlations of the Region of Interest(Seed in native or MNI space) with the rest of brain voxels.
    The map is normalized to contain Z-scores, mapped in standard space and treated with spatial smoothing.

    Arguments
    name : name to be given to the workflow
    smooth_fwhm : smoothing kernel width (mm) for smoothing the final connectivity maps


    Notes
    -----
    Workflow Inputs::
 
        inputspec.rest_res_filt : string (existing nifti file)
            Band passed Image with Global Signal , white matter, csf and motion regression. Recommended bandpass filter (0.001,0.1) )

        inputspec.timeseries_one_d : string (existing nifti file)
            1D 3dTcorr1D compatible timeseries file. 1D file can be timeseries from a mask or from a parcellation containing ROIs

        
    Workflow Outputs::

        outputspec.correlation_file : string (nifti file)
            Correlations of the functional file and the input time series 

        outputspec.correlation_ztransf : string (nifti file)
            Fisher Z transformed correlations of the seed 

        outputspec.correlation_ztransf_smooth : string (nifti file)
            Smoothed version of the fisher Z correlations

    """

    sca = pe.Workflow(name=name_sca)
    inputNode = pe.Node(util.IdentityInterface(fields=[
        'timeseries_one_d',
        'functional_file',
    ]),
                        name='inputspec')
    
    
    outputNode = pe.Node(util.IdentityInterface(fields=[
        'correlation_file',
        'correlation_ztransf',
        'correlation_ztransf_smooth',
    ]),
                         name='outputnode')


    # # 2. Compute voxel-wise correlation with Seed Timeseries
    corr = pe.Node(interface=preprocess.TCorr1D(),
                      name='3dTCorr1D')
    corr.inputs.pearson = True
    corr.inputs.outputtype = 'NIFTI_GZ'

    sca.connect(inputNode, 'timeseries_one_d',
                corr, 'y_1d')
    sca.connect(inputNode, 'functional_file',
                corr, 'xset')
    sca.connect(corr, 'out_file',
                outputNode, 'functional_file')


    # Fisher-z transform the correlation map
    # 3dcalc -a subjX_corrmap_r.nii -expr 'log((1+a)/(1-a))/2' -prefix subjX_corrmap_z.nii
    ztransf = pe.Node(interface=afni.Calc(),
                      name='fisherz')
    ztransf.inputs.expr='log((1+a)/(1-a))/2'
    ztransf.inputs.out_file = 'corr_ztransf.nii.gz'
    ztransf.inputs.outputtype="NIFTI"

    sca.connect(corr,   'out_file',ztransf,   'in_file_a')
    sca.connect(ztransf,'out_file',outputNode,'correlation_ztransf')


    # Spatial smoothing
    #smooth = pe.Node(interface=fsl.ImageMaths(),
    #                    name='smooth')
    smooth = pe.Node(interface=fsl.IsotropicSmooth(),
                     name='smooth')

    smooth.inputs.fwhm = smooth_fwhm
    
    sca.connect(ztransf,'out_file',smooth,    'in_file')
    sca.connect(smooth, 'out_file',outputNode,'correlation_ztransf_smooth')
    
    
    return sca







