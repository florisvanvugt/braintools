# Just adding these includes here because otherwise you will only
# see that you don't have a module when running the functions
import nitime.fmri.io as io
import nibabel as nib
from nipype.utils.filemanip import fname_presuffix


from nipype.interfaces.base import BaseInterface
from nipype.interfaces.base import BaseInterfaceInputSpec
from nipype.interfaces.base import File, TraitedSpec
from nipype.interfaces.base import traits
from nipype.interfaces.base import CommandLineInputSpec
from nipype.interfaces.base import CommandLine

    
import nipype.interfaces.freesurfer as fs

from nipype.utils.filemanip import fname_presuffix
from nipype.algorithms.confounds import is_outlier
from nipype.utils.filemanip import list_to_filename


import os
import subprocess
import numpy as np


def strip_rois_func(in_file, t_min):
    '''
    Removing intial volumes from a time series
    '''
    import numpy as np
    import nibabel as nb
    import os
    from nipype.utils.filemanip import split_filename
    nii = nb.load(in_file)
    new_nii = nb.Nifti1Image(nii.get_data()[:,:,:,t_min:], nii.affine, nii.header)
    new_nii.set_data_dtype(np.float32)
    _, base, _ = split_filename(in_file)
    nb.save(new_nii, base + "_roi.nii.gz")
    return os.path.abspath(base + "_roi.nii.gz")





def time_normalizer(in_file, tr):
    '''
    Mean centering and variance normalizing a time series
    '''
    import os
    import nitime.fmri.io as io
    import nibabel as nib
    from nipype.utils.filemanip import fname_presuffix
    
    T= io.time_series_from_file(str(in_file),normalize='zscore', TR=tr) # FVV added str() because the function doesn't take unicode file names
    normalized_data = T.data
    
    out_img = nib.Nifti1Image(normalized_data,nib.load(in_file).affine)
    out_file = fname_presuffix(in_file, suffix='_norm',newpath=os.getcwd())
    out_img.to_filename(out_file)
    
    return out_file








def build_filter1(motion_params, outliers, comp_norm=None, detrend_poly=None):
    ## THIS IS THE ONE FROM THE LEMON DATASET
    """From https://github.com/nipy/nipype/blob/master/examples/
    rsfmri_vol_surface_preprocessing_nipy.py#L261
    
    Builds a regressor set comprisong motion parameters, composite norm and
    outliers. The outliers are added as a single time point column for each outlier
    
    Parameters
    ----------
    motion_params: a text file containing motion parameters and its derivatives
    comp_norm: a text file containing the composite norm
    outliers: a text file containing 0-based outlier indices
    detrend_poly: number of polynomials to add to detrend
    
    Returns
    -------
    components_file: a text file containing all the regressors
    """
    
    from nipype.utils.filemanip import filename_to_list
    import numpy as np
    import os
    from scipy.special import legendre
    
    out_files = []
    for idx, filename in enumerate(filename_to_list(motion_params)):
        params = np.genfromtxt(filename)
        if comp_norm:
            norm_val = np.genfromtxt(filename_to_list(comp_norm)[idx])
            out_params = np.hstack((params, norm_val[:, None]))
        else:
            out_params = params
        try:
            outlier_val = np.genfromtxt(filename_to_list(outliers)[idx])
        except IOError:
            outlier_val = np.empty((0))
        for index in np.atleast_1d(outlier_val):
            outlier_vector = np.zeros((out_params.shape[0], 1))
            outlier_vector[int(index)] = 1   # FVV added int(index), otherwise problems appeared
            assert index==int(index)         # FVV added for good order
            out_params = np.hstack((out_params, outlier_vector))
        if detrend_poly:
            timepoints = out_params.shape[0]
            X = np.ones((timepoints, 1))
            for i in range(detrend_poly):
                X = np.hstack((X, legendre(
                    i + 1)(np.linspace(-1, 1, timepoints))[:, None]))
            out_params = np.hstack((out_params, X))
        filename = os.path.join(os.getcwd(), "mcart_regressor.txt") #"filter_regressor%02d.txt" % idx)
        np.savetxt(filename, out_params, fmt=str("%.10f"))
        out_files.append(filename)
    return out_files





def extract_noise_components(realigned_file, mask_file, num_components=5,
extra_regressors=None):
    """From https://github.com/nipy/nipype/blob/master/examples/
    rsfmri_vol_surface_preprocessing_nipy.py#L261
    
    Derive components most reflective of physiological noise according to 
    aCompCor method (Behzadi 2007)
      
    Parameters
    ----------
    realigned_file: a 4D Nifti file containing realigned volumes
    mask_file: a 3D Nifti file containing white matter + ventricular masks
    num_components: number of components to use for noise decomposition
    extra_regressors: additional regressors to add
    
    Returns
    -------
    components_file: a text file containing the noise components
    """
    
    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import filename_to_list
    import scipy.linalg as linalg
    import os
    
    imgseries = nb.load(realigned_file)
    components = None
    for filename in filename_to_list(mask_file):
        mask = nb.load(filename).get_data()
        if len(np.nonzero(mask > 0)[0]) == 0:
            continue
        voxel_timecourses = imgseries.get_data()[mask > 0]
        voxel_timecourses[np.isnan(np.sum(voxel_timecourses, axis=1)), :] = 0
        # remove mean and normalize by variance
        # voxel_timecourses.shape == [nvoxels, time]
        X = voxel_timecourses.T
        stdX = np.std(X, axis=0)
        stdX[stdX == 0] = 1.
        stdX[np.isnan(stdX)] = 1.
        stdX[np.isinf(stdX)] = 1.
        X = (X - np.mean(X, axis=0))/stdX
        u, _, _ = linalg.svd(X, full_matrices=False)
        if components is None:
            components = u[:, :num_components]
        else:
            components = np.hstack((components, u[:, :num_components]))
    if extra_regressors:
        regressors = np.genfromtxt(extra_regressors)
        components = np.hstack((components, regressors))
    components_file = os.path.join(os.getcwd(), 'noise_regressor.txt')
    np.savetxt(components_file, components, fmt=str("%.10f"))
    return components_file







def motion_regressors(motion_params, order=0, derivatives=1):
    """From https://github.com/nipy/nipype/blob/master/examples/
    rsfmri_vol_surface_preprocessing_nipy.py#L261

    Compute motion regressors upto given order and derivative
    motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic)
    """
    from nipype.utils.filemanip import filename_to_list
    import numpy as np
    import os
    
    out_files = []
    for idx, filename in enumerate(filename_to_list(motion_params)):
        params = np.genfromtxt(filename)
        out_params = params
        for d in range(1, derivatives + 1):
            cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0),
                                 params))
            out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))
        out_params2 = out_params
        for i in range(2, order + 1):
            out_params2 = np.hstack((out_params2, np.power(out_params, i)))
        filename = os.path.join(os.getcwd(), "motion_regressor_der%d_ord%d.txt" % (derivatives, order))
        np.savetxt(filename, out_params2, fmt=str("%.10f"))
        out_files.append(filename)
    return out_files







def selectindex(files, idx):
    import numpy as np
    from nipype.utils.filemanip import filename_to_list, list_to_filename
    return list_to_filename(np.array(filename_to_list(files))[idx].tolist())




def median_timeseries(in_files):
    """Computes an average of the median of each realigned timeseries
    Parameters
    ----------
    in_files: one or more realigned Nifti 4D time series
    Returns
    -------
    out_file: a 3D Nifti file
    """
    import nibabel as nb
    import numpy as np
    import os
    from nipype.utils.filemanip import filename_to_list
    from nipype.utils.filemanip import split_filename
    
    average = None
    for idx, filename in enumerate(filename_to_list(in_files)):
        img = nb.load(filename)
        data = np.median(img.get_data(), axis=3)
        if average is None:
            average = data
        else:
            average = average + data
    median_img = nb.Nifti1Image(average/float(idx + 1),
                                img.affine, img.header)
    #filename = os.path.join(os.getcwd(), 'median.nii.gz')
    #median_img.to_filename(filename)
    _, base, _ = split_filename(filename_to_list(in_files)[0])
    nb.save(median_img, base + "_median.nii.gz")
    return os.path.abspath(base + "_median.nii.gz")
    return filename



def get_info(dicom_file):
    """Given a Siemens dicom file return metadata
    Returns
    -------
    RepetitionTime
    Slice Acquisition Times
    Spacing between slices
    """
    from dcmstack.extract import default_extractor
    import numpy as np
    from dicom import read_file
    from nipype.utils.filemanip import filename_to_list
    
    meta = default_extractor(read_file(filename_to_list(dicom_file)[0],
                                       stop_before_pixels=True,
                                       force=True))
    
    TR=meta['RepetitionTime']/1000.
    slice_times_pre=meta['CsaImage.MosaicRefAcqTimes']
    slice_times = (np.array(slice_times_pre)/1000.).tolist()
    slice_thickness = meta['SpacingBetweenSlices']
    
    return TR, slice_times, slice_thickness



def fix_hdr_func(data_file, header_file):
    '''
    Overwrites the header of data_file with the header of header_file
    USE WITH CAUTION
    '''
    
    import nibabel as nb
    import os
    from nipype.utils.filemanip import split_filename
    
    data=nb.load(data_file).get_data()
    hdr=nb.load(header_file).header
    affine=nb.load(header_file).affine
    
    new_file=nb.Nifti1Image(data, affine, hdr)
    _, base, _ = split_filename(data_file)
    nb.save(new_file, base + "_fixed.nii.gz")
    return os.path.abspath(base + "_fixed.nii.gz")





def get_encoding(nii):
    """ 
    Get phase encoding for a nii file. Works for FVV Audiomotor MRS dataset. Not tried on anything else.
    """
    import nibabel as nib
    import re
    img = nib.load(nii)
    hdr = img.header
    #pe = hdr.get("PhaseEncodingDirection") # doesn't work...
    #for epi in epi_fmaps:
    #hdr = nib.load(epi).header
    descrip = str(hdr.get('descrip'))
    fnd = re.search('phaseDir=([+-])',descrip)
    if fnd:
        return fnd.groups()[0]
    else:
        return None











#
# The interfaces below come from fmriprep
#
#




class StructuralReference(fs.RobustTemplate):
    """ Variation on RobustTemplate that simply copies the source if a single
    volume is provided.
    >>> from fmriprep.utils.bids import collect_data
    >>> t1w = collect_data('ds114', '01')[0]['t1w']
    >>> template = StructuralReference()
    >>> template.inputs.in_files = t1w
    >>> template.inputs.auto_detect_sensitivity = True
    >>> template.cmdline  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    'mri_robust_template --satit --mov .../sub-01_ses-retest_T1w.nii.gz
        .../sub-01_ses-test_T1w.nii.gz --template mri_robust_template_out.mgz'
    """
    @property
    def cmdline(self):
        import nibabel as nb
        cmd = super(StructuralReference, self).cmdline
        if len(self.inputs.in_files) > 1:
            return cmd

        img = nb.load(self.inputs.in_files[0])
        if len(img.shape) > 3 and img.shape[3] > 1:
            return cmd

        out_file = self._list_outputs()['out_file']
        copyfile(self.inputs.in_files[0], out_file)
        return "echo Only one time point!"


    
class SimpleInterface(BaseInterface):
    """ An interface pattern that allows outputs to be set in a dictionary """
    def __init__(self, **inputs):
        super(SimpleInterface, self).__init__(**inputs)
        self._results = {}

    def _list_outputs(self):
        return self._results


class CopyHeaderInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the file we get the data from')
    hdr_file = File(exists=True, mandatory=True, desc='the file we get the header from')


class CopyHeaderOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')



class CopyHeader(SimpleInterface):
    """
    Copy a header from the `hdr_file` to `out_file` with data drawn from
    `in_file`.
    """
    input_spec = CopyHeaderInputSpec
    output_spec = CopyHeaderOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        in_img = nb.load(self.inputs.hdr_file)
        out_img = nb.load(self.inputs.in_file)
        new_img = out_img.__class__(out_img.get_data(), in_img.affine, in_img.header)
        new_img.set_data_dtype(out_img.get_data_dtype())

        out_name = fname_presuffix(self.inputs.in_file,
                                   suffix='_fixhdr', newpath='.')
        new_img.to_filename(out_name)
        self._results['out_file'] = out_name
        return runtime





class ValidateImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input image')


class ValidateImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='validated image')
    out_report = File(exists=True, desc='HTML segment containing warning')


class ValidateImage(SimpleInterface):
    input_spec = ValidateImageInputSpec
    output_spec = ValidateImageOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        img = nb.load(self.inputs.in_file)
        out_report = os.path.abspath('report.html')

        qform_code = img.header._structarr['qform_code']
        sform_code = img.header._structarr['sform_code']

        # Valid affine information
        if (qform_code, sform_code) != (0, 0):
            self._results['out_file'] = self.inputs.in_file
            open(out_report, 'w').close()
            self._results['out_report'] = out_report
            return runtime

        out_fname = fname_presuffix(self.inputs.in_file, suffix='_valid', newpath=runtime.cwd)

        # Nibabel derives a default LAS affine from the shape and zooms
        # Use scanner xform code to indicate no alignment has been done
        img.set_sform(img.affine, nb.nifti1.xform_codes['scanner'])

        img.to_filename(out_fname)
        self._results['out_file'] = out_fname

        snippet = (r'<h3 class="elem-title">WARNING - Invalid header</h3>',
                   r'<p class="elem-desc">Input file does not have valid qform or sform matrix.',
                   r'A default, LAS-oriented affine has been constructed.',
                   r'A left-right flip may have occurred.',
                   r'Analyses of this dataset MAY BE INVALID.</p>')

        with open(out_report, 'w') as fobj:
            fobj.write('\n'.join('\t' * 3 + line for line in snippet))
            fobj.write('\n')

        self._results['out_report'] = out_report
        return runtime



class EstimateReferenceImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="4D EPI file")
    mc_method = traits.Enum("AFNI", "FSL", dsec="Which software to use to perform motion correction",
                            usedefault=True)


class EstimateReferenceImageOutputSpec(TraitedSpec):
    ref_image = File(exists=True, desc="3D reference image")
    n_volumes_to_discard = traits.Int(desc="Number of detected non-steady "
                                           "state volumes in the beginning of "
                                           "the input file")


    

class EstimateReferenceImage(SimpleInterface):
    """
    Given an 4D EPI file estimate an optimal reference image that could be later
    used for motion estimation and coregistration purposes. If detected uses
    T1 saturated volumes (non-steady state) otherwise a median of
    of a subset of motion corrected volumes is used.
    """
    input_spec = EstimateReferenceImageInputSpec
    output_spec = EstimateReferenceImageOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        in_nii = nb.load(self.inputs.in_file)
        data_slice = in_nii.dataobj[:, :, :, :50]
        global_signal = data_slice.mean(axis=0).mean(
            axis=0).mean(axis=0)

        # Slicing may induce inconsistencies with shape-dependent values in extensions.
        # For now, remove all. If this turns out to be a mistake, we can select extensions
        # that don't break pipeline stages.
        #in_nii.header.extensions.clear() # FVV removed this

        n_volumes_to_discard = is_outlier(global_signal)

        out_ref_fname = os.path.abspath("ref_image.nii.gz")

        if n_volumes_to_discard == 0:
            if in_nii.shape[-1] > 40:
                slice_fname = os.path.abspath("slice.nii.gz")
                nb.Nifti1Image(data_slice[:, :, :, 20:40], in_nii.affine,
                               in_nii.header).to_filename(slice_fname)
            else:
                slice_fname = self.inputs.in_file

            if self.inputs.mc_method == "AFNI":
                res = afni.Volreg(in_file=slice_fname, args='-Fourier -twopass',
                                  zpad=4, outputtype='NIFTI_GZ').run()
            elif self.inputs.mc_method == "FSL":
                res = fsl.MCFLIRT(in_file=slice_fname,
                                  ref_vol=0, interpolation='sinc').run()
            mc_slice_nii = nb.load(res.outputs.out_file)

            median_image_data = np.median(mc_slice_nii.get_data(), axis=3)
            nb.Nifti1Image(median_image_data, in_nii.affine,
                           in_nii.header).to_filename(out_ref_fname)
        else:
            median_image_data = np.median(
                data_slice[:, :, :, :n_volumes_to_discard], axis=3)
            nb.Nifti1Image(median_image_data, in_nii.affine,
                           in_nii.header).to_filename(out_ref_fname)

        self._results["ref_image"] = out_ref_fname
        self._results["n_volumes_to_discard"] = n_volumes_to_discard

        return runtime








class WarpInvertInputSpec(CommandLineInputSpec):
    in_file  = File(exists=True,  mandatory=True, desc='input warp field', argstr="'INV(%s)'",position=1)
    out_file = File(exists=False, mandatory=True, desc='output name for inverted warp field that will be created', argstr='-prefix %s',position=0)
    

class WarpInvertOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='inverse warp')


class WarpInvert(CommandLine):
    input_spec  = WarpInvertInputSpec
    output_spec = WarpInvertOutputSpec

    # `3dNwarpCat -prefix <outputfile> 'INV(...)'
    _cmd = '3dNwarpCat -overwrite'

    def _list_outputs(self):
        import os
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs
    
    #if __name__ == "__main__":
    #
    #warpinv = WarpInvert(in_file='Qwarp_PLUS_WARP.nii.gz',out_file='warpinv.nii.gz')
    #print(warpinv.cmdline)
    #warpinv.run()





    


# Helper functions
# ------------------------------------------------------------


def fix_hdr_for_ants(in_file):
    import nibabel as nb
    import os

    nii = nb.load(in_file)
    hdr = nii.header.copy()
    hdr.set_data_dtype('<f4')
    hdr.set_intent('vector', (), '')

    out_file = os.path.abspath("warpfield.nii.gz")

    nb.Nifti1Image(nii.get_data().astype('<f4'), nii.affine, hdr).to_filename(out_file)

    return out_file


#def _get_ec(in_dict):
#    return float(in_dict['EffectiveEchoSpacing'])


#def _get_pedir_bids(in_dict):
#    return in_dict['PhaseEncodingDirection']


#def _get_pedir_fugue(in_dict):
#    return in_dict['PhaseEncodingDirection'].replace('i', 'x').replace('j', 'y').replace('k', 'z')






def make_lut(colour,greyscale=.7):
    """ 
    Make a look-up table for an overlay, with a particular colour in mind.
    This is for FSL, and it basically makes a two-part colour scale, the first
    part of which is a greyscale colour scale, and the second is a constant
    colour (which will correspond to the colour of the overlay).

    Arguments
    colour : the colour that the overlay will get (a 3-item tuple of values in [0,1])
    greyscale : what the top of the greyscale will be. if this goes to 1.0 then the greyscale will go all the way to white.

    Colours have to be coded as three-tuple from 0 to 1 (float).
    
    """

    lut = "%!VEST-LUT\n"+\
          "%%BeginInstance\n"+\
          "<<\n"+\
          "/SavedInstanceClassName /ClassLUT\n"+\
          "/PseudoColorMinimum 0.00\n"+\
          "/PseudoColorMaximum 1.00\n"+\
          "/PseudoColorMinControl /Low\n"+\
          "/PseudoColorMaxControl /High\n"+\
          "/PseudoColormap [\n"

    # Now make 2x100 rows of colours; first grey
    # then the colour in question.
    for i in np.linspace(0,greyscale,100):
        lut += "<-color{%.05f,%.05f,%.05f}->\n"%(i,i,i)
    for _ in range(100):
        lut += "<-color{%.05f,%.05f,%.05f}->\n"%colour
    lut += "]\n"+\
           ">>\n"+\
           "\n"+\
           "%%EndInstance\n"+\
           "%%EOF\n"

    fname = '/tmp/lut.txt'
    open(fname,'w').write(lut)
    return fname




pj = os.path.join


def gather(filename,target,info,check_not_exist=None):
    """
    Gather a particular filename, i.e. copy it away from its
    embedded location somewhere deep in the results folder,
    and put it somewhere with a reasonable filename.

    if check_not_exist we will check whether the target file does not already
    exist, in which case check_not_exist is a list of file names
    that we have previously generated and the current target
    is not allowed to be among them.
    """
    global gather_history
    
    if not os.path.exists(info["gather_dir"]):
        os.makedirs(info["gather_dir"])

    if not os.path.exists(filename):
        print("## Error: file %s does not exist ##"%filename)
        assert False

    target = pj(info["gather_dir"],target)
    if check_not_exist!=None:
        if target in check_not_exist:
            #if os.path.exists(target):
            print("Error -- gather target %s already exists"%target)
            assert False
        
        
    print("%s -> %s"%(filename,target))
    
    cmd = ["cp",filename,target]
    subprocess.call(cmd)

    if check_not_exist!=None:
        check_not_exist.append(target)
    
    return target




def exists(info,item):
    """ Check whether a particular item exists and is not-None in the configuration dict. """
    return item in info and info[item] not in [None,"","None"]
    





