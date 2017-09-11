
from nipype.pipeline import engine as pe
from nipype.interfaces import afni, ants, fsl, utility as niu
import nipype.interfaces.io as nio
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.afni as afni
from nipype.interfaces.base import traits
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import BaseInterface
from nipype.algorithms.confounds import is_outlier
import numpy as np

 
# Based heavily on
# https://github.com/poldracklab/fmriprep/blob/master/fmriprep/workflows/bold.py


def get_encoding(nii):
    """ Get phase encoding for a nii file."""
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




# Here I'm trying to build by hand a double-blip unwarping of EPIs.




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

from nipype.interfaces.base import BaseInterfaceInputSpec
#from nipype.interfaces.base import BaseInterfaceOutputSpec

from nipype.interfaces.base import File, TraitedSpec
#(
#    File, BaseInterfaceInputSpec, TraitedSpec, traits
#)

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


    
    
def init_enhance_and_skullstrip_bold_wf(name='enhance_and_skullstrip_bold_wf',
                                        omp_nthreads=1):
    """
    This workflow takes in a BOLD volume, and attempts to enhance the contrast
    between gray and white matter, and skull-stripping the result.
    .. workflow ::
        :graph2use: orig
        :simple_form: yes
        from fmriprep.workflows.util import init_enhance_and_skullstrip_bold_wf
        wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=1)
    Inputs
        in_file
            BOLD image (single volume)
    Outputs
        bias_corrected_file
            the ``in_file`` after `N4BiasFieldCorrection`_
        skull_stripped_file
            the ``bias_corrected_file`` after skull-stripping
        mask_file
            mask of the skull-stripped input file
        out_report
            reportlet for the skull-stripping
    .. _N4BiasFieldCorrection: https://hdl.handle.net/10380/3053
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['mask_file',
                                                       'skull_stripped_file',
                                                       'bias_corrected_file',
                                                       'out_report']),
                         name='outputnode')
    n4_correct = pe.Node(
        ants.N4BiasFieldCorrection(dimension=3,
                                   #copy_header=True,
                                   num_threads=omp_nthreads),
        name='n4_correct', n_procs=omp_nthreads)
    skullstrip_first_pass = pe.Node(fsl.BET(frac=0.2, mask=True),
                                    name='skullstrip_first_pass')
    unifize = pe.Node(afni.Unifize(t2=True, outputtype='NIFTI_GZ',
                                   args='-clfrac 0.4',
                                   out_file="uni.nii.gz"), name='unifize')
    skullstrip_second_pass = pe.Node(afni.Automask(dilate=1,
                                                   outputtype='NIFTI_GZ'),
                                     name='skullstrip_second_pass')
    combine_masks = pe.Node(fsl.BinaryMaths(operation='mul'),
                            name='combine_masks')
    apply_mask = pe.Node(fsl.ApplyMask(),
                         name='apply_mask')
    #mask_reportlet = pe.Node(SimpleShowMaskRPT(), name='mask_reportlet')

    workflow.connect([
        (inputnode, n4_correct, [('in_file', 'input_image')]),
        (n4_correct, skullstrip_first_pass, [('output_image', 'in_file')]),
        (skullstrip_first_pass, unifize, [('out_file', 'in_file')]),
        (unifize, skullstrip_second_pass, [('out_file', 'in_file')]),
        (skullstrip_first_pass, combine_masks, [('mask_file', 'in_file')]),
        (skullstrip_second_pass, combine_masks, [('out_file', 'operand_file')]),
        (unifize, apply_mask, [('out_file', 'in_file')]),
        (combine_masks, apply_mask, [('out_file', 'mask_file')]),
        #(n4_correct, mask_reportlet, [('output_image', 'background_file')]),
        #(combine_masks, mask_reportlet, [('out_file', 'mask_file')]),
        (combine_masks, outputnode, [('out_file', 'mask_file')]),
        #(mask_reportlet, outputnode, [('out_report', 'out_report')]),
        (apply_mask, outputnode, [('out_file', 'skull_stripped_file')]),
        (n4_correct, outputnode, [('output_image', 'bias_corrected_file')]),
        ])

    return workflow




def init_prepare_epi_wf(omp_nthreads, name="prepare_epi_wf"):
    """
    This workflow takes in a set of EPI files with with the same phase
    encoding direction and returns a single 3D volume ready to be used in
    field distortion estimation.

    The procedure involves: estimating a robust template using FreeSurfer's
    'mri_robust_template', bias field correction using ANTs N4BiasFieldCorrection
    and AFNI 3dUnifize, skullstripping using FSL BET and AFNI 3dAutomask,
    and rigid coregistration to the reference using ANTs.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.fieldmap.unwarp import init_prepare_epi_wf
        wf = init_prepare_epi_wf(omp_nthreads=8)


    Inputs

        fmaps
            list of 3D or 4D NIfTI images
        ref_brain
            coregistration reference (skullstripped and bias field corrected)

    Outputs

        out_file
            single 3D NIfTI file

    """
    inputnode = pe.Node(niu.IdentityInterface(fields=['fmaps', 'ref_brain']),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']),
                         name='outputnode')

    split = pe.MapNode(fsl.Split(dimension='t'), iterfield='in_file',
                       name='split')

    merge = pe.Node(
        StructuralReference(auto_detect_sensitivity=True,
                            initial_timepoint=1,
                            fixed_timepoint=True,  # Align to first image
                            intensity_scaling=True,
                            # 7-DOF (rigid + intensity)
                            no_iteration=True,
                            subsample_threshold=200,
                            out_file='template.nii.gz'),
        name='merge')

    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf(
        omp_nthreads=omp_nthreads)

    #ants_settings = pkgr.resource_filename('fmriprep',
    #                                       'data/translation_rigid.json')
    
    fmap2ref_reg = pe.Node(ants.Registration(from_file="ANTS_translation_rigid.json",
                                             output_warped_image=True,
                                             num_threads=omp_nthreads),
                           name='fmap2ref_reg', n_procs=omp_nthreads)

    workflow = pe.Workflow(name=name)

    def _flatten(l):
        from nipype.utils.filemanip import filename_to_list
        return [item for sublist in l for item in filename_to_list(sublist)]

    workflow.connect([
        (inputnode, split, [('fmaps', 'in_file')]),
        (split, merge, [(('out_files', _flatten), 'in_files')]),
        (merge, enhance_and_skullstrip_bold_wf, [('out_file', 'inputnode.in_file')]),
        (enhance_and_skullstrip_bold_wf, fmap2ref_reg, [
            ('outputnode.skull_stripped_file', 'moving_image')]),
        (inputnode, fmap2ref_reg, [('ref_brain', 'fixed_image')]),
        (fmap2ref_reg, outputnode, [('warped_image', 'out_file')]),
    ])

    return workflow



# Helper functions
# ------------------------------------------------------------


def _fix_hdr(in_file):
    import nibabel as nb
    import os

    nii = nb.load(in_file)
    hdr = nii.header.copy()
    hdr.set_data_dtype('<f4')
    hdr.set_intent('vector', (), '')

    out_file = os.path.abspath("warpfield.nii.gz")

    nb.Nifti1Image(nii.get_data().astype('<f4'), nii.affine, hdr).to_filename(out_file)

    return out_file


def _get_ec(in_dict):
    return float(in_dict['EffectiveEchoSpacing'])


def _get_pedir_bids(in_dict):
    return in_dict['PhaseEncodingDirection']


def _get_pedir_fugue(in_dict):
    return in_dict['PhaseEncodingDirection'].replace('i', 'x').replace('j', 'y').replace('k', 'z')


def _hz2rads(in_file, out_file=None):
    """Transform a fieldmap in Hz into rad/s"""
    import os
    from math import pi
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix
    if out_file is None:
        out_file = fname_presuffix(in_file, suffix='_rads',
                                   newpath=os.getcwd())
    nii = nb.load(in_file)
    data = nii.get_data() * 2.0 * pi
    nb.Nifti1Image(data, nii.get_affine(),
                   nii.get_header()).to_filename(out_file)
    return out_file


def _demean(in_file, in_mask, out_file=None):
    import os
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    if out_file is None:
        out_file = fname_presuffix(in_file, suffix='_demeaned',
                                   newpath=os.getcwd())
    nii = nb.load(in_file)
    msk = nb.load(in_mask).get_data()
    data = nii.get_data()
    data -= np.median(data[msk > 0])
    nb.Nifti1Image(data, nii.affine, nii.header).to_filename(
        out_file)
    return out_file


def _fill_with_ones(in_file):
    import nibabel as nb
    import numpy as np
    import os

    nii = nb.load(in_file)
    data = np.ones(nii.shape)

    out_name = os.path.abspath("out.nii.gz")
    nb.Nifti1Image(data, nii.affine, nii.header).to_filename(out_name)

    return out_name







def init_pepolar_unwarp_wf(fmaps_matching_pe,
                           fmaps_opposite_pe,
                           bold_file,
                           bold_file_pe,
                           omp_nthreads,
                           name="pepolar_unwarp_wf"):
    """
    This workflow takes in a set of EPI files with opposite phase encoding
    direction than the target file and calculates a displacements field
    (in other words, an ANTs-compatible warp file).

    The warp field correcting for the distortions is estimated using AFNI's
    3dQwarp, with displacement estimation limited to the target file phase
    encoding direction.

    It also calculates a new mask for the input dataset that takes into
    account the distortions.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.fieldmap.unwarp import init_pepolar_unwarp_wf
        wf = init_pepolar_unwarp_wf(fmaps=['/dataset/sub-01/fmap/sub-01_epi.nii.gz'],
                                    fmaps_pes=['j-'],
                                    bold_file='/dataset/sub-01/func/sub-01_task-rest_bold.nii.gz',
                                    bold_file_pe='j',
                                    omp_nthreads=8)


    Inputs

        in_reference
            the reference image  (this is an enhanced EPI image, enhanced contrast and uniform-ized etc.)
        in_reference_brain
            the reference image skullstripped
        in_mask
            a brain mask corresponding to ``in_reference``

    Outputs

        out_reference
            the ``in_reference`` after unwarping
        out_reference_brain
            the ``in_reference`` after unwarping and skullstripping
        out_warp
            the corresponding :abbr:`DFM (displacements field map)` compatible with
            ANTs
        out_mask
            mask of the unwarped input file
        out_mask_report
            reportlet for the skullstripping

        epi_matching_out_file 
            the preprocessed "fieldmap" epi with matching phase encoding (used for entry into QWarp) -- so not "corrected"
        epi_opposite_out_file 
            the preprocessed "fieldmap" epi with matching phase encoding (used for entry into QWarp) -- so not "corrected"

    """

    args = '-noXdis -noYdis -noZdis'
    rm_arg = {'i': '-noXdis',
              'j': '-noYdis',
              'k': '-noZdis'}[bold_file_pe] #bold_file_pe[0]]
    args = args.replace(rm_arg, '')


    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_reference', 'in_reference_brain', 'in_mask']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_reference', 'out_reference_brain', 'out_warp', 'out_mask',
                'epi_opposite_out_file','epi_matching_out_file',
                'out_mask_report']),
        name='outputnode')

    prepare_epi_opposite_wf = init_prepare_epi_wf(omp_nthreads=omp_nthreads,
                                                  name="prepare_epi_opposite_wf")
    prepare_epi_opposite_wf.inputs.inputnode.fmaps = fmaps_opposite_pe

    qwarp = pe.Node(afni.QwarpPlusMinus(pblur=[0.05, 0.05],
                                        blur=[-1, -1],
                                        noweight=True,
                                        minpatch=9,
                                        nopadWARP=True,
                                        environ={'OMP_NUM_THREADS': '%d' % omp_nthreads},
                                        args=args),
                    name='qwarp', n_procs=omp_nthreads)

    workflow.connect([
        (inputnode,                prepare_epi_opposite_wf,   [('in_reference_brain', 'inputnode.ref_brain')]),
        (prepare_epi_opposite_wf,  qwarp,                     [('outputnode.out_file', 'base_file')]),
        (prepare_epi_opposite_wf,  outputnode,                [('outputnode.out_file', 'epi_opposite_out_file')])
    ])

    prepare_epi_matching_wf = init_prepare_epi_wf(omp_nthreads=omp_nthreads,
                                                  name="prepare_epi_matching_wf")
    prepare_epi_matching_wf.inputs.inputnode.fmaps = fmaps_matching_pe
    
    workflow.connect([
        (inputnode,               prepare_epi_matching_wf, [('in_reference_brain', 'inputnode.ref_brain')]),
        (prepare_epi_matching_wf, qwarp,                   [('outputnode.out_file', 'source_file')]),
        (prepare_epi_matching_wf, outputnode,              [('outputnode.out_file', 'epi_matching_out_file')])
    ])

    to_ants = pe.Node(niu.Function(function=_fix_hdr), name='to_ants',
                      mem_gb=0.01)

    cphdr_warp = pe.Node(CopyHeader(), name='cphdr_warp',
                         mem_gb=0.01, run_without_submitting=True)

    unwarp_reference = pe.Node(ants.resampling.ApplyTransforms(dimension=3,
                                                               #generate_report=False,
                                                               float=True,
                                                               interpolation='LanczosWindowedSinc'),
                               name='unwarp_reference')

    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)

    workflow.connect([
        (inputnode,          cphdr_warp, [('in_reference', 'hdr_file')]),
        (qwarp,              cphdr_warp, [('source_warp', 'in_file')]),
        (cphdr_warp,         to_ants, [('out_file', 'in_file')]),
        (to_ants,            unwarp_reference, [('out', 'transforms')]),
        (inputnode,          unwarp_reference, [('in_reference', 'reference_image'),
                                                ('in_reference', 'input_image')]),
        (unwarp_reference,   enhance_and_skullstrip_bold_wf, [
            ('output_image', 'inputnode.in_file')]),
        (unwarp_reference,   outputnode, [('output_image', 'out_reference')]),
        (enhance_and_skullstrip_bold_wf, outputnode, [
            ('outputnode.mask_file', 'out_mask'),
            ('outputnode.out_report', 'out_mask_report'),
            ('outputnode.skull_stripped_file', 'out_reference_brain')]),
        (to_ants, outputnode, [('out', 'out_warp')]),
    ])

    return workflow






def init_bold_reference_wf(omp_nthreads, bold_file=None, name='bold_reference_wf'):
    """
    This workflow generates reference BOLD images for a series
    The raw reference image is the target of :abbr:`HMC (head motion correction)`, and a
    contrast-enhanced reference is the subject of distortion correction, as well as
    boundary-based registration to T1w and template spaces.
    .. workflow::
        :graph2use: orig
        :simple_form: yes
        from fmriprep.workflows.bold import init_bold_reference_wf
        wf = init_bold_reference_wf(omp_nthreads=1)
    Parameters
        bold_file : str
            BOLD series NIfTI file
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``bold_reference_wf``)
    Inputs
        bold_file
            BOLD series NIfTI file
    Outputs
        bold_file
            Validated BOLD series NIfTI file
        raw_ref_image
            Reference image to which BOLD series is motion corrected
        skip_vols
            Number of non-steady-state volumes detected at beginning of ``bold_file``
        ref_image
            Contrast-enhanced reference image
        ref_image_brain
            Skull-stripped reference image
        bold_mask
            Skull-stripping mask of reference image
        bold_mask_report
            Reportlet showing quality of masking
        validation_report
            HTML reportlet indicating whether ``bold_file`` had a valid affine
    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_file', 'raw_ref_image', 'skip_vols', 'ref_image',
                                      'ref_image_brain', 'bold_mask', 'bold_mask_report',
                                      'validation_report']),
        name='outputnode')

    # Simplify manually setting input image
    #if bold_file is not None:
    #    inputnode.inputs.bold_file = bold_file

    validate = pe.Node(ValidateImage(), name='validate', mem_gb=DEFAULT_MEMORY_MIN_GB,
                       run_without_submitting=True)
    gen_ref = pe.Node(EstimateReferenceImage(), name="gen_ref",
                      mem_gb=1)  # OE: 128x128x128x50 * 64 / 8 ~ 900MB.
    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)

    workflow.connect([
        (inputnode, validate, [('bold_file', 'in_file')]),
        (validate, gen_ref, [('out_file', 'in_file')]),
        (gen_ref, enhance_and_skullstrip_bold_wf, [('ref_image', 'inputnode.in_file')]),
        (validate, outputnode, [('out_file', 'bold_file'),
                                ('out_report', 'validation_report')]),
        (gen_ref, outputnode, [('ref_image', 'raw_ref_image'),
                               ('n_volumes_to_discard', 'skip_vols')]),
        (enhance_and_skullstrip_bold_wf, outputnode, [
            ('outputnode.bias_corrected_file', 'ref_image'),
            ('outputnode.mask_file', 'bold_mask'),
            ('outputnode.out_report', 'bold_mask_report'),
            ('outputnode.skull_stripped_file', 'ref_image_brain')]),
        ])

    return workflow






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



DEFAULT_MEMORY_MIN_GB = .01



if True:
    import os

    # Let's test this!
    # for subject s05
    anat = "s05_T1_brain.nii.gz" # output from Freesurfer so intensity normalised (normally)

    bold_file         = os.path.abspath("s05_pre_BOLD.nii.gz") # the actual EPI of interest
    fmap_matching_pe  = os.path.abspath("s05_pre_AP.nii.gz") # the matching-encoding field map
    fmap_opposite_pe  = os.path.abspath("s05_pre_PA.nii.gz") # the opposite-encoding field map


    import nipype.interfaces.nipy as nipy ## at first required that we run in py3 though, for some weird reason that I don't understand...

    # Let's do basic moco -- wow, this actually takes half an hour or so on Colossus!
    slicemoco = pe.Node(nipy.SpaceTimeRealigner(),                 
                     name="spacetime_realign")
    #slicemoco.inputs.slice_info = 2
    #slicemoco.inputs.tr = 1.05
    slicemoco.inputs.in_file = [bold_file]
    #slicemoco.inputs = "bold_moco.nii.gz"
    #slicemoco.run()
    
    realigned_bold = slicemoco.outputs.out_file

    # outputs already saved under corr_s05_pre_BOLD.nii.gz and .par
    realigned_bold = os.path.abspath('corr_s05_pre_BOLD.nii.gz')

    # I then took the first few volumes so that I can test the pipeline
    # 3dTcat -prefix corr_s05_pre_BOLD_firstfew.nii.gz corr_s05_pre_BOLD.nii.gz[0..10]
    realigned_bold = os.path.abspath('corr_s05_pre_BOLD_firstfew.nii.gz')

    omp_nthreads = 4



    # This will create a reference image from the BOLD sequence
    # I guess this is some kind of enhanced sample image or aggregated image so that
    # we can nicely align other EPIs to it.
    bold_ref_wf = init_bold_reference_wf(omp_nthreads)
    bold_ref_wf.inputs.inputnode.bold_file = realigned_bold
    bold_ref_wf.base_dir = "work"
    #bold_ref_wf.run()
    



    
    #for epi in epi_fmaps:
    #    import nibabel as nib
    #    hdr = nib.load(epi).header
    #    print(epi,hdr.get('descrip')) # will show the phase encoding
    
    # https://github.com/poldracklab/fmriprep/blob/7ed29d6e75de951df43107b192155094742197f8/fmriprep/workflows/bold.py#L372
    #if fmap['type'] == 'epi':
    #epi_fmaps = [] #fmap_['epi'] for fmap_ in fmaps if fmap_['type'] == 'epi']
    sdc_unwarp_wf = init_pepolar_unwarp_wf(fmaps_matching_pe = fmap_matching_pe,
                                           fmaps_opposite_pe = fmap_opposite_pe,
                                           bold_file         = realigned_bold,
                                           bold_file_pe      = "j",
                                           omp_nthreads      = 2,
                                           name              = 'pepolar_unwarp_wf')


    sdc_unwarp_wf.inputs.inputnode.in_reference       = os.path.abspath('uni.nii.gz')
    sdc_unwarp_wf.inputs.inputnode.in_reference_brain = os.path.abspath('uni_masked.nii.gz')
    sdc_unwarp_wf.inputs.inputnode.in_mask            = os.path.abspath('ref_image_corrected_brain_mask_maths.nii.gz')
    
    
    #workflow.connect([
    #    (inputnode, sdc_unwarp_wf, [('bold_file', 'inputnode.name_source')]),
    #    (bold_reference_wf, sdc_unwarp_wf, [
    #        ('outputnode.ref_image', 'inputnode.in_reference'),
    #        ('outputnode.ref_image_brain', 'inputnode.in_reference_brain'),
    #        ('outputnode.bold_mask', 'inputnode.in_mask')]),
    #    (sdc_unwarp_wf, bold_reg_wf, [
    #        ('outputnode.out_warp', 'inputnode.fieldwarp'),
    #        ('outputnode.out_reference_brain', 'inputnode.ref_bold_brain'),
    #        ('outputnode.out_mask', 'inputnode.ref_bold_mask')]),
    #    (sdc_unwarp_wf, func_reports_wf, [
    #        ('outputnode.out_mask_report', 'inputnode.bold_mask_report')])
    #])

    #sdc_unwarp_wf.write_
    #sdc_unwarp_wf.write_graph(dotfilename='sdc_unwarp_wf', graph2use='colored', format='pdf', simple_form=True)

    sink = pe.Node(nio.DataSink(), name='output')
    #sink.inputs.container      = seed # create a subdirectory for each seed
    sink.inputs.base_directory = os.path.abspath("output")
    
    workflow = pe.Workflow(name='master')
    workflow.base_dir = "work"
    workflow.connect( sdc_unwarp_wf, "outputnode.out_reference",        sink, "out_reference" )
    workflow.connect( sdc_unwarp_wf, "outputnode.out_reference_brain",  sink, "out_reference_brain" )
    workflow.connect( sdc_unwarp_wf, "outputnode.out_warp",             sink, "out_warp" )
    workflow.connect( sdc_unwarp_wf, "outputnode.out_mask",             sink, "out_mask" )
    workflow.connect( sdc_unwarp_wf, "outputnode.epi_matching_out_file",sink, "out_epi_matching" )
    workflow.connect( sdc_unwarp_wf, "outputnode.epi_opposite_out_file",sink, "out_epi_opposite" )


    #
    # Now apply the unwarping we got from the sdc_unwarp_wf script to unwarp the original BOLD
    #
    #    unwarp_bold = pe.Node(ants.resampling.ApplyTransforms(dimension=3,
    #                                                      #generate_report=False,
    #                                                      float=True,
    #                                                      interpolation='LanczosWindowedSinc'),
    #                      name='unwarp_reference')
    #enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)
    #unwarp_bold.inputs.in_reference = 
    #workflow.connect( sdc_unwarp_wf, "outputnode.out_warp", unwarp_bold, "transforms")

    workflow.write_graph(dotfilename='sdc_unwarp_wf', graph2use='colored', format='pdf', simple_form=True)
    workflow.run()
    


    overlays = [("subject anat","T1.nii.gz")]
    # TODO -- compare how the original, opposite-phase and "corrected" images match the anatomical.
    for (filetype,f) in [('matching_uncorr',      'output/out_epi_matching/transform_Warped.nii.gz'), # non-transformed matching PE EPI
                         ('opposte_uncorr',       'output/out_epi_opposite/transform_Warped.nii.gz'), # non-transformed opposite PE EPI
                         ('reference_uncorr',     'uni_masked.nii.gz'),
                         ('reference_corr',       'output/out_reference_brain/uni_masked.nii.gz')]:

        bbregister = pe.Node(fs.BBRegister(contrast_type='t2',
                                        out_fsl_file='func2anat.mat',
                                        out_reg_file='func2anat.dat',
                                        registered_file='func2anat.nii.gz',
                                        init='fsl',
                                        epi_mask=True),
                          name='bbregister')
        bbregister.inputs.source_file = os.path.abspath(f)
        bbregister.inputs.subjects_dir = "/brains/audiomotor.mrs.nipype/processing/work/freesurfer"
        bbregister.inputs.subject_id = "s05"
        targetdir= 'bbregister.%s'%filetype
        if not os.path.exists(targetdir):
            os.makedirs(targetdir)
        bbregister.base_dir = targetdir

        bbregister.run()

        overlays.append(("%s2anat"%filetype,
                         "bbregister.%s/bbregister/func2anat.nii.gz"%filetype))
    

    cmd = 'fsleyes /usr/share/fsl/5.0/data/standard/MNI152_T1_1mm.nii.gz '
    for (lbl,ovl) in overlays:
        cmd+="%s --name '%s' "%(ovl,lbl)

    print(cmd)
