
from nipype.pipeline.engine import Node, MapNode, Workflow
import nipype.interfaces.utility    as util
from nipype.interfaces.utility import Function


import nipype.interfaces.utility    as util
import nipype.interfaces.io         as nio
import nipype.interfaces.fsl        as fsl
import nipype.interfaces.afni       as afni
import nipype.interfaces.ants       as ants
import nipype.interfaces.freesurfer as fs

import nipype.algorithms.rapidart as ra
#import nipype.algorithms.misc     as misc
import nipype.algorithms.confounds as confounds


from nipype.utils.filemanip import list_to_filename

from nipype.interfaces.fsl import Merge
from nipype.interfaces.fsl import TOPUP
import os



# Just a little hack to be able to use fslmerge on the opposite-phase encoded images
def combinef(func_AP,func_PA):
    #print(func_AP)
    #print(func_PA)
    return [func_AP,func_PA]



def create_sdc_workflow():

    """

    A lay-person-made workflow for performing Susceptibility Distortion Correction (SDC) 
    on EPI (fMRI) data. This processes field map data acquired using the opposite-phase encode method.

    This follows the procedure outlined here: https://lcni.uoregon.edu/kb-articles/kb-0003
    
    Inputs:
    inputnode.func_AP  - The AP-encoded EPI image
    inputnode.func_PA  - The PA-encoded EPI image
    inputnode.encoding_file - A file specifying the encoding parameters (see link above)
    
    Outputs:
    outputnode.func_AP_PA - The merged AP-PA EPI image
    outputnode.topup_out_corrected - The distortion-corrected AP & PA images
    outputnode.topup_out_field - The field map (raw as it came from topup)
    outputnode.topup_out_fieldcoef - The field coefficients (from topup)
    outputnode.topup_out_logfile - The topup log file
    outputnode.fieldmap_rads - The fieldmap (in radians/s)
    outputnode.fieldmap_mag - The fieldmap magnitude image
    outputnode.fieldmap_mag_brain - The brain-extracted fieldmap magnitude image
    """

    
    
    sdcwf = Workflow(name='suscept_distort_corr')
    inputnode = Node(util.IdentityInterface(fields=['func_AP',     # AP-encoded EPI image
                                                    'func_PA',     # PA-encoded EPI image
                                                    'encoding_file'  # the file specifying the phase encodings and readout times of the EPIs
    ]),
                     name='inputnode')
    
    outputnode = Node(util.IdentityInterface(fields=['func_AP_PA',  
                                                     'topup_out_field',
                                                     'topup_out_fieldcoef',
                                                     'topup_out_logfile',
                                                     'topup_out_corrected',
                                                     'fieldmap_rads',
                                                     'fieldmap_mag',
                                                     'fieldmap_mag_brain']),
                      name='outputnode')

    # This follows the workflow described here:
    # https://lcni.uoregon.edu/kb-articles/kb-0003

    # IMPORTANT WARNING: Until FSL release 5.0.9, topup up did not write any orientation information into the outputted nifti header files. The consequences of this omission range from minor to disastrous. Be sure you are using the latest release of FSL, or fix the orientation information in the nifti files after the fact.

    
    combine = Node(name='list_ap_pa',interface=Function(input_names=['func_AP','func_PA'],
                                                        output_names=["func_AP_PA"],
                                                        function=combinef))


    # Now each of the AP and PA may contain multiple volumes.
    # Question is what to do with these, because we can use only a single volume
    # when computing the field map.
    # For now, let's take the time-average and use that as single-volume input to topup,
    # hoping to increase SNR perhaps.
    timeavg = MapNode(interface=fsl.MeanImage(dimension="T"),name='time_avg',iterfield=['in_file'])
    

    # 1. Use fslmerge to combine the two se-epi images into one 4D file: fslmerge -t [output] [inputAP] [inputPA].
    # So you end up with a single nifti where the first "timepoint" is the AP encoded image and the second "timepoint" is the PA encoded image.

    
    merger = Node(Merge(),
                  name='merge_ap_pa')
    merger.inputs.output_type="NIFTI_GZ"
    merger.inputs.dimension = "t"

    sdcwf.connect([ (inputnode, combine,     [('func_AP',       'func_AP'),
                                              ('func_PA',       'func_PA')]),
                    (combine,   timeavg,     [('func_AP_PA',    'in_file')]),
                    (timeavg,   merger,      [('out_file',      'in_files')]),
                    (merger,    outputnode,  [('merged_file',   'func_AP_PA')]) ])

    # 2. Create the "datain" file (see https://lcni.uoregon.edu/kb-articles/kb-0003)
    # Assumed to have been done once and for all (should be the same for all our EPIs)
    
    # 3. Run topup. You can use the default config file b02b0.cnf (you do not need to create this file). The command will look something like this:
    # topup --imain=se_epi_merged --datain=datain.txt --config=b02b0.cnf --fout=my_fieldmap --iout=se_epi_unwarped
    topup = Node(TOPUP(output_type = "NIFTI_GZ"),
                 name='topup')

    # So you're gonna need an encoding_file here, for details see:
    # https://lcni.uoregon.edu/kb-articles/kb-0003
    # TO calculate the total readout time, you can use the formula below:
    # The total readout time is given by the (number of echoes minus one) * echo spacing.
    
    sdcwf.connect([ (inputnode,   topup,       [('encoding_file',   'encoding_file')]),
                    (merger,      topup,       [('merged_file',     'in_file')]),
                    (topup,       outputnode,  [('out_corrected',   'topup_out_corrected'),
                                                ('out_field',       'topup_out_field'),
                                                ('out_fieldcoef',   'topup_out_fieldcoef'),
                                                ('out_logfile',     'topup_out_logfile')])
                    ])
                    
    
    # 4. The fieldmap will be in units of Hz, but you'll need it in units of rad/s for use with fugue or feat, so multiply by 2pi:
    # fslmaths my_fieldmap -mul 6.28 my_fieldmap_rads
    multiply = Node(fsl.ImageMaths(op_string= '-mul 6.28'),name='hz2rad')
    sdcwf.connect([ (topup,       multiply,   [('out_field',        'in_file')]) ])
                    
    # 5. Finally, if you will be using this with Feat, you'll need a single magnitude image, and a brain extracted version of that image. You can get that this way:
    # fslmaths se_epi_unwarped -Tmean my_fieldmap_mag   # this takes the mean across "time" (the two unwarped images)
    tomag = Node(fsl.MeanImage(dimension="T"),name='tomag')
    sdcwf.connect([ (topup,       tomag,      [('out_corrected',     'in_file')]) ])

    # Brain-extract the fieldmap image
    # bet2 my_fieldmap_mag my_fieldmap_mag_brain
    bet2 = Node(fsl.BET(),name='bet')
    sdcwf.connect([ (tomag,       bet2,       [('out_file',             'in_file')]) ])
    
    
    sdcwf.connect([ (multiply,    outputnode, [('out_file',             'fieldmap_rads')]) ])
    sdcwf.connect([ (tomag,       outputnode, [('out_file',             'fieldmap_mag')]) ])
    sdcwf.connect([ (bet2,        outputnode, [('out_file',             'fieldmap_mag_brain')]) ])
    return sdcwf







# The original images contained more than one volume...
# To select just the last volume (hopefully least magnetically-challenged)
# 3dTcat 20170209_103555SMSTOPUPAPs004a001.nii.gz[$] -prefix TOPUPAP.nii.gz


master = Workflow(name='wrapper')

sdcwf = create_sdc_workflow()

sdcwf.inputs.inputnode.func_AP = os.path.abspath('20170209_103555SMSTOPUPAPs004a001.nii.gz')
sdcwf.inputs.inputnode.func_PA = os.path.abspath('20170209_103555SMSTOPUPPAs005a001.nii.gz')
sdcwf.inputs.inputnode.encoding_file = os.path.abspath('example_enc_file.txt')


datasink = Node(nio.DataSink(), name='sinker')
datasink.inputs.base_directory = os.path.abspath('./sdcwf_output')

master.connect([ (sdcwf, datasink, [ ('outputnode.func_AP_PA',          'func_AP_AP'),
                                     ('outputnode.topup_out_corrected', 'topup_out_corrected'),
                                     ('outputnode.topup_out_field',     'topup_out_field'),
                                     ('outputnode.topup_out_fieldcoef', 'topup_out_fieldcoef'),
                                     ('outputnode.topup_out_logfile',   'topup_out_logfile'),
                                     ('outputnode.fieldmap_rads',       'fieldmap_rads'),
                                     ('outputnode.fieldmap_mag',        'fieldmap_mag'),
                                     ('outputnode.fieldmap_mag_brain',  'fieldmap_mag_brain') ]) ])

#sdcwf.run()
master.write_graph(dotfilename='sdcwf',
                  graph2use='colored', format='pdf', simple_form=True)

master.base_dir = "sdcwf_work"

master.run()
