import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import os
import pandas as pd

import numpy as np
import sys
import yaml

import aux_group
get_path = aux_group.get_path

pj = os.path.join


"""

This is trying to do a group analysis of data that comes from seed-based connectivity
analysis (SCA).

A lot of this is based on
https://github.com/NeuroanatomyAndConnectivity/pipelines/blob/master/src/mindwandering/resting_state_volume_analysis_group_level.py

For inspiration, also let's see the CPAC
https://github.com/FCP-INDI/C-PAC/blob/master/CPAC/group_analysis/group_analysis.py

Also, creating a Flame model also is shown here:
https://github.com/neurohackweek/NI_cloud_computing

"""



if len(sys.argv)<2:
    print("give the yaml config file as argument.")
    print("usage:")
    print("python run_groupanalysis.py <config.yml>")
    sys.exit(-1)
    
yamlf = sys.argv[1]
with open(yamlf,'r') as f:
    config = yaml.load(f)


# The configuration directory
configdir = os.path.dirname(yamlf)
    


modelname = config["model_name"]


Z_THRESH = config["cluster_forming_z"]
P_THRESH = config["cluster_p_threshold"]



FSLDIR = os.environ['FSLDIR'] #'/usr/share/fsl/5.0'



##
## Let's build our design matrix
##

designmatf = pj(configdir,config['design_matrix'])
design = pd.DataFrame.from_csv(designmatf).reset_index() # the precursor of the design matrix (still contains columns we will remove)



# Now drop the columns we no longer need from the design matrix
design_matrix = design.copy()
for col in config["design_columns_remove"]:
    design_matrix = design_matrix.drop(col,1) # drop the column in question
design_matrix = design_matrix.drop(config["grouping_column"],1) # drop the grouping column as well
    


##
## Let's build the .mat, .grp, .con files for use with FSL Flame
##

# Design matrix file
mat_file = pj(configdir,'%s.mat'%modelname)
aux_group.write_mat_file(design_matrix,mat_file)

# Grouping file
grp_file = pj(configdir,'%s.grp'%modelname)
aux_group.write_grp_file(design[config["grouping_column"]],grp_file)


# Contrasts file
contrasts = aux_group.parse_contrast_csv(pj(configdir,config["contrasts_file"]),design_matrix)
con_file = pj(configdir,'%s.con'%modelname)
aux_group.write_con_file(contrasts,con_file)


# F test file
config["use_ftest"]=False
if config["ftests_file"] and config["ftests_file"]!="None":
    ftests = aux_group.parse_ftest_csv(pj(configdir,config["ftests_file"]),contrasts)
    ftest_file = pj(configdir,'%s.fts'%modelname)
    aux_group.write_ftest_file(ftests,ftest_file)
    config["use_ftest"]=True
else:
    ftests = None
    ftest_file = None




# The directory for temporary files
work_dir   = get_path("work_dir",config)

# The directory where we will output everything we care about
output_dir = get_path("output_dir",config)



seeds = config["seeds"]

for seed in seeds:

    ##
    ## Build a list of the input files
    ##
    # Compute the z maps for these subjects/scans and for this seed
    zmaps = []
    masks = []
    for i,row in design.iterrows():
        # Compute the path to the given input file (a FC map)
        info = config.copy()
        info["subject_id"]  = row["subject_id"]
        info["run_id"]      = row["run_id"]
        info["seed_id"]     = seed

        path = get_path("subject_fc_map",info)
        zmaps.append(path)

        path = get_path("subject_func_mask",info)
        masks.append(path)
        

    gpa = aux_group.create_group_analysis(config["use_ftest"])
    gpa.config['execution']['crashdump_dir'] = pj(output_dir,"crash_files")

    #gpa.basedir = 'output_%s'%modelname
    #gpa.base_dir = '/brains/audiomotor.mrs/analysis/cpac/modelspec/work_%s/%i'%(modelname,seed)
    gpa.inputs.inputspec.mat_file = os.path.abspath(mat_file) #'%s.mat'%modelname
    gpa.inputs.inputspec.con_file = os.path.abspath(con_file) # '%s.con'%modelname
    gpa.inputs.inputspec.fts_file = os.path.abspath(ftest_file) # '%s.fts'%modelname
    gpa.inputs.inputspec.grp_file = os.path.abspath(grp_file) # '%s.grp'%modelname
    gpa.inputs.inputspec.z_threshold = Z_THRESH
    gpa.inputs.inputspec.p_threshold = P_THRESH
    gpa.inputs.inputspec.parameters = (FSLDIR, 'MNI152')

    assert os.path.exists(FSLDIR) # got really annoying error messages when this was false



    # Declare source input files
    sourcefiles = pe.Node(interface=util.IdentityInterface(fields=['zmaps','brainmasks']),
                          name='sourcefiles')
    sourcefiles.inputs.zmaps      = zmaps
    sourcefiles.inputs.brainmasks = masks



    ### create analysis specific mask
    maskmerger = pe.Node(interface=fsl.Merge(),
                     name='maskmerger')
    maskmerger.inputs.dimension   = 't'
    maskmerger.inputs.output_type = 'NIFTI_GZ'
    maskmerger.inputs.merged_file = 'allsubjects_mask.nii.gz'


    #-Tmin: min across time
    # -abs: absolute value
    #-bin: use (current image>0) to binarise
    computemask = pe.Node(interface=fsl.ImageMaths(),
                         name='merge_mask')
    computemask.inputs.op_string = '-abs -Tmin -bin'




    # Merge the FC (z) maps for the subjects
    merger = pe.Node(interface=fsl.Merge(),
                     name='merger')
    merger.inputs.dimension   = 't'
    merger.inputs.output_type = 'NIFTI_GZ'
    merger.inputs.merged_file = 'allsubjects_FC_zmap.nii.gz'




    sink = pe.Node(nio.DataSink(), name='output')
    sink.inputs.container      = seed # create a subdirectory for each seed
    sink.inputs.base_directory = output_dir



    mainwf = pe.Workflow(name='groupanalysis_%s'%seed)
    mainwf.base_dir = work_dir # "/brains/audiomotor.mrs/analysis/cpac/modelspec/work"

    mainwf.connect(sourcefiles, 'zmaps',       merger,        'in_files')                # feed the source files into the merger (will produce a single output Nifti file)
    mainwf.connect(sourcefiles, 'brainmasks',  maskmerger,    'in_files')                # feed the source brain mask files into the merger (will produce a single output Nifti file)
    mainwf.connect(maskmerger , 'merged_file', computemask,   'in_file')                 # from the merged file, compute the average (i.e. brain mask where at least one subject has a brain voxel).
    mainwf.connect(merger,      'merged_file', gpa,           'inputspec.merged_file' )  # feed the merged file into the group analysis
    mainwf.connect(computemask, 'out_file',    gpa,           'inputspec.merge_mask' )   # feed the merged mask file into the group analysis
    mainwf.connect(merger,      'merged_file', sink,          'merged')                  # save the merged file separately as well
    mainwf.connect(maskmerger , 'merged_file', sink,          'merged_mask')             # from the merged file, compute the average (i.e. brain mask where at least one subject has a brain voxel).
    mainwf.connect(computemask, 'out_file',    sink,          'group_mask')              # save the merged file separately as well


    # Set a whole lot more output files
    mainwf.connect(gpa, 'outputspec.zstats',                  sink, 'stats.unthreshold')
    mainwf.connect(gpa, 'outputspec.zfstats',                 sink, 'stats.unthreshold.@01')
    mainwf.connect(gpa, 'outputspec.fstats',                  sink, 'stats.unthreshold.@02')
    mainwf.connect(gpa, 'outputspec.cluster_threshold_zf',    sink, 'stats.threshold')
    mainwf.connect(gpa, 'outputspec.cluster_index_zf',        sink, 'stats.clusterMap')
    mainwf.connect(gpa, 'outputspec.cluster_localmax_txt_zf', sink, 'stats.clusterMap.@01')
    mainwf.connect(gpa, 'outputspec.overlay_threshold_zf',    sink, 'rendered')
    mainwf.connect(gpa, 'outputspec.rendered_image_zf',       sink, 'rendered.@01')
    mainwf.connect(gpa, 'outputspec.cluster_threshold',       sink, 'stats.threshold.@01')
    mainwf.connect(gpa, 'outputspec.cluster_index',           sink, 'stats.clusterMap.@02')
    mainwf.connect(gpa, 'outputspec.cluster_localmax_txt',    sink, 'stats.clusterMap.@03')
    mainwf.connect(gpa, 'outputspec.overlay_threshold',       sink, 'rendered.@02')
    mainwf.connect(gpa, 'outputspec.rendered_image',          sink, 'rendered.@03')


    mainwf.write_graph(dotfilename='groupanalysis', graph2use='colored', format='pdf', simple_form=True)


    mainwf.run() 
