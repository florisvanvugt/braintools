#
# Running seed-based connectivity analysis
#

from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.utility as util

import nipype.interfaces.utility    as util
import nipype.interfaces.io         as nio
from nipype.interfaces import afni as afni

from nipype import SelectFiles

import sys
import json
import os
pj = os.path.join

import seedbased_connectivity

import yaml
import time
timestamp = time.strftime('%Y%m%d_%H%M')


if len(sys.argv)<2:
    print("You need to supply the ID for the seed set.")
    sys.exit()
seedset_id = sys.argv[1]
print("Seed set ID %s"%seedset_id)


# Open a file that tells us where the data for each subject is located
#subjects = list(json.load(open('subject_runs.json','r')).keys())
#subjects = ['s03'] # if you want to only run one subject (but why would you?)
basedir="/brains/pianists/output/pipeline_pianists-3mm-ants"
subjects = os.listdir(basedir)


# Functional runs
#runs = ["01pre","02post"]
runs = ["REST"]


N_PARALLEL_SUBJECTS = 8 # how many subjects to process in parallel
MAX_MEMORY_GB       = 30 # how much memory to use at most

# Amount of smoothing to perform (Gaussian FWHM in mm)
smooth_fwhm = 6


# The definition of the seeds, each seed is a file which is the mask for that seed (we'll also take the names of the seeds)
seed_dir    = '/brains/pianists/seed_regions/seeds/%s'%seedset_id

# Output directory
out_dir     = "/brains/pianists/output/sca_%s_%s/"%(seedset_id,timestamp)

# Preprocessing directory, where we get out info
preproc_dir = "/brains/pianists/output/reorganised_pipeline_pianists-3mm-ants"

# Working directory
work_dir    = "/brains/pianists/work/sca_%s"%seedset_id

# Crash log directory
crash_dir    = "/brains/pianists/crashlog"

# Templates of pre-processed data
preproc_templates = {
    # Data collected  (links to these files are created in make_links.py)
    #"func"      :"%s/{subject_id}/func/{run_id}/final/func_preprocessed_brain2mni.nii.gz"%preproc_dir,  # has to be in MNI space
    "func"      :"%s/{subject_id}/func/{run_id}/final/func_preprocessed2mni.nii.gz"%preproc_dir,  # has to be in MNI space
    "funcmask"  :"%s/{subject_id}/func/{run_id}/final/funcmask_mni.nii.gz"%preproc_dir
}





infosource = Node(interface=util.IdentityInterface(fields=['subject_id']),
                  name="select_subjects")
# Specify the subjects that we want to run (could be all)
infosource.iterables = ('subject_id',subjects)

runsource = Node(interface=util.IdentityInterface(fields=['run_id']),
                 name="select_run")
# Specify the subjects that we want to run (could be all)
runsource.iterables = ('run_id', runs)
#runsource.inputs.run_id = '01pre'  # -- Use this if you want to only analyse a single run


seedsource = Node(interface=util.IdentityInterface(fields=['seed_id']),name="select_seeds")
# Grab everything that looks like a seed in the seed directory
seeds = yaml.load(open('%s/seednames.yaml'%seed_dir,'r'))['seeds']
#seeds = [ f[:-7] for f in os.listdir(seed_dir) if f.endswith('nii.gz') ]
#seeds.sort()
seedsource.iterables = ('seed_id', seeds)




# Templates for the seed masks
seed_templates = {
    # Data collected  (links to these files are created in make_links.py)
    "seed_mask"   :"%s/{seed_id}.nii.gz"%seed_dir,  # has to be in MNI space
}



sourcefiles = Node(SelectFiles(preproc_templates),name='selectfiles')
sourcefiles.base_directory = "/"



seedfiles = Node(SelectFiles(seed_templates),name='selectseeds')
seedfiles.base_directory = "/"



datasink = Node(nio.DataSink(), name='sinker')
datasink.inputs.base_directory = out_dir

substitutions = [('_subject_id_', ''),
                 ('warped_out', 'final')]
datasink.inputs.substitutions = substitutions


#
# This is the main workflow that computes the connectivity map
#

# First, let's give it access to the data files
mainwf = Workflow(name='sca_master')
mainwf.base_dir = work_dir
mainwf.connect( infosource, "subject_id", sourcefiles, "subject_id" )
mainwf.connect( runsource,  "run_id",     sourcefiles, "run_id" )
mainwf.connect( seedsource, "seed_id",    seedfiles,   "seed_id" )


# For each mask get the average time course
maskave = Node(afni.Maskave(),name='extract_roi_ts')
maskave.inputs.quiet= True
maskave.inputs.out_file='seed_timecourse.1D'
mainwf.connect(sourcefiles, 'func',     maskave,  'in_file')
mainwf.connect(seedfiles,   'seed_mask',maskave,  'mask')



sca_w = seedbased_connectivity.create_sca("seedbased_connectivity_analysis",smooth_fwhm)

mainwf.connect( maskave,    'out_file',   sca_w,'inputspec.timeseries_one_d') # give the seed time course to the seed-based analysis
mainwf.connect( sourcefiles,"func",       sca_w,'inputspec.functional_file')  # give the original functional file to the seed-based analysis



def container_name(subject_id,run_id,seed_id):
    """ Return the name of the container for this subject and this run """
    return "%s/%s/%s"%(subject_id,run_id,seed_id) # this will hopefully create a subdirectory for each subject and each run
getcontainer = Node(name='get_container_name',interface=util.Function(input_names=['subject_id','run_id','seed_id'],
                                                                      output_names=["container_name"],
                                                                      function=container_name))
mainwf.connect( infosource,   'subject_id',     getcontainer, 'subject_id')
mainwf.connect( runsource,    'run_id',         getcontainer, 'run_id')
mainwf.connect( seedsource,   'seed_id',        getcontainer, 'seed_id')

mainwf.connect( getcontainer, 'container_name', datasink, 'container' )
mainwf.connect( maskave,      'out_file',       datasink, 'seedtimecourse_1d')
mainwf.connect( sourcefiles,  'funcmask',       datasink, 'funcmask')
mainwf.connect([ (sca_w,        datasink,         [('outputnode.correlation_file',           'correlation_file'),
                                                   ('outputnode.correlation_ztransf',        'correlation_ztransf'),
                                                   ('outputnode.correlation_ztransf_smooth', 'correlation_ztransf_smooth')]) ])
                 
mainwf.config['execution']['crashdump_dir'] = crash_dir
mainwf.write_graph(dotfilename='master_sca', graph2use='colored', format='pdf', simple_form=True)

multiproc_args = {'memory_gb':        MAX_MEMORY_GB,
                  #'status_callback':  log_nodes_cb,
                  'n_procs':          N_PARALLEL_SUBJECTS}

mainwf.run(plugin='MultiProc', plugin_args=multiproc_args)





# Let's put copies of the functional masks so that they will be easier to find,
# and so that the SCA directory is self-contained (you don't need any other
# directories anymore when you take the next processing steps).
# +TODO




# When completed, let's put a YAML file that indicates the data source and all so that I can trace it back.
config = {"seed_dir":seed_dir,
          "seedset_id":seedset_id,
          "preproc_dir":preproc_dir,
          "seeds":seeds,
          "timestamp":timestamp}
open('%s/sources.yaml'%out_dir,'w').write(yaml.dump(config))
