"""

The purpose of this is to create a simple QC interface,
probably just output a script that people can run to check subjects
in turn.

"""

import yaml
import os
import pandas as pd
import subprocess
import sys

from aux import *

pj = os.path.join




# Perhaps in the future for something much more sophisticated,
# I could take a look at https://github.com/poldrack/fmriqa


# Also this looks nice:
# https://faculty.washington.edu/madhyt/2016/12/10/180/


def get_path(pathname,info):
    """ 
    Given a path template, look for the appropriate items in the info dict
    to replace the placeholders with the actual values. 
    """
    template = info[pathname]
    for ky in info:
        template = template.replace('{%s}'%ky,str(info[ky]))
    #TODO 
    return template





if True:


    if len(sys.argv)<2:
        print("give the yaml config file as argument.")
        print("usage:")
        print("python make_qc.py <config.yml>")
        sys.exit(-1)

    yamlf = sys.argv[1] # = 'qc_pianists.yaml'

    
    with open(yamlf,'r') as f:
        config = yaml.load(f)


    # Get the root directory
    root = config["root_dir"]

    #config["FSLDIR"] = os.environ['FSLDIR']

    if exists(config,"subjects_file"):
        subjects = [ s.strip() for s in open(config["subjects_file"],'r').readlines() ]
    elif exists(config,"subjects"):
        subjects = config["subjects"]
    else:
        print("Error, you need to enter the subjects, either as <subjects_file> or <subjects>")
        sys.exit(-1)


    # A script that we will output
    output = ""

    motions = []
    
    for subject in subjects:

        info = config.copy()
        info["subject"]=subject
        output += "\n\n#\n# subject %s\n#\n"%subject

	if exists(info,"sample_func"):
            ## Compare the alignment of the mean functional to the subject's anatomical
            anat_nii     = gather(pj(root,get_path("anat_brain",info)),    "%s_anat_subj.nii.gz"%(subject),       info)
            meanfunc_nii = gather(pj(root,get_path("sample_func",info)),   "%s_samplefunc_subj.nii.gz"%(subject), info)
            output += "# sample_func to anat\nfsleyes %s %s\n"%(anat_nii,meanfunc_nii)


        if exists(info,"anat_standard"):
            ## Compare the subject anatomical to the standard
            template_nii = gather(pj(root,get_path('template_skull_for_anat',info)), "standard_template.nii.gz",          info)
            anat_std_nii = gather(pj(root,get_path("anat_standard",info))          , "%s_anat_standard.nii.gz"%(subject), info)
        
        output += "# anat to standard\nfsleyes %s %s\n"%(template_nii,anat_std_nii)

        ## Extract motion time series
        if exists(info,"motion_parameters"):
            motion_file = pj(root,get_path('motion_parameters',info))
            assert os.path.exists(motion_file)
            motions.append(pd.DataFrame.from_csv(motion_file))


    if len(motions)>0 and exists(info,"motion_file"):
        motions = pd.concat(motions)
        motions.to_csv(config["motion_file"])
        

    with open(config["script_file"],'w') as f:
        f.write(output)



# TODO Basic QC func_to_anat alignment: open mean_functional_in_anat and open anatomical_brain, look for CSF at the interhemispheric parietal area

# CHeck motion
