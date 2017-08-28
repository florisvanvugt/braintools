"""

The purpose of this is to create a simple QC interface,
probably just output a script that people can run to check subjects
in turn.

"""

import yaml
import os
import pandas as pd
import subprocess


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



def gather(filename,target,info):
    """
    Gather a particular filename, i.e. copy it away from its
    embedded location somewhere deep in the results folder,
    and put it somewhere with a reasonable filename.
    """
    if not os.path.exists(info["gather_dir"]):
        os.makedirs(info["gather_dir"])

    assert os.path.exists(filename)
        
    target = pj(info["gather_dir"],target)
    print("%s -> %s"%(filename,target))
    
    cmd = ["cp",filename,target]
    subprocess.call(cmd)

    return target



yamlf = 'qc_pianists.yaml'

if True:
    
    with open(yamlf,'r') as f:
        config = yaml.load(f)


    # Get the root directory
    root = config["root_dir"]

    #config["FSLDIR"] = os.environ['FSLDIR']
    
    subjects = [ s.strip() for s in open(config["subjects_file"],'r').readlines() ]


    # A script that we will output
    output = ""

    motions = []
    
    for subject in subjects:

        info = config.copy()
        info["subject"]=subject
        output += "\n\n#\n# subject %s\n#\n"%subject

        ## Compare the alignment of the mean functional to the subject's anatomical
        anat_nii     = gather(pj(root,get_path("anat_brain",info)),  "%s_anat_subj.nii.gz"%(subject),     info)
        meanfunc_nii = gather(pj(root,get_path("mean_func",info)),   "%s_meanfunc_subj.nii.gz"%(subject), info)
        
        output += "# mean_func to anat\nfsleyes %s %s\n"%(anat_nii,meanfunc_nii)

        
        ## Compare the subject anatomical to the standard
        template_nii = gather(pj(root,get_path('template_skull_for_anat',info)), "standard_template.nii.gz",         info)
        anat_std_nii = gather(pj(root,get_path("anat_standard",info))          , "%s_meanfunc_std.nii.gz"%(subject), info)
        
        output += "# anat to standard\nfsleyes %s %s\n"%(template_nii,anat_std_nii)

        ## Extract motion time series
        motion_file = pj(root,get_path('motion_parameters',info))
        assert os.path.exists(motion_file)
        motions.append(pd.DataFrame.from_csv(motion_file))


    motions = pd.concat(motions)
    motions.to_csv(config["motion_file"])
        

    with open(config["script_file"],'w') as f:
        f.write(output)



# TODO Basic QC func_to_anat alignment: open mean_functional_in_anat and open anatomical_brain, look for CSF at the interhemispheric parietal area

# CHeck motion
