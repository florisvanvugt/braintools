

# This is supposed to be a very simple script that creates an HTML file that contains
# all the rendered files from a CPAC SCA pipeline output, just for allowing quick inspection.

import os
import re
import pandas as pd
import base64
import subprocess
import sys
import StringIO

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import yaml


USE_SEABORN = True

if USE_SEABORN:
    import seaborn as sns
    sns.set(style="ticks",font_scale=1.5,font='Helvetica')
    sns.set_palette(sns.hls_palette(8, l=.3, s=.8))


    
FULL_LIST = True


# Whether to show a (hidden) table with behavioural data in the output document
SHOW_BEHAV_TABLE = False


# The factor to use when rendering. When we make a rendering, we usually
# show all axial slices, except if the render factor is set here, in which
# case we show only 1 in every RENDER_FACTOR images. We also make the output
# image larger at first and then use imagemagick to scale it down.
#RENDER_FACTOR = 3
#TEMPLATE = "/usr/share/fsl/5.0/data/standard/MNI152_T1_1mm_brain.nii.gz"

#RENDER_FACTOR = 1
#TEMPLATE = "/usr/share/fsl/5.0/data/standard/MNI152_T1_3mm_brain.nii.gz"





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











import colorsys

def get_colors(num_colors):
    """ Returns a specified number of colours, maximally spaced in the colour cone.

    Arguments
    num_colours : number of colours to be generated
    """
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (30 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors





def make_lut(colour,greyscale=.7):
    """ 
    Make a look-up table for an overlay, with a particular colour in mind.
    This is for FSL, and it basically makes a two-part colour scale, the first
    part of which is a greyscale colour scale, and the second is a constant
    colour (which will correspond to the colour of the overlay).

    Arguments
    colour : the colour that the overlay will get
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






def make_zmap_rendering(thresh_stat_overlay,info):
    """
    Makes a rendering of a particular z map or f map.
    I wrote this function because for high-resolution analyses (e.g. functional at 1mm)
    CPAC outputs renderings in a very narrow but long image format.

    thresh_stat_overlay : the nifty file of the combined anatomical & stats map overlay file
    image_width : the desired image output width.

    Returns
    A base64-encoded string of the image
    """

    image_width = info["image_width"]
    RENDER_FACTOR = info["render_factor"]
    
    print("Making zmap %s"%thresh_stat_overlay)
    tmp_image    = "/tmp/_overlay.png"
    img_output = "/tmp/overlay.png"
    
    # Then produce an image rendition of this overlay
    cmd = ["slicer",thresh_stat_overlay,
           #"-l",lut,
           "-S",str(RENDER_FACTOR),str(image_width*RENDER_FACTOR),
           tmp_image
    ]
    subprocess.call(cmd)

    # Now convert the image back to a usable size
    cmd = ["convert","-resize",str(image_width),"-quality","9",
           tmp_image,img_output]
    subprocess.call(cmd)

    return base64.b64encode(open(img_output, "rb").read())

    




def make_cluster_rendering(clusterfile,cluster_n,colour,info):
    """ 
    Makes a rendering of a particular cluster, overlaid on a template.

    Arguments
    clusterfile : the file that defines the cluster. each voxel is assumed to be numbered, with 0 meaning the voxel is not part of any cluster, 1 means it is part of the first cluster, etc. etc.
    cluster_n : the number of the cluster to produce

    Returns
    A base64-encoded string of the image
    """

    image_width   = info["image_width"]
    RENDER_FACTOR = info["render_factor"]
    template      = info["template"]
    lut = make_lut(colour,greyscale=.4)
    
    # Define some temporary files
    cluster_only  = "/tmp/cluster_only.nii.gz"
    overlay       = "/tmp/zstat_overlay.nii.gz" # file that will be created in this process
    tmp_image     = "/tmp/tmp.png" # image file, the actual output that we care about
    overlay_image = "/tmp/cluster_overlay.png" # image file, the actual output that we care about

    # Remove any previous files so that we don't accidentally produce the same figure
    subprocess.call(['rm','-f',overlay_image,overlay,cluster_only,tmp_image])
    
    # First create a cluster mask for just that one cluster (0 for all voxels except for the voxels in the cluster which have value 1).
    cmd = ["3dcalc","-a",clusterfile,
           "-expr","1-bool(a-%i)"%cluster_n,
           "-prefix",cluster_only]
    subprocess.call(cmd)

    ## See e.g. https://faculty.washington.edu/madhyt/2016/12/10/180/
    cmd = ["overlay","0","0",template,"-a",
           cluster_only,"1","1",#str(cluster_n-.001),str(cluster_n+.001),
           overlay]
    #print(" ".join(cmd))
    subprocess.call(cmd)

    # Then produce an image rendition of this
    cmd = ["slicer",overlay,
           #"-L",
           "-l",lut,
           #"-A",str(image_width*RENDER_FACTOR),#"750",
           "-S",str(RENDER_FACTOR),str(image_width*RENDER_FACTOR),#"750",
           tmp_image
    ]

    #print(" ".join(cmd))
    subprocess.call(cmd)

    # Now convert the image back to a usable size
    cmd = ["convert","-resize",str(image_width),
           tmp_image,overlay_image]
    subprocess.call(cmd)


    return base64.b64encode(open(overlay_image, "rb").read())











# Now, for each seed which has some actual clusters, let's get some subject-level data about those
# clusters.

# Ensures that calls to AFNI 3dmaskave will be quiet
os.environ["AFNI_NIFTI_TYPE_WARN"] = "NO"





def make_cluster_scatter(clustermaskf,cluster_n,mergedf,stat_index,stat_name,is_f_test,info,label,color="red"):
    """ 
    Given a cluster mask file, and a subject merged file, extract all the connectivity
    values for a given cluster for all subjects separately. Average them and then
    compare them against the behavioural EVs producing a scatter plot.

    Arguments
    clustermaskf   : the file name containing the cluster mask (each voxel 0=no cluster, 1=part of cluster 1, etc.)
    cluster_n      : which cluster to produce a scatter plot for
    mergedf        : the file name containing the merged data, i.e. one volume for every subject containing their correlation values
    design_matrix  : the design matrix of the study, assumed to be a pandas DataFrame
    stat_index     : the number of the statistic (e.g. the number of the contrast or the number of the f-test
    stat_name      : the name of the statistic (e.g. the name of the contrast)
    is_f_test      : boolean telling us whether this is an F-test (True) or a contrast (False)
    contrast       : the contrast to be plotted
    info           : general configuration information
    label          : label to be added to the plot
    color          : the plotting color
    """


    def get_contrast_dependencies(contrast_def):
        """ 
        Given a particular contrast, return us the columns
        from the original phenotype table that this contrast
        depends on (usually just one column really).

        Arguments
        contrast_i : the index of the contrast
        """

        # Find the columns of the design matrix implicated in this contrast
        nonzerocols = [ info["design_mat_columns"][i] for i in np.where(contrast_def!=0)[0] ]

        behavs = []
        for col in nonzerocols:

            if col in info["pheno"].columns:
                behavs.append(col)
            else:
                # If this is one level of a multilevel factor
                mtc = re.match(r'C\((.+)\)\[(\d+)\]',col)
                if mtc:
                    behavs.append(mtc.group(1))

                
                
        behavs = list(set(behavs)) # make unique
        return behavs


    #tab = {"subject":cl["subject.list"]}
    tab = pd.DataFrame(info["design_mat"])
    tab.columns = info["design_mat_columns"]
    contrast_mat  = info["contrast_mat"]

    
    is_categorical = False
    if not is_f_test:
        
        contrast_i = stat_index-1
        # Extract just that one contrast
        #design_matrix = info["design_mat"]
        contrast_def = np.array(contrast_mat[contrast_i]).flatten() # list of the multipliers of the design matrix columns to find the contrast for each subject
        deps = get_contrast_dependencies(contrast_def)

    else:

        if ftestf==None:
            print("Error, no ftest file for %s"%stat_name)
            exit()

        ftest_mat = info["ftest_mat"]
        # Let's extract the contrasts that this f-test depends on...
        fdef = np.array(ftest_mat[stat_index-1]).flatten()
        contrast_is, = np.where( fdef!=0 )
        #print("%s depends on contrasts"%stat_name,contrast_is)

        deps = []
        for i in contrast_is:
            #print("Contrast %i"%i)
            #design_matrix = info["design_mat"]
            contrast_def = np.array(contrast_mat[i]).flatten() # list of the multipliers of the design matrix columns to find the contrast for each subject
            
            deps += get_contrast_dependencies(contrast_def)
        deps = list(set(deps))


        
    if len(deps)==1:
        # Great, we are good to go!
        behav = str(deps[0])
        tab[behav] = info["pheno"][behav]
        EV_name = behav

        is_categorical = behav in info["ev_selections"].get("categorical",[])

    else:

        if not is_f_test:
            # Compute the actual contrast and show that
            EV_name = stat_name
            # Let's now add this to the design matrix
            tab[EV_name]=np.array(np.dot(np.matrix(info["design_mat"]),contrast_def)).flatten()

        else:
            print("## Not sure what behavioural variable to plot for f-test %s"%stat_name)
            return {'table':pd.DataFrame({}),'png':""} #,"cluster.rendering":clusterrender}
        

    

    for valtype in ["mean","min","max","median"]:

        cmd = ["3dmaskave",
               "-quiet",
               "-mask",clustermaskf,
               "-mrange",str(cluster_n),str(cluster_n) ]

        if valtype!="mean":
            cmd+= ["-%s"%valtype]

        cmd += [mergedf]

        result = subprocess.check_output(cmd) #, stdout=subprocess.PIPE)
        tb = [ float(f) for f in str(result).split() ]
        assert len(tb)==tab.shape[0] #len(cl["subject.list"])
        tab[valtype] = tb

    #df = pd.DataFrame(tab)
    tab = tab.reset_index()


    # So that gives us for every subject the connectivity values within that cluster.
    # Now the question is how that relates to their EVs.
    # What we have is a contrast, so for each subject we can calculate the value
    # of the contrast, e.g. if the contrast is EV1>EV2 then we can calculate EV1-EV2
    # which is the relevant value to plot in a scatter plot.

    # Okay, now a little hack -- TODO: make this proper, compute the actual contrast value...
    # For now, we just find the regressor with the most appropriate looking name
    #behav = ""
    #regressors = list(design_matrix.columns.values)
    #for regr in regressors:
    #    if contrast.find(regr)>-1: # if this contains the regressor
    #        behav = regr

    # Now let's plot this
    if not USE_SEABORN:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        ax.plot(tab[EV_name],tab["mean"],'o',color=color)
    else:

            
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        if is_categorical:
            for i,dat in tab.groupby(EV_name):
                mn = np.mean(dat["mean"])
                ax.bar(i,mn,color=color,alpha=.3)
            
            ax.plot(tab[EV_name],tab["mean"],'o',color=color)
        else:
            sns.regplot(tab[EV_name],tab["mean"],color=color,ax=ax)

    # If the values cross zero, add a zero line
    minm,maxm= min(tab["mean"]),max(tab["mean"])
    if np.sign(minm)!=np.sign(maxm):
        plt.plot(tab[EV_name],[0]*len(tab[EV_name]),'-',color="gray",alpha=.5)

    #for i,row in tab.iterrows():
    #    ax.text(row[EV_name],row["mean"],row[info["pheno_subject_column"]],fontsize=8,alpha=.5)
    # TODO -- fix
    
    sns.despine(offset=5)
    ax.set_title(label)
    ax.set_xlabel(EV_name)
    ax.set_ylabel("FC")
    plt.tight_layout()
    fig.savefig('/tmp/sca.png',dpi=75)
    plt.close()
    encoded = base64.b64encode(open('/tmp/sca.png', "rb").read())


    return {'table':tab,'png':encoded} #,"cluster.rendering":clusterrender}











if __name__=="__main__":



    if len(sys.argv)<2:
        print("give the yaml config file as argument.")
        print("usage:")
        print("python make_sca_report <specs.yml>")

        from Tkinter import *
        from tkFileDialog import *
        #from Tkinter.filedialog import askopenfilename
        fname = askopenfilename(filetypes=[("Yaml config files", ("*.yml","*.yaml")),("All files","*.*")])
        if fname:
            yamlf = fname
            print("Inhaling config filename %s"%yamlf)
        else:
            sys.exit(-1)


    else:
            
        # Let's read the YAML
        yamlf = sys.argv[1]

    with open(yamlf,'r') as f:
        gpa_dat = yaml.load(f)

        
    startdir   = gpa_dat["output_dir"]
    modelname  = gpa_dat["model_name"]
    analysis_title =  "%s %s"%(os.path.basename(startdir),modelname)

    # Read the pheno type file
    phenofile = gpa_dat["pheno_file"]
    gpa_dat["pheno"]=pd.DataFrame.from_csv(phenofile).reset_index()

    
    # This is the file that we will write our html to
    outputfile = os.path.join(gpa_dat["output_dir"],"sca_report.html")
    #if outputfile.endswith("/"):
    #    outputfile=outputfile[:-1]
    #outputfile+=".html"

    htmlout=""
    
    
    print("Writing output to %s"%outputfile)


    basedir = startdir
    


    # Let's read the design-general files (design matrix, contrasts, etc.)
    design_mat_file        = os.path.join(get_path("output_dir",gpa_dat),get_path("design_matrix_file",gpa_dat))

    dmf = open(design_mat_file).read()
    dmf_i = dmf.find('/Matrix')
    assert dmf_i>-1 # it should be
    designmat = np.genfromtxt(StringIO.StringIO(dmf[dmf_i+7:]))
    # columns are the columns in the design matrix
    # rows are the different contrasts
    gpa_dat["design_mat"]=designmat

    gpa_dat["design_mat_columns"]= gpa_dat["regressors"].split()


    subject_list_file      = os.path.join(get_path("output_dir",gpa_dat),get_path("subject_list_file",gpa_dat))
    subjectl               = [ v.strip() for v in open(subject_list_file,'r').readlines() ]


    modelf = os.path.join(get_path("output_dir",gpa_dat),get_path("contrasts_file",gpa_dat))
    with open(modelf,'r') as f:
        contf = f.read()
    contrasts = dict([ (int(x),n) for (x,n) in re.findall(r'/ContrastName(\d+)\s*([<>\.\w_\-\+]+)',contf)])
    f_tests   = dict([ (f+1   ,n) for (f,n) in enumerate(re.findall(r'[fF]_[tT][Ee][Ss][Tt]_?([<>\w\_\-\+]+)',contf)) ])


    print("Contrasts:")
    print(contrasts)

    gpa_dat["contrasts"]=contrasts
    gpa_dat["n_contrasts"]=len(contrasts)

    # Read in the contrasts file
    with open(modelf,'r') as f:
        contrasts = f.read()

    # Find the contrast matrix definition and read it into a matrix
    mati = contrasts.find('/Matrix')
    if mati>-1: # it should be
        contrasts_mat = np.genfromtxt(StringIO.StringIO(contrasts[mati+7:]))
        # columns are the columns in the design matrix
        # rows are the different contrasts
        gpa_dat["contrast_mat"]=contrasts_mat


    # Read the f-tests file if we have one
    ftest_mat = None
    ftestf = gpa_dat["ftests_file"]
    if ftestf!=None and ftestf!="None":
        
        with open(ftestf,'r') as f:
            ftestcont = f.read()
        # Find the contrast matrix definition and read it into a matrix
        mati = ftestcont.find('/Matrix')
        if mati>-1: # it should be
            ftest_mat = np.genfromtxt(StringIO.StringIO(ftestcont[mati+7:]))
            # columns are the columns in the design matrix
            # rows are the different contrasts
        gpa_dat["ftest_mat"]=ftest_mat


    # First, let's find the subpath where we have the split up by seed,
    # i.e. the path where you have the _fisher_z_scoreXX
    # subdirectories.

   

    master_cluster_list = []
    #i = 0

    for i,seed in enumerate(gpa_dat["seeds"]):

        info = gpa_dat.copy()

        info["seed_n"] = i
        info["seed"]   = seed
        seed_dir = os.path.join(get_path("output_dir",info),
                                get_path("seed_dir",info))

        # Now we walk through the results, looking for all contrasts and F tests in turn,
        # start looking for the first contrast, and keep going until we hit a contrast
        # that does not exist...
        for stattype in [""]: # ,"f"]: TODO-- make this work for F tests too

            for j in range(gpa_dat["n_contrasts"]):

                info["contrast"]  =j
                info["contrast+1"]=j+1

                cluster_list_file = os.path.join(seed_dir,get_path("cluster_list_file",info))
                merged_file       = os.path.join(seed_dir,get_path("merged_file",info))
                rendered_nii      = os.path.join(seed_dir,get_path("rendered_nii",info))
                clustermask       = os.path.join(seed_dir,get_path("cluster_mask_file",info))
                
                bodyname     = "z%sstat%i"%(stattype,j)
                
                print(cluster_list_file,merged_file,rendered_nii)
                                
                assert os.path.exists(cluster_list_file)
                assert os.path.exists(merged_file)
                assert os.path.exists(rendered_nii)


                # Open the cluster listing
                with open(cluster_list_file,'r') as f:
                    lns = f.readlines()
                if len(lns)>1: # if there are actually any clusters (the first line is to be ignored because it's the header)
                    clustertab = pd.DataFrame.from_csv(cluster_list_file,sep="\t") #np.genfromtxt(clusterl,delimiter="\t",names=True)
                else:
                    clustertab = None

                statname = ""
                if stattype=="": # contrast
                    statname=info["contrasts"][j+1]
                elif stattype=="f": # F-test
                    statname=info["f_tests"][j+1] #"F-"+f_tests[j]
                    

                # Make a rendering of the zmap (overlay)
                renderedimg = make_zmap_rendering(rendered_nii,info)


                master_cluster_list.append({"seed"          :i+1,
                                            "path"          :seed_dir,
                                            "body"          :bodyname,
                                            "stattype"      :stattype,
                                            "n"             :j,
                                            "stat"          :"%s%i"%(stattype,j),
                                            "cluster.file"  :cluster_list_file,
                                            "cluster.mask"  :clustermask,
                                            "n.cluster"     :len(lns)-1,
                                            "rendered"      :rendered_nii,
                                            "rendered_img"  :renderedimg,
                                            "clustertab"    :clustertab,
                                            "merged.file"   :merged_file,
                                            "statname"      :statname
                })




if True:
    

    # Now we go through the list again and find seeds for which
    # there is at least one cluster. In those cases, we have to
    # make a table and go through each of the seeds in turn.

    for cl in master_cluster_list:
        cl["persubject"]={}
        if cl["n.cluster"]>0 and cl["merged.file"]!=None:
            # Given a cluster map (a nifty file where each voxel is numbered according to which cluster it is part of, or zero if it is not part of any cluster),
            # create a cluster mask and then find the individual average correlation coefficients for the voxels
            # within that cluster.
            colours = get_colors(cl["n.cluster"])

            # Then we can take the average from that mask for each of the volumes in the merged file, i.e. for each subject
            for cluster_n in range(cl["n.cluster"]):


                clusterrender = make_cluster_rendering(cl["cluster.mask"],cluster_n+1,colours[cluster_n],info)


                clsc = make_cluster_scatter(clustermaskf = cl["cluster.mask"],
                                            cluster_n    = cluster_n+1,
                                            mergedf      = cl["merged.file"],
                                            stat_index   = cl["n"],
                                            stat_name    = cl["statname"],
                                            is_f_test    = cl["stattype"]=="f",
                                            info         = info,
                                            label        = "Seed %i stat %s cluster %i"%(cl["seed"],cl["statname"],cluster_n+1),
                                            color         = colours[cluster_n]
                )

                clsc["cluster.rendering"]=clusterrender

                cl["persubject"][cluster_n+1]=clsc





    htmlout+="""<html><body>
    <style>
    #seedresults {
        font-family: Helvetica, sans-serif;
        border-collapse: collapse;
    //width: 100%;
    }

    #seedresults td, #seedresults th {
        border: 1px solid #ddd;
        padding: 6px;
    }

    #seedresults tr:nth-child(even){background-color: #f2f2f2;}

    #seedresults tr:hover {background-color: #ddd;}

    #seedresults th {
        padding-top: 12px;
        padding-bottom: 12px;
        text-align: left;
        background-color: #e31c63;
        color: white;
    }

    .dataframe {
        font-family: Helvetica, sans-serif;
        border-collapse: collapse;
    }

    .dataframe th {
        text-align: left;
        background-color: #1e28a7;
        color: white;
    }



    </style>
    """






    htmlout+="<h1>%s</h1>"%analysis_title
    htmlout+="<h1>Cluster list</h1>"
    htmlout+="<table id=\"seedresults\">"
    htmlout+="<tr><th>Seed</th><th>Stat</th><th>Cluster file</th><th># clust</th><th>Render</th></tr>\n"
    for cl in master_cluster_list:

        nm = cl["statname"] if cl["stattype"]=="" else "F["+cl["statname"]+"]"
        
        if cl["n.cluster"]==0:
            nclust = "."
            render = "."
        else:
            nclust = str(cl["n.cluster"])
            render = "<a href=\"#%s\">render</a>"%cl["rendered"]
        htmlout+="<tr><td>%i</td><td>%s</td><td><a href=\"%s\">clusters</a></td><td>%s</td><td>%s</td></tr>\n"%(cl["seed"],
                                                                                                                #cl["stat"],
                                                                                                                nm,
                                                                                                                cl["cluster.file"],
                                                                                                                nclust,
                                                                                                                render)
    htmlout+="</table>"



    if FULL_LIST:

        htmlout+="<h1>Full listing</h1>\n"

        for cl in master_cluster_list:
            rendered_filename = cl["rendered"]


            if cl["n.cluster"]>0:
                htmlout+="<h3><a name=\"%s\">Seed %i stat %s</a></h3>"%(rendered_filename,cl["seed"],cl["statname"])

                #print("<p><img src=\"%s\" /></p>\n"%r)

                # Now read the image and put it directly into the HTML (so that it is standalone)
                htmlout+="<p><img src=\"data:image/png;base64,%s\" /></p>"%cl["rendered_img"]


                fullp = "%s/stats/threshold/thresh_%s.nii.gz"%(cl["path"],cl["body"])
                #fullp = "%s/%s/rendered/thresh_%s_overlay.nii.gz"%(cwd,cl["path"],cl["body"])
                cmd = "fsleyes /usr/share/fsl/5.0/data/standard/MNI152_T1_1mm.nii.gz --name MNI152_T1_1mm --brightness 40 %s -n \"seed %i %s\" --cmap red-yellow"%(fullp,cl["seed"],cl["statname"])
                htmlout+="<p style=\"font-size: small;\">%s</p>"%cmd

                # Print the list of clusters and some data associated with it
                cl["clustertab"]["P"] = [ "%.09f"%f for f in list(cl["clustertab"]["P"]) ]
                htmlout+=cl["clustertab"].to_html()

                for cli in cl["persubject"]:
                    htmlout+="<h4>Cluster %i</h4>"%cli

                    # Show the cluster table
                    if SHOW_BEHAV_TABLE:
                        htmlout+="<span style=\"display:none\">\n"
                        htmlout+=cl["persubject"][cli]["table"].to_html()
                        htmlout+="</span>"
                        
                    #print("<p>Cluster %i - subject values %s"%(cli,cl["persubject"][cli]))
                    htmlout+="<p><table><tr><td><img src=\"data:image/png;base64,%s\" /></td>\n"%cl["persubject"][cli]["png"]
                    htmlout+="<td><img style=\"width : 600; height:auto\" src=\"data:image/png;base64,%s\" /></td></tr></table>\n"%cl["persubject"][cli]["cluster.rendering"]


                htmlout+="<p style=\"padding:50px\" />" # add a little space to clarify



            else:
                #print("<p><a href=\"%s\">show</a></p>\n"%r)
                pass

                #print("<p>fsleyes --scene ortho --layout horizontal /usr/share/fsl/5.0/data/standard/MNI152_T1_1mm.nii.gz --overlayType volume --name MNI152_T1_1mm --volume 0 %s --cmap red-yellow &amp;</p>"%fullp)




    htmlout+="</body></html>\n"


    # Now actually write the output to file
    fout = open(outputfile,'w')
    fout.write(htmlout)
    fout.close()
    
