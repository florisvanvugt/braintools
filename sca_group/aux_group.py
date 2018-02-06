
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util



def easy_thresh(wf_name):
    """
    Workflow for carrying out cluster-based thresholding 
    and colour activation overlaying
    
    Parameters
    ----------
    wf_name : string 
        Workflow name
        
    Returns
    -------
    easy_thresh : object 
        Easy thresh workflow object
    
    Notes
    -----
    
    `Source <https://github.com/FCP-INDI/C-PAC/blob/master/CPAC/easy_thresh/easy_thresh.py>`_
        
    Workflow Inputs::
    
        inputspec.z_stats : string (nifti file)
            z_score stats output for t or f contrast from flameo
        
        inputspec.merge_mask : string (nifti file)
            mask generated from 4D Merged derivative file
        
        inputspec.z_threshold : float
            Z Statistic threshold value for cluster thresholding. It is used to 
            determine what level of activation would be statistically significant. 
            Increasing this will result in higher estimates of required effect.
        
        inputspec.p_threshold : float
            Probability threshold for cluster thresholding.
        
        inputspec.paramerters : string (tuple)
            tuple containing which MNI and FSLDIR path information
            
    Workflow Outputs::
    
        outputspec.cluster_threshold : string (nifti files)
           the thresholded Z statistic image for each t contrast
        
        outputspec.cluster_index : string (nifti files)
            image of clusters for each t contrast; the values 
            in the clusters are the index numbers as used 
            in the cluster list.
        
        outputspec.overlay_threshold : string (nifti files)
            3D color rendered stats overlay image for t contrast
            After reloading this image, use the Statistics Color 
            Rendering GUI to reload the color look-up-table
        
        outputspec.overlay_rendered_image : string (nifti files)
           2D color rendered stats overlay picture for each t contrast
        
        outputspec.cluster_localmax_txt : string (text files)
            local maxima text file, defines the coordinates of maximum value
            in the cluster
    
    
    Order of commands:
    
    - Estimate smoothness of the image::
        
        smoothest --mask= merge_mask.nii.gz --zstat=.../flameo/stats/zstat1.nii.gz
        
        arguments
        --mask  :  brain mask volume
        --zstat :  filename of zstat/zfstat image
    
    - Create mask. For details see `fslmaths <http://www.fmrib.ox.ac.uk/fslcourse/lectures/practicals/intro/index.htm#fslutils>`_::
        
        fslmaths ../flameo/stats/zstat1.nii.gz 
                 -mas merge_mask.nii.gz 
                 zstat1_mask.nii.gz
        
        arguments
        -mas   : use (following image>0) to mask current image

    - Copy Geometry image dimensions, voxel dimensions, voxel dimensions units string, image orientation/origin or qform/sform info) from one image to another::
    
        fslcpgeom MNI152_T1_2mm_brain.nii.gz zstat1_mask.nii.gz
    
    - Cluster based thresholding. For details see `FEAT <http://www.fmrib.ox.ac.uk/fsl/feat5/detail.html#poststats>`_::
        
        cluster --dlh = 0.0023683100 
                --in = zstat1_mask.nii.gz 
                --oindex = zstat1_cluster_index.nii.gz 
                --olmax = zstat1_cluster_localmax.txt
                --othresh = zstat1_cluster_threshold.nii.gz 
                --pthresh = 0.0500000000 
                --thresh = 2.3000000000 
                --volume = 197071
                
        arguments 
        --in    :    filename of input volume
        --dlh   :    smoothness estimate = sqrt(det(Lambda))
        --oindex  :  filename for output of cluster index
        --othresh :  filename for output of thresholded image
        --olmax   :  filename for output of local maxima text file
        --volume  :  number of voxels in the mask
        --pthresh :  p-threshold for clusters
        --thresh  :  threshold for input volume
        
     Z statistic image is thresholded to show which voxels or clusters of voxels are activated at a particular significance level.
     A Z statistic threshold is used to define contiguous clusters. Then each cluster's estimated significance level (from GRF-theory) 
     is compared with the cluster probability threshold. Significant clusters are then used to mask the original Z statistic image.
    
    - Get the maximum intensity value of the output thresholded image. This used is while rendering the Z statistic image:: 
        
        fslstats zstat1_cluster_threshold.nii.gz -R
        
        arguments
        -R  : output <min intensity> <max intensity>

    - Rendering. For details see `FEAT <http://www.fmrib.ox.ac.uk/fsl/feat5/detail.html#poststats>`_::
         
        overlay 1 0 MNI152_T1_2mm_brain.nii.gz 
               -a zstat1_cluster_threshold.nii.gz 
               2.30 15.67 
               zstat1_cluster_threshold_overlay.nii.gz
               
        slicer zstat1_cluster_threshold_overlay.nii.gz 
               -L  -A 750 
               zstat1_cluster_threshold_overlay.png
    
      The Z statistic range selected for rendering is automatically calculated by default, 
      to run from red (minimum Z statistic after thresholding) to yellow (maximum Z statistic, here 
      maximum intensity).
      
    High Level Workflow Graph:
    
    .. image:: ../images/easy_thresh.dot.png
       :width: 800
    
    
    Detailed Workflow Graph:
    
    .. image:: ../images/easy_thresh_detailed.dot.png
       :width: 800
               
    Examples
    --------
    
    >>> import easy_thresh
    >>> preproc = easy_thresh.easy_thresh("new_workflow")
    >>> preproc.inputs.inputspec.z_stats= 'flameo/stats/zstat1.nii.gz'
    >>> preproc.inputs.inputspec.merge_mask = 'merge_mask/alff_Z_fn2standard_merged_mask.nii.gz'
    >>> preproc.inputs.inputspec.z_threshold = 2.3
    >>> preproc.inputs.inputspec.p_threshold = 0.05
    >>> preproc.inputs.inputspec.parameters = ('/usr/local/fsl/', 'MNI152')
    >>> preporc.run()  -- SKIP doctest
    
    """

    easy_thresh = pe.Workflow(name=wf_name)

    inputnode = pe.Node(util.IdentityInterface(fields=['z_stats',
                                                       'merge_mask',
                                                       'z_threshold',
                                                       'p_threshold',
                                                       'parameters']),
                         name='inputspec')

    outputnode = pe.Node(util.IdentityInterface(fields=['cluster_threshold',
                                                        'cluster_index',
                                                        'cluster_localmax_txt',
                                                        'overlay_threshold',
                                                        'rendered_image']),
                         name='outputspec')

    ### fsl easythresh 
    # estimate image smoothness
    smooth_estimate = pe.MapNode(interface=fsl.SmoothEstimate(),
                                    name='smooth_estimate',
                                    iterfield=['zstat_file'])

    # run clustering after fixing stats header for talspace
    zstat_mask = pe.MapNode(interface=fsl.MultiImageMaths(),
                                  name='zstat_mask',
                                  iterfield=['in_file'])
    #operations to perform
    #-mas use (following image>0) to mask current image
    zstat_mask.inputs.op_string = '-mas %s'

    #fslcpgeom
    #copy certain parts of the header information (image dimensions, 
    #voxel dimensions, voxel dimensions units string, image orientation/origin 
    #or qform/sform info) from one image to another
    copy_geometry = pe.MapNode(util.Function(input_names=['infile_a', 'infile_b'],
                                             output_names=['out_file'],
                                             function=copy_geom),
                                             name='copy_geometry',
                                             iterfield=['infile_a', 'infile_b'])

    ##cluster-based thresholding
    #After carrying out the initial statistical test, the resulting 
    #Z statistic image is then normally thresholded to show which voxels or 
    #clusters of voxels are activated at a particular significance level.
    #A Z statistic threshold is used to define contiguous clusters. 
    #Then each cluster's estimated significance level (from GRF-theory) is 
    #compared with the cluster probability threshold. Significant clusters 
    #are then used to mask the original Z statistic image for later production 
    #of colour blobs.This method of thresholding is an alternative to 
    #Voxel-based correction, and is normally more sensitive to activation. 
#    cluster = pe.MapNode(interface=fsl.Cluster(),
#                            name='cluster',
#                            iterfield=['in_file', 'volume', 'dlh'])
#    #output of cluster index (in size order)
#    cluster.inputs.out_index_file = True
#    #thresholded image
#    cluster.inputs.out_threshold_file = True
#    #local maxima text file
#    #defines the cluster cordinates
#    cluster.inputs.out_localmax_txt_file = True

    cluster = pe.MapNode(util.Function(input_names =  ['in_file',
                                                       'volume', 
                                                       'dlh',
                                                       'threshold', 
                                                       'pthreshold', 
                                                       'parameters'],
                                       output_names = ['index_file', 
                                                       'threshold_file', 
                                                       'localmax_txt_file'],
                                       function = call_cluster),
                                       name = 'cluster',
                                       iterfield = ['in_file', 'volume', 'dlh'])

    #max and minimum intensity values
    image_stats = pe.MapNode(interface=fsl.ImageStats(),
                             name='image_stats',
                             iterfield=['in_file'])
    image_stats.inputs.op_string = '-R'


    #create tuple of z_threshold and max intensity value of threshold file
    create_tuple = pe.MapNode(util.Function(input_names=['infile_a', 'infile_b'],
                                            output_names=['out_file'],
                                            function=get_tuple),
                                            name='create_tuple',
                                            iterfield=['infile_b'])


    #colour activation overlaying
    overlay = pe.MapNode(interface=fsl.Overlay(),
                            name='overlay',
                            iterfield=['stat_image', 'stat_thresh'])
    overlay.inputs.transparency = True
    overlay.inputs.auto_thresh_bg = True
    overlay.inputs.out_type = 'float'


    #colour rendering
    slicer = pe.MapNode(interface=fsl.Slicer(), name='slicer',
                           iterfield=['in_file'])
    #set max picture width
    slicer.inputs.image_width = 750
    # set output all axial slices into one picture
    slicer.inputs.all_axial = True

    #function mapnode to get the standard fsl brain image 
    #based on parameters as FSLDIR,MNI and voxel size
    get_backgroundimage = pe.MapNode(util.Function(input_names=['in_file',
                                                                'file_parameters'],
                                                   output_names=['out_file'],
                                                   function=get_standard_background_img),
                                     name='get_bckgrndimg1', iterfield=['in_file'])
    
    #function node to get the standard fsl brain image
    #outputs single file
    get_backgroundimage2 = pe.Node(util.Function(input_names=['in_file',
                                                              'file_parameters'],
                                                 output_names=['out_file'],
                                                 function=get_standard_background_img),
                                   name='get_backgrndimg2')

    #connections
    easy_thresh.connect(inputnode, 'z_stats', smooth_estimate, 'zstat_file' )
    easy_thresh.connect(inputnode, 'merge_mask', smooth_estimate, 'mask_file' )

    easy_thresh.connect(inputnode, 'z_stats', zstat_mask, 'in_file')
    easy_thresh.connect(inputnode, 'merge_mask', zstat_mask, 'operand_files')

    easy_thresh.connect(zstat_mask, 'out_file', get_backgroundimage, 'in_file' )
    easy_thresh.connect(inputnode, 'parameters', get_backgroundimage, 'file_parameters')

    easy_thresh.connect(get_backgroundimage, 'out_file', copy_geometry, 'infile_a' )
    easy_thresh.connect(zstat_mask, 'out_file', copy_geometry, 'infile_b')

    easy_thresh.connect(copy_geometry, 'out_file', cluster, 'in_file')
    easy_thresh.connect(inputnode, 'z_threshold', cluster, 'threshold')
    easy_thresh.connect(inputnode, 'p_threshold', cluster, 'pthreshold')
    easy_thresh.connect(smooth_estimate, 'volume', cluster, 'volume')
    easy_thresh.connect(smooth_estimate, 'dlh', cluster, 'dlh')
    easy_thresh.connect(inputnode, 'parameters', cluster, 'parameters')

    easy_thresh.connect(cluster, 'threshold_file', image_stats, 'in_file')

    easy_thresh.connect(image_stats, 'out_stat', create_tuple, 'infile_b')
    easy_thresh.connect(inputnode, 'z_threshold', create_tuple, 'infile_a')

    easy_thresh.connect(cluster, 'threshold_file', overlay, 'stat_image')
    easy_thresh.connect(create_tuple, 'out_file', overlay, 'stat_thresh')

    easy_thresh.connect(inputnode, 'merge_mask', get_backgroundimage2, 'in_file' )
    easy_thresh.connect(inputnode, 'parameters', get_backgroundimage2, 'file_parameters')

    easy_thresh.connect(get_backgroundimage2, 'out_file', overlay, 'background_image')

    easy_thresh.connect(overlay, 'out_file', slicer, 'in_file')

    easy_thresh.connect(cluster, 'threshold_file', outputnode, 'cluster_threshold')
    easy_thresh.connect(cluster, 'index_file', outputnode, 'cluster_index')
    easy_thresh.connect(cluster, 'localmax_txt_file', outputnode, 'cluster_localmax_txt')
    easy_thresh.connect(overlay, 'out_file', outputnode, 'overlay_threshold')
    easy_thresh.connect(slicer, 'out_file', outputnode, 'rendered_image')

    return easy_thresh





def call_cluster(in_file, volume, dlh, threshold, pthreshold, parameters):
    
    import os
    import re
    import subprocess as sb
  
    out_name = re.match('z(\w)*stat(\d)+', os.path.basename(in_file))
    
    filename, ext = os.path.splitext(os.path.basename(in_file))
    ext=  os.path.splitext(filename)[1] + ext
    filename = os.path.splitext(filename)[0]

    if out_name:
        out_name= out_name.group(0)
    else:
        out_name = filename
            
    #print "out_name --> ", out_name
    
    FSLDIR = parameters[0]
    
    index_file = os.path.join(os.getcwd(), 'cluster_mask_' + out_name + ext)
    threshold_file = os.path.join(os.getcwd(), 'thresh_' + out_name + ext)
    localmax_txt_file = os.path.join(os.getcwd(), 'cluster_'+ out_name +'.txt')

    cmd_path = os.path.join(FSLDIR, 'bin/cluster')    
        
    f = open(localmax_txt_file,'wb')
    
    cli = [ cmd_path,
                    '--dlh=' + str(dlh),
                    '--in=' + in_file,
                    '--oindex=' + index_file,
                    '--othresh=' + threshold_file,
                    '--pthresh=' + str(pthreshold),
                    '--thresh=' +  str(threshold),
                    '--volume=' + str(volume)]
    #print "calling %s"%(" ".join(cli))    
    cmd = sb.Popen(cli,
                    stdout= f)

    stdout_value, stderr_value = cmd.communicate()
    f.close()
    
    #print "stdout_value ", stdout_value
    #print "stderr_value ", stderr_value
    
    return index_file, threshold_file, localmax_txt_file
    
 
def copy_geom(infile_a, infile_b):
    """
    Method to call fsl fslcpgeom command to copy 
    certain parts of the header information (image dimensions, 
    voxel dimensions, voxel dimensions units string, image 
    orientation/origin or qform/sform info) from one image to another
    
    Parameters
    -----------
    infile_a : nifti file
        input volume from which the geometry is copied from
    
    infile_b : nifti file
       input volume from which the geometry is copied to
       
    Returns
    -------    
    out_file : nifti file
        Input volume infile_b with modified geometry information
        in the header.
        
    Raises
    ------
    Exception 
        If fslcpgeom fails
    
    """
    import subprocess as sb
    #try:
    out_file = infile_b
    cmd = sb.Popen(['fslcpgeom',
                    infile_a, out_file], stdin=sb.PIPE, stdout=sb.PIPE,)
    stdout_value, stderr_value = cmd.communicate()
    return out_file
    #except Exception:
    #    print "Error while using fslcpgeom to copy geometry"
    #    raise 




    

def get_standard_background_img(in_file, file_parameters):
    """
    Method to get the standard brain image from FSL 
    standard data directory, matching the resolution
    of a particular given input file.
    
    Parameters
    ----------
    in_file : nifti file
        Merged 4D Zmap volume
    file_parameters : tuple
       Value FSLDIR and MNI from config file
    
    Returns
    -------
    standard_path : string
        Standard FSL Image file path
    
    Raises
    ------
    Exception 
        If nibabel cannot load the input nifti volume
    
    """
    import os
    from nibabel import load
    #try:
    img = load(in_file)
    #hdr = img.get_header()
    hdr = img.header
    group_mm = int(hdr.get_zooms()[2])
    FSLDIR, MNI = file_parameters
    #print "group_mm -> ", group_mm
    standard_path = os.path.join(FSLDIR, 'data/standard/', '%s_T1_%smm_brain.nii.gz'% (MNI, group_mm))
    #print "path -> ", standard_path
    return os.path.abspath(standard_path)

#except Exception:
#print "Error while loading background image"
#        raise



#def get_standard_background_img(in_file,file_parameters):
#    """ Something """
#    import os
#    from nibabel import load
#    return "/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz"
    

def get_tuple(infile_a, infile_b):

    """
    Simple method to return tuple of z_threhsold
    maximum intensity values of Zstatistic image
    for input to the overlay.
    
    Parameters
    ----------
    z_theshold : float
        z threshold value
    intensity_stat : tuple of float values
        minimum and maximum intensity values
    
    Returns
    -------
    img_min_max : tuple (float)
        tuple of zthreshold and maximum intensity 
        value of z statistic image
    
    """
    out_file = (infile_a, infile_b[1])
    return out_file






def create_group_analysis(ftest=False, wf_name='groupAnalysis'):
    """
    FSL `FEAT <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FEAT>`_
    BASED Group Analysis

    Parameters
    ----------
    ftest : boolean, optional(default=False)
        Ftest help investigate several contrasts at the same time
        for example to see whether any of them (or any combination of them) is 
        significantly non-zero. Also, the F-test allows you to compare the 
        contribution of each contrast to the model and decide on significant 
        and non-significant ones
 
    wf_name : string 
        Workflow name
    
    Returns 
    -------
    grp_analysis : workflow object
        Group Analysis workflow object
    
    Notes
    -----
    `Source <https://github.com/openconnectome/C-PAC/blob/master/CPAC/group_analysis/group_analysis_preproc.py>`_
 
    Workflow Inputs::
        
        inputspec.mat_file : string (existing file)
           Mat file containing  matrix for design 
        
        inputspec.con_file : string (existing file)
           Contrast file containing contrast vectors 
        
        inputspec.grp_file : string (existing file)
           file containing matrix specifying the groups the covariance is split into
        
        inputspec.zmap_files : string (existing nifti file)
           derivative or the zmap file for which the group analysis is to be run
        
        inputspec.z_threshold : float
            Z Statistic threshold value for cluster thresholding. It is used to 
            determine what level of activation would be statistically significant. 
            Increasing this will result in higher estimates of required effect.
        
        inputspec.p_threshold : float
            Probability threshold for cluster thresholding.
            
        inputspec.fts_file : string (existing file)
           file containing matrix specifying f-contrasts
           
        inputspec.paramerters : string (tuple)
            tuple containing which MNI and FSLDIR path information
                      
    Workflow Outputs::
    
        outputspec.merged : string (nifti file)
            4D volume file after merging all the derivative 
            files from each specified subject.
            
        outputspec.zstats : list (nifti files)
            Z statistic image for each t contrast
            
        outputspec.zfstats : list (nifti files)
            Z statistic image for each f contrast
        
        outputspec.fstats : list (nifti files)
            F statistic for each contrast  
        
        outputspec.cluster_threshold : list (nifti files)
           the thresholded Z statistic image for each t contrast
        
        outputspec.cluster_index : list (nifti files)
            image of clusters for each t contrast; the values 
            in the clusters are the index numbers as used 
            in the cluster list.
        
        outputspec.cluster_localmax_txt : list (text files)
            local maxima text file for each t contrast, 
            defines the coordinates of maximum value in the cluster
        
        outputspec.overlay_threshold : list (nifti files)
            3D color rendered stats overlay image for t contrast
            After reloading this image, use the Statistics Color 
            Rendering GUI to reload the color look-up-table
        
        outputspec.overlay_rendered_image : list (nifti files)
           2D color rendered stats overlay picture for each t contrast
            
        outputspec.cluster_threshold_zf : list (nifti files)
           the thresholded Z statistic image for each f contrast
        
        outputspec.cluster_index_zf : list (nifti files)
            image of clusters for each f contrast; the values 
            in the clusters are the index numbers as used 
            in the cluster list.
            
        outputspec.cluster_localmax_txt_zf : list (text files)
            local maxima text file for each f contrast, 
            defines the coordinates of maximum value in the cluster
        
        outputspec.overlay_threshold_zf : list (nifti files)
            3D color rendered stats overlay image for f contrast
            After reloading this image, use the Statistics Color 
            Rendering GUI to reload the color look-up-table
        
        outputspec.overlay_rendered_image_zf : list (nifti files)
           2D color rendered stats overlay picture for each f contrast
    
    Order of commands:

    - Merge all the Z-map 3D images into 4D image file.  For details see `fslmerge <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Fslutils>`_::
    
        fslmerge -t sub01/sca/seed1/sca_Z_FWHM_merged.nii 
                    sub02/sca/seed1/sca_Z_FWHM.nii.gz ....  
                    merge.nii.gz
                    
        arguments 
            -t : concatenate images in time
            
    - Create mask specific for analysis. For details see `fslmaths <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Fslutils>`_::
    
        fslmaths merged.nii.gz 
                -abs -Tmin -bin mean_mask.nii.gz
        
        arguments 
             -Tmin  : min across time
             -abs   : absolute value
             -bin   : use (current image>0) to binarise
    
    - FSL FLAMEO to perform higher level analysis.  For details see `flameo <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FEAT>`_::
        
        flameo --copefile = merged.nii.gz --covsplitfile = anova_with_meanFD.grp --designfile = anova_with_meanFD.mat 
               --fcontrastsfile = anova_with_meanFD.fts --ld=stats --maskfile = mean_mask.nii.gz --runmode=ols 
               --tcontrastsfile = anova_with_meanFD.con
           
        arguments
            --copefile        : cope regressor data file
            --designfile      : design matrix file
            --maskfile        : mask file
            --tcontrastsfile  : file containing an ASCII matrix specifying the t contrasts
            --fcontrastsfile  : file containing an ASCII matrix specifying the f contrasts
            --runmode         : Interference to perform (mixed effects - OLS)
            
    - Run FSL Easy thresh 
        
      Easy thresh is a simple script for carrying out cluster-based thresholding and colour activation overlaying::
        
        easythresh <raw_zstat> <brain_mask> <z_thresh> <prob_thresh> <background_image> <output_root> [--mm]
      
      A seperate workflow called easythresh is called to run easythresh steps.
      
    High Level Workflow Graph:
    
    .. image:: ../images/group_analysis.dot.png
       :width: 800
    
    
    Detailed Workflow Graph:
    
    .. image:: ../images/group_analysis_detailed.dot.png
       :width: 800

    Examples
    --------
    
    >>> from group_analysis_preproc import create_group_analysis
    >>> preproc = create_group_analysis()
    >>> preproc.inputs.inputspec.mat_file = '../group_models/anova_with_meanFD/anova_with_meanFD.mat'
    >>> preproc.inputs.inputspec.con_file = '../group_models/anova_with_meanFD/anova_with_meanFD.con'
    >>> preproc.inputs.inputspec.grp_file = '../group_models/anova_with_meanFD/anova_with_meanFD.grp'
    >>> preproc.inputs.inputspec.zmap_files = ['subjects/sub01/seeds_rest_Dickstein_DLPFC/sca_Z_FWHM.nii.gz', 
                                               'subjects/sub02/seeds_rest_Dickstein_DLPFC/sca_Z_FWHM.nii.gz']
    >>> preproc.inputs.inputspec.z_threshold = 2.3
    >>> preproc.inputs.inputspec.p_threshold = 0.05
    >>> preproc.inputs.inputspec.parameters = ('/usr/local/fsl/', 'MNI152')
    >>> preproc.run()  -- SKIP doctest
            
    """
    grp_analysis = pe.Workflow(name=wf_name)

    inputnode = pe.Node(util.IdentityInterface(fields=['merged_file',
                                                       'merge_mask',
                                                       'mat_file',
                                                       'con_file',
                                                       'grp_file',
                                                       'fts_file',
                                                       'z_threshold',
                                                       'p_threshold',
                                                       'parameters']),
                         name='inputspec')

    outputnode = pe.Node(util.IdentityInterface(fields=['merged',
                                                        'zstats',
                                                        'zfstats',
                                                        'fstats',
                                                        'cluster_threshold',
                                                        'cluster_index',
                                                        'cluster_localmax_txt',
                                                        'overlay_threshold',
                                                        'rendered_image',
                                                        'cluster_localmax_txt_zf',
                                                        'cluster_threshold_zf',
                                                        'cluster_index_zf',
                                                        'overlay_threshold_zf',
                                                        'rendered_image_zf']),
                         name='outputspec')



    '''
    merge_to_4d = pe.Node(interface=fsl.Merge(),
                          name='merge_to_4d')
    merge_to_4d.inputs.dimension = 't'
    

    ### create analysis specific mask
    #-Tmin: min across time
    # -abs: absolute value
    #-bin: use (current image>0) to binarise
    merge_mask = pe.Node(interface=fsl.ImageMaths(),
                         name='merge_mask')
    merge_mask.inputs.op_string = '-abs -Tmin -bin'
    '''

    fsl_flameo = pe.Node(interface=fsl.FLAMEO(),
                         name='fsl_flameo')
    fsl_flameo.inputs.run_mode = 'ols'

    ### create analysis specific mask
    # fslmaths merged.nii.gz -abs -bin -Tmean -mul volume out.nii.gz
    #-Tmean: mean across time
    # create group_reg file
    # this file can provide an idea of how well the subjects 
    # in our analysis overlay with each other and the MNI brain.
    # e.g., maybe there is one subject with limited coverage.
    # not attached to sink currently 
    merge_mean_mask = pe.Node(interface=fsl.ImageMaths(),
                              name='merge_mean_mask')

    #function node to get the operation string for fslmaths command
    get_opstring = pe.Node(util.Function(input_names=['in_file'],
                                         output_names=['out_file'],
                                         function=get_operation),
                       name='get_opstring')

    #connections
    '''
    grp_analysis.connect(inputnode, 'zmap_files',
                         merge_to_4d, 'in_files')
    grp_analysis.connect(merge_to_4d, 'merged_file',
                         merge_mask, 'in_file')
    '''
    grp_analysis.connect(inputnode, 'merged_file',
                         fsl_flameo, 'cope_file')
    grp_analysis.connect(inputnode, 'merge_mask',
                         fsl_flameo, 'mask_file')
    grp_analysis.connect(inputnode, 'mat_file',
                         fsl_flameo, 'design_file')
    grp_analysis.connect(inputnode, 'con_file',
                         fsl_flameo, 't_con_file')
    grp_analysis.connect(inputnode, 'grp_file',
                         fsl_flameo, 'cov_split_file')

    if ftest:

        #calling easythresh for zfstats file
        grp_analysis.connect(inputnode, 'fts_file',
                             fsl_flameo, 'f_con_file')

        easy_thresh_zf = easy_thresh('easy_thresh_zf')

        grp_analysis.connect(fsl_flameo, 'zfstats',
                             easy_thresh_zf, 'inputspec.z_stats')
        grp_analysis.connect(inputnode, 'merge_mask',
                             easy_thresh_zf, 'inputspec.merge_mask')
        grp_analysis.connect(inputnode, 'z_threshold',
                             easy_thresh_zf, 'inputspec.z_threshold')
        grp_analysis.connect(inputnode, 'p_threshold',
                             easy_thresh_zf, 'inputspec.p_threshold')
        grp_analysis.connect(inputnode, 'parameters',
                             easy_thresh_zf, 'inputspec.parameters')
        grp_analysis.connect(easy_thresh_zf, 'outputspec.cluster_threshold',
                             outputnode, 'cluster_threshold_zf')
        grp_analysis.connect(easy_thresh_zf, 'outputspec.cluster_index',
                             outputnode, 'cluster_index_zf')
        grp_analysis.connect(easy_thresh_zf, 'outputspec.cluster_localmax_txt',
                             outputnode, 'cluster_localmax_txt_zf')
        grp_analysis.connect(easy_thresh_zf, 'outputspec.overlay_threshold',
                             outputnode, 'overlay_threshold_zf')
        grp_analysis.connect(easy_thresh_zf, 'outputspec.rendered_image',
                             outputnode, 'rendered_image_zf')

    #calling easythresh for zstats files
    easy_thresh_z = easy_thresh('easy_thresh_z')
    grp_analysis.connect(fsl_flameo, 'zstats',
                         easy_thresh_z, 'inputspec.z_stats')
    grp_analysis.connect(inputnode, 'merge_mask',
                         easy_thresh_z, 'inputspec.merge_mask')
    grp_analysis.connect(inputnode, 'z_threshold',
                         easy_thresh_z, 'inputspec.z_threshold')
    grp_analysis.connect(inputnode, 'p_threshold',
                         easy_thresh_z, 'inputspec.p_threshold')
    grp_analysis.connect(inputnode, 'parameters',
                         easy_thresh_z, 'inputspec.parameters')

    grp_analysis.connect(inputnode, 'merged_file',
                         get_opstring, 'in_file')
    grp_analysis.connect(inputnode, 'merged_file',
                         merge_mean_mask, 'in_file')
    grp_analysis.connect(get_opstring, 'out_file',
                         merge_mean_mask, 'op_string')

    grp_analysis.connect(fsl_flameo, 'zfstats',
                         outputnode, 'zfstats')
    grp_analysis.connect(fsl_flameo, 'fstats',
                         outputnode, 'fstats')
    grp_analysis.connect(inputnode, 'merged_file',
                         outputnode, 'merged')
    grp_analysis.connect(fsl_flameo, 'zstats',
                         outputnode, 'zstats')



    grp_analysis.connect(easy_thresh_z, 'outputspec.cluster_threshold',
                         outputnode, 'cluster_threshold')
    grp_analysis.connect(easy_thresh_z, 'outputspec.cluster_index',
                         outputnode, 'cluster_index')
    grp_analysis.connect(easy_thresh_z, 'outputspec.cluster_localmax_txt',
                         outputnode, 'cluster_localmax_txt')
    grp_analysis.connect(easy_thresh_z, 'outputspec.overlay_threshold',
                         outputnode, 'overlay_threshold')
    grp_analysis.connect(easy_thresh_z, 'outputspec.rendered_image',
                         outputnode, 'rendered_image')

    return grp_analysis



def get_operation(in_file):
    """
    Method to create operation string 
    for fslmaths
    
    Parameters
    ----------
    in_file : file
       input volume
    
    Returns
    -------
    op_string : string
        operation string for fslmaths
    
    Raises
    ------
    IOError
      If unable to load the input volume
    
    """
    #try:
    from nibabel import load
    img = load(in_file)
    hdr = img.get_header()
    n_vol = int(hdr.get_data_shape()[3])
    op_string = '-abs -bin -Tmean -mul %d' % (n_vol)
    return op_string
    #except:
    #    raise IOError("Unable to load the input nifti image")




    





def create_dummy_string(length):
    ppstring = ""
    for i in range(0, length):
        ppstring += '\t' + '%1.5e' %(1.0)
    ppstring += '\n' 
    return ppstring



def write_mat_file(design_matrix,
                   output_file):
    """ 
    Writes a design matrix file (.mat) for use with FSL FLAME.

    Butchered from C-PAC source code.

    Arguments
    design_matrix : a pandas DataFrame (with column names)
    output_file : the file to which we will write the output
    """
    
    import os
    import numpy as np

    dimx = None
    dimy = None

    # Ensure that it's actually a table
    if len(design_matrix.shape) == 1:
        dimy = 1
        dimx = design_matrix.shape[0]
    else:
        dimx, dimy = design_matrix.shape

    # Extract column names
    column_names = design_matrix.columns

    ppstring = '/PPheights'

    for i in range(0, dimy):
        ppstring += '\t' + '%1.5e' %(1.0)
    ppstring += '\n'

    out_file = output_file

    with open(out_file, 'wt') as f:

        print >>f, '/NumWaves\t%d' %dimy
        print >>f, '/NumPoints\t%d' %dimx
        print >>f, ppstring
        # print labels for the columns - mainly for double-checking your model
        col_string = '\n'
        for col in column_names:
            col_string = col_string + col + '\t'
        print >>f, col_string, '\n'
        print >>f, '/Matrix'
        np.savetxt(f, design_matrix, fmt='%1.5e', delimiter='\t')

    return out_file






def write_grp_file(grp_vector,filename):
    """
    Write the grouping file (.grp) for use with FSL FLAME. Butchered from CPAC.

    Arguments
    grp_vector : a grouping vector with the same length as the number of design matrix rows
    filename : the filename where we will save the grouping file
    """ 

    import os
    import numpy as np

    out_file = filename
    dimx = len(grp_vector)

    with open(out_file, "wt") as f:
        print >>f, '/NumWaves\t1'
        print >>f, '/NumPoints\t%d\n' %dimx
        print >>f, '/Matrix'
        np.savetxt(f, grp_vector, fmt='%d', delimiter='\t')

    return out_file











def write_ftest_file(fts_matrix,filename):
    """
    Write the f-tests file (.fts) for use with FSL FLAME. Butchered from CPAC.

    Arguments
    fts_matrix : an f-test matrix, where each row corresponds to one contrast, and the columns are the different f tests. each element should be zero or one (inclusion)
    filename : the filename where we will save the f test file
    """ 
    
    import os
    import numpy as np
    
    out_file = filename

    contrast_names = fts_matrix["Contrast"] # the contrast name column
    fts_matrix = fts_matrix.drop("Contrast",1) # then drop the column

    with open(out_file,"wt") as f:

        print >>f, '/NumWaves\t',     (fts_matrix.shape)[0]
        print >>f, '/NumContrasts\t', (fts_matrix.shape)[1]
        f.write('\n')

        # print labels for the contrasts - mainly for double-checking your model
        f.write('\t'.join(contrast_names))
        f.write('\n\n')

        print >>f, '/Matrix'
        np.savetxt(f, np.matrix(fts_matrix).T,
                   fmt='%i',
                   delimiter=' ')


    return out_file










def write_con_file(con_matrix,filename):
    """
    Write the contrasts file (.con) for use with FSL FLAME. Butchered from CPAC.

    Arguments
    con_matrix : a contrast matrix, where each row corresponds to one contrast, and the columns should be the same as the design matrix (this will NOT be checked here though)
    filename : the filename where we will save the contrast file
    """ 
    
    import os
    import numpy as np
    
    out_file = filename

    contrast_names = con_matrix["Contrast"] # the contrast name column
    con_matrix = con_matrix.drop("Contrast",1) # then drop the column

    regressors = con_matrix.columns

    with open(out_file,"wt") as f:

        idx = 1
        pp_str = '/PPheights'
        re_str = '/RequiredEffect'
        for name in contrast_names:

            print >>f, '/ContrastName%d' %idx, '\t', name
            pp_str += '\t%1.5e' %(1)
            re_str += '\t%1.5e' %(1)
            idx += 1

        print >>f, '/NumWaves\t', (con_matrix.shape)[1]
        print >>f, '/NumContrasts\t', (con_matrix.shape)[0]
        print >>f, pp_str
        print >>f, re_str + '\n'

        # print labels for the columns - mainly for double-checking your model
        col_string = '\n'
        for ev in regressors:
            col_string = col_string + ev + '\t'
        print >>f, col_string, '\n'

        print >>f, '/Matrix'
        np.savetxt(f, con_matrix, fmt='%1.5e', delimiter='\t')


    return out_file









def parse_ftest_csv(ftest_fname,contrasts):
    """ 
    Parse the contents of a CSV specifying contrasts, and perform some necessary checks.
    If all is well then return the contrast table.
    """

    import pandas as pd
    ftests = pd.DataFrame.from_csv(ftest_fname).reset_index()

    if ftests.columns[0]!="Contrast":
        print("The first column of the f-test file should be called 'Contrast'")
        assert False
    fnames = ftests.columns[1:]# Note the [1:] because we ignore the first column (contrast name)

    # The contrasts as defined in the f test file
    f_contrasts = [ c["Contrast"] for _,c in ftests.iterrows() ]

    #c_contrasts = contrasts.columns[1:]# Note the [1:] because we ignore the first column (contrast name)
    
    c_contrasts = [ c["Contrast"] for _,c in contrasts.iterrows() ]
    
    if len(f_contrasts)!=len(c_contrasts) or not (f_contrasts==c_contrasts):
        print("The rows of the f test file (the contrasts) do not match the contrasts defined in the contrast file.")
        print("F test columns: %s"%(",".join(f_contrasts)))
        print("Contrast matrix contrasts: %s"%(",".join(c_contrasts)))
        assert False

    return ftests





def parse_contrast_csv(contrast_fname,design_matrix):
    """ 
    Parse the contents of a CSV specifying contrasts, and perform some necessary checks.
    If all is well then return the contrast table.
    """

    import pandas as pd
    contrasts = pd.DataFrame.from_csv(contrast_fname).reset_index()

    if contrasts.columns[0]!="Contrast":
        print("The first column of the contrasts matrix should be called 'Contrast'")
        assert False

    ccols = contrasts.columns[1:]# Note the [1:] because we ignore the first column (contrast name)
    dcols = design_matrix.columns
    if len(ccols)!=len(dcols) or not (ccols==dcols).all():
        print("The columns of the contrasts file do not match those in the design matrix.")
        print("Contrasts file columns: %s"%(",".join(contrasts.columns[1:])))
        print("Design matrix columns: %s"%(",".join(design_matrix.columns)))
        assert False

    return contrasts






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

