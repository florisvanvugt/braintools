

import matplotlib as mpl
mpl.rcParams['backend'] = 'Agg'

import matplotlib.pyplot as plt
import numpy as np

import nibabel as nib







def get_grid_matrix(data,grid):
    """ Given a 3d matrix, return it as a grid,
    where the grid table tells us the slice we want
    to plot on each cell."""
    
    n,m,_=data.shape
    dummy = np.array([np.nan]*n*m).reshape(m,n) # make a dummy slice

    def get_slice(i):
        if i==-1: return dummy
        #print(i)
        return data[:,:,i].T
    
    # First stack the columns in each row
    rows = [ np.concatenate([ get_slice(slice_i) for slice_i in row[::-1] ],axis=1) for row in grid ]
    
    # Then stack the columns
    stacked = np.concatenate(rows, axis=0 )

    return stacked




def mapcolor(d,scalecolours,minval=None,maxval=None,alpha=1.,colour_range=[None,None]):
    """ 
    Map a 2D matrix to a 3D-matrix where each pixel encodes desired RGBA colour.
    Scale colours are the colours for the minimum and maximum values.
    
    Arguments
    d : input matrix of values
    scalecolours : colours of the extremities
    alpha : alpha of the image as a whole (multiplier)
    minval,maxval : the range of values to plot (anything outside will not be plot)
    """
    if minval==None: minval=np.nanmin(d)
    if maxval==None: maxval=np.nanmax(d)
    assert len(scalecolours)==2 # for now, just minimum and maximum colour please
    mincol = np.array(scalecolours[0])
    maxcol = np.array(scalecolours[1])
    diffcol = maxcol-mincol

    # Determine the values where we will plot between
    assert len(colour_range)==2
    colminval = colour_range[0]
    colmaxval = colour_range[1]
    colminval = minval if colminval==None else colminval
    colmaxval = maxval if colmaxval==None else colmaxval
    
    def rgba(i):
        """ Map an individual value to RGBA """
        #print(i)
        if np.isnan(i) or i<minval or i>maxval: return (0,0,0,0) # show nothin' (transparent)
        lumin = (i-colminval)/float(colmaxval-colminval) # normalise (but the value could be outside the range
        if lumin<0: lumin=0
        if lumin>1: lumin=1 # ensure we are back in plottable range
        return np.concatenate([mincol + (diffcol*lumin),[alpha]])
    nrow,ncol = d.shape
    return [ [ rgba(d[i,j]) for i in range(nrow) ] for j in range(ncol) ]



def colorname(col):
    """ Return RGB for a particular colour name."""
    if col==None: return [1,1,1]
    col = col.lower().strip()
    if col=="red": return [1,0,0]
    if col=="blue": return [0,0,1]
    if col=="green": return [0,1,0]
    if col=="white": return [1,1,1]
    if col=="black": return [0,0,0]
    if col=="yellow": return [1,1,0]
    return [1,1,1] # white if nothing else
    
    
def interpret_colour(col):
    """
    Try to interpret the argument as a colour, ideally as a colour range
    """
    if col==None:     return [np.zeros(3),np.ones(3)] # RGB
    colpair =col.split("-")
    if len(colpair)==2:
        return [colorname(c) for c in colpair]
    if col=="red":  return [np.array([1.,0,0]),np.array([1.,0,0])]
    if col=="blue": return [np.array([0,0,1.]),np.array([0,0,1.])]
    return [np.zeros(3),np.ones(3)] # RGB




def overlay_grid(overlays,output_specs):


    # First determine the data shapes (must all be the same)
    images = [ nib.load(o["file"]) for o in overlays ]
    matrices = [ img.get_data() for img in images ]
    shapes = [ mat.shape for mat in matrices ]

    shape = shapes[0]

    assert all([ s==shape for s in shapes ])

    # Now determine which slices we will take to make it fit with the grid we are given
    n_slices = shape[2]
    rows,cols = output_specs["grid"]
    max_slices = rows*cols # the maximum number of slices we can make
    slice_step = int(np.ceil(n_slices/max_slices))



    # Now determine the grid, in each cell, which slice will we show?
    slices = list(range(0,n_slices,slice_step))[::-1]
    slicegrid = np.array(slices+[-1]*(rows*cols-len(slices))).reshape((rows,cols)) # fill up with NAN and make into a grid 
    slicegrid


    fig = plt.figure()
    fig.patch.set_facecolor('black')

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    for (overlay,mat) in zip(overlays,matrices):
        #print("Overlay %s"%overlay)
        bigmat = get_grid_matrix(mat,slicegrid)

        # See if we need to multiply by anything
        bigmat = bigmat*overlay.get("multiply",1)

        # Map the voxel values to colours
        color = interpret_colour(overlay.get("colour",None))

        colormat = mapcolor(bigmat.T,scalecolours=color,
                            minval=overlay.get("threshold",None),
                            alpha=overlay.get('alpha',1),
                            colour_range=overlay.get('colrange',[None,None])
        )

        # Now stack the images so that we get a large matrix
        ax.imshow(colormat, origin='lower',interpolation='none',aspect='equal')

    # Compute the width/height of the plot
    h,w = bigmat.shape

    output_specs["dpi"]=output_specs.get('dpi',150) # set default DPI
    if   "width"  in output_specs: output_specs['height']=output_specs['width']*(h/w)
    elif "height" in output_specs: output_specs['width'] =output_specs['height']*(w/h)

    plt.axis('off')
    fig.set_size_inches( (output_specs["width"]/output_specs["dpi"],output_specs["height"]/output_specs["dpi"]))

    fig.savefig(output_specs["file"],dpi=output_specs["dpi"],frameon=True,transparent=False,facecolor='black')
    plt.close(fig)
    return














if __name__=="__main__":


    tresh = .1
    colorrange = [.1,.8] # the range for the colours
    
    # Example usage
    layers = [
        # Contains a listing of the various overlays, from bottom to top

        {"file":"/usr/share/data/fsl-mni152-templates/MNI152_T1_2mm.nii.gz",
         "alpha":.7 # makes the layer transparent (background is black)
        },
        {"file":"FC-zmap.nii.gz", # the nifti file for the overlay; has to exist of course
         "threshold":.1,         # plot only voxels that cross the threshold (signed)
         "colour":"red-yellow", # red-yellow means that the colours will go from red to yellow
         "colrange":[.1,.8]  # means that .1 is mapped to red, .8 (and anything above) is mapped to yellow, and in between it's a trade-off
        },
        {"file":"FC-zmap.nii.gz",
         "multiply":-1, # means the map is inverted (applies before thresholding and colour coding)
         "threshold":.1,"colour":"blue-white","colrange":[.1,.8]},
    ]



            
    output_specs = {
        "file"  : 'example.png', # the file we will output to
        "grid"  : (4,8), # tells us that we want 4 rows and 8 columns.
        "dpi"   : 150,   # resolution of the output image
        "width" : 800 # tells us that we want a particular output width (you can also specify a height instead)
    }
    

    overlay_grid(layers,output_specs)
