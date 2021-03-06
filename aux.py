import os
import subprocess
import numpy as np



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


def gather(filename,target,info,check_not_exist=False):
    """
    Gather a particular filename, i.e. copy it away from its
    embedded location somewhere deep in the results folder,
    and put it somewhere with a reasonable filename.

    if check_not_exist we will check whether the target file does not already
    exist.
    """
    if not os.path.exists(info["gather_dir"]):
        os.makedirs(info["gather_dir"])

    if not os.path.exists(filename):
        print("## Error: file %s does not exist ##"%filename)
        assert False

    target = pj(info["gather_dir"],target)
    if check_not_exist:
        if os.path.exists(target):
            print("Error -- gather target %s already exists"%target)
            assert False
        
        
    print("%s -> %s"%(filename,target))
    
    cmd = ["cp",filename,target]
    subprocess.call(cmd)

    return target




def exists(info,item):
    """ Check whether a particular item exists and is not-None in the configuration dict. """
    return item in info and info[item] not in [None,"","None"]
    

