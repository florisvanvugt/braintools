
# Various brain-imaging related tools

## QC builder

This takes an output tree of the subject-level analysis (probably from Nipype) and returns a script that will run the usual bare-minimum quality control steps:

* Checking alignment of mean functional image to subject's anatomical
* Checking alignment of subject's anatomical to the standard anatomical (MNI) template
* Build a table of motion correction.


## Usage

Edit `qc_pianists.yaml` to define where data is to be found and what the names of the folders are that you expect.

Then run `python make_qc.py qc_pianists.yaml`. 





## SCA Nipype Reporter

The purpose of this tool is to generate an overview for a seed-based correlation analysis (SCA). I have in mind to cater to SCA analysis generated in nipype but perhaps it's flexible enough to apply this to other analysis frameworks too.

### Requirements

Probably some things, yes.

This worked with Python 2.7 and Nipype 0.13.1. Need Seaborn as well. 

You also need Imagemagick installed (being able to run `convert` on Linux).



### Usage

Edit `sca_specs.yaml` to define where data is to be found and what the names of the folders are that you expect.

Then run `python sca_nipype_reporter.py sca_specs.yaml`. 



### TODO
- [x] Make this work for repeated-measures designs: each subject may have multiple datapoints (so multiple occurrences in the `merged` files).
- [x] Subject names in the scatter plots
- [ ] Actually check correctness of scatter plots
- [x] Seed names in overviews
- [x] Update fsleyes run command file names
- [x] If you enter None as a variable then just plot all points on the same value on the x axis.
- [ ] Plot design matrix? At least I would say add it as a table, joined with pheno data
- [x] Include aux files
- [ ] Fix size of large image for each seed
- [ ] Add error bars to bar plots?
- [x] Gathering output as well?
- [ ] When gathering output files then keep track of whether we have already output that time before (not whether it already exists)



### Troubleshooting
I had some trouble with slicer not giving the correct output. I think this related to a broken FSL installation.


On colossus, I needed
```
source deactivate
bash
/usr/bin/python etc. etc. 
```

