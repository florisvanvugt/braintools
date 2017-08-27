
# SCA Nipype Reporter

The purpose of this tool is to generate an overview for a seed-based correlation analysis (SCA). I have in mind to cater to SCA analysis generated in nipype but perhaps it's flexible enough to apply this to other analysis frameworks too.

## Requirements

Probably some things, yes.

This worked with Python 2.7 and Nipype 0.13.1. Need Seaborn as well.


## Usage

Edit `specs.yaml` to define where data is to be found and what the names of the folders are that you expect.

Then run `python sca_nipype_reporter.py specs.yaml`. 



## TODO
- [ ] Make this work for repeated-measures designs: each subject may have multiple datapoints (so multiple occurrences in the `merged` files).

