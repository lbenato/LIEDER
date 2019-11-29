# LIEDER
Long lIvEd Displaced jEt classifieR

First draft of a Fully Connected Neural Network aiming at event-wise binary classification (signal vs background) for searches for long lived particles with CMS detector.

Input and output root files are designed to be compatible with LLP repository.

## Preliminary naf gpu setup
See: https://confluence.desy.de/pages/viewpage.action?spaceKey=UHHML&title=Using+GPUs+in+naf

## write_pd_skim.py
It reads input root files and transforms them into h5 files.

## samples.py
Same as LLP repo; used to load sample names.

##dnn_functions.py
Simple function to draw training and validation losses and accuracies.

##dnn.py
- Split training/test samples for both signal and background
- Transform them into h5
- Training function
- Calculate performances
- Write the output scores of test samples
- Convert h5 back to root files, compatible with any macro of LLP repository