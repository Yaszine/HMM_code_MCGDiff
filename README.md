# HMM_code_MCGDiff

Code for sampling with MCGdiff algorithm the posterior of GMM where d_y dimension has been masked.
The output of the code are .npz files containing the true posterior, the prior, MCG_diff samples and other parameters.
The folder that contains the .npz needs to be changed. 
# Choice of Parameters

3000 particles
20,100,200 steps 
d_y between [1,2,4]
d_x between [8,80,800]
Use of seeds because the model can degenerate so the choice of sigma_y is vast.
