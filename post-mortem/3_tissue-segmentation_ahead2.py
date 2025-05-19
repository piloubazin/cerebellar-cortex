import nighres
import os
import nibabel
import numpy
import glob
import scipy.ndimage
import ants

indir = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/ahead2/coreg/'
outdir = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/ahead2/tissues/'

#bfimg = indir+'Ahead_brain_122017_blockface-image_cr_def-img.nii.gz'
thio = indir+'Ahead_brain_152017_thionin-interpolated_cr_def-img.nii.gz'
#parv = indir+'Ahead_brain_122017_parvalbumin-interpolated_cr_def-img.nii.gz'
silv = indir+'Ahead_brain_152017_Bielschowsky-interpolated_cr_def-img.nii.gz'

cbprior = indir+'Ahead_brain_152017_blockface-image_cr_cbmask_def-img.nii.gz'
cbwmprior = indir+'Ahead_brain_152017_blockface-image_cr_cbwmmask_def-img.nii.gz'


recompute=False

img = nighres.io.load_volume(thio)
    
# crop outside region for now (TODO: make levelset, crop from some distance outside)
cbsurf = nighres.surface.probability_to_levelset(cbprior, mask_image=None,
                            save_data=True, overwrite=False, output_dir=outdir)['result']

# mask at 30 voxels (6mm) from the p=0.5 boundary of the MNI cb prior 
# (very generous, but needed given the deformations of post-mortem data)
cbmask = nighres.io.load_volume(cbsurf).get_fdata()<30
cbmsk = cbsurf.replace('.nii','_msk.nii').replace(indir,outdir)
if recompute or not os.path.exists(cbmsk):
    img = nibabel.Nifti1Image(cbmask,img.affine,img.header)
    nighres.io.save_volume(cbmsk,img)


#bfmsk = bfimg.replace('.nii','_msk.nii').replace(indir,outdir)
#if recompute or not os.path.exists(bfmsk):
#    img = nibabel.Nifti1Image(cbmask*numpy.nan_to_num(img.get_fdata()),img.affine,img.header)
#    nighres.io.save_volume(bfmsk,img)

recompute=False

thiomsk = thio.replace('.nii','_msk.nii').replace(indir,outdir)
if recompute or not os.path.exists(thiomsk):
    img = nibabel.Nifti1Image(cbmask*img.get_fdata(),img.affine,img.header)
    nighres.io.save_volume(thiomsk,img)


recompute=False

silvmsk = silv.replace('.nii','_msk.nii').replace(indir,outdir)
if recompute or not os.path.exists(silvmsk):
    img = nighres.io.load_volume(silv)
    img = nibabel.Nifti1Image(cbmask*numpy.nan_to_num(img.get_fdata()),img.affine,img.header)
    nighres.io.save_volume(silvmsk,img)

# N4 correction? not improving anything...
silvn4 = silvmsk.replace('.nii','_n4.nii')
if recompute or not os.path.exists(silvn4):
    image = ants.image_read(silvmsk)
    mask = ants.image_read(cbmsk)
    n4img = ants.n4_bias_field_correction(image,mask=mask,verbose=True)
    ants.image_write(n4img,silvn4)

# slice-based intensity adjustments?
silvsir = nighres.microscopy.stack_intensity_regularisation(silvmsk, cutoff=50, mask=cbmsk,
                            save_data=True, overwrite=False, output_dir=outdir)['result']

# N4 correction after adjustment? looks pretty good
silvn4 = silvsir.replace('.nii','_n4.nii')
if recompute or not os.path.exists(silvn4):
    image = ants.image_read(silvsir)
    mask = ants.image_read(cbmsk)
    n4img = ants.n4_bias_field_correction(image,mask=mask,verbose=True)
    ants.image_write(n4img,silvn4)

silvmsk = silvn4

#parvmsk = parv.replace('.nii','_msk.nii').replace(indir,outdir)
#if recompute or not os.path.exists(parvmsk):
#    img = nighres.io.load_volume(parv)
#    img = nibabel.Nifti1Image(cbmask*numpy.nan_to_num(img.get_fdata()),img.affine,img.header)
#    nighres.io.save_volume(parvmsk,img)

# tests: supervoxels not helpful (too much driven by residual noise)
# tests: FCM works better with 4 classes, to model the spread of GM values and inhomogeneities

# a combination of FCM to get large WM, GM, CSF regions and ridge filtering to get the fine branches
# seems promising


fcm = nighres.segmentation.fuzzy_cmeans(thiomsk, clusters=5, max_iterations=50, max_difference=0.01, 
                    smoothing=0.1, fuzziness=2.0, mask_zero=True, map_intensity=True,
                    save_data=True, overwrite=False, output_dir=outdir, 
                    file_name=os.path.basename(thiomsk).replace('.nii','_cl5.nii'))

fcm2 = nighres.segmentation.fuzzy_cmeans(silvmsk, clusters=6, max_iterations=50, max_difference=0.01, 
                    smoothing=0.1, fuzziness=2.0, mask_zero=True, map_intensity=True,
                    save_data=True, overwrite=False, output_dir=outdir, 
                    file_name=os.path.basename(silvmsk).replace('.nii','_cl6.nii'))

# stop and check here

allbranch = nighres.filtering.recursive_ridge_diffusion(thiomsk, ridge_intensities='dark', ridge_filter='2D',
                              surface_levelset=None, orientation='undefined',
                              loc_prior=None,
                              min_scale=0, max_scale=0,
                              diffusion_factor=0.95,
                              similarity_scale=0.1,
                              max_iter=100, max_diff=1e-3,
                              threshold=0.5, rescale=False,
                              save_data=True, overwrite=False, output_dir=outdir,
                              file_name=os.path.basename(thiomsk).replace('.nii','_wm095.nii'))['propagation']

wmbranch = nighres.filtering.recursive_ridge_diffusion(silvmsk, ridge_intensities='bright', ridge_filter='2D',
                              surface_levelset=None, orientation='undefined',
                              loc_prior=None,
                              min_scale=0, max_scale=0,
                              diffusion_factor=0.95,
                              similarity_scale=0.1,
                              max_iter=100, max_diff=1e-3,
                              threshold=0.5, rescale=False,
                              save_data=True, overwrite=False, output_dir=outdir,
                              file_name=os.path.basename(silvmsk).replace('.nii','_wm095.nii'))['propagation']

# results are too fine for proper surface reconstruction, we need to add some artificial PV on both WM ans CSF
allthick = nighres.intensity.intensity_propagation(allbranch, mask=None, combine='max', distance_mm=0.4,
                      target='lower', scaling=0.5, domain=None,
                      save_data=True, overwrite=False, output_dir=outdir)['result']

wmthick = nighres.intensity.intensity_propagation(wmbranch, mask=None, combine='max', distance_mm=0.4,
                      target='lower', scaling=0.5, domain=None,
                      save_data=True, overwrite=False, output_dir=outdir)['result']

# prior to remove cortical GM: not used so far, let's see what topology correction brings
nocbprior = nighres.surface.levelset_to_probability(cbsurf, distance_mm=-2,
                                    save_data=True, overwrite=False, output_dir=outdir)['result']

cbwmsurf = nighres.surface.probability_to_levelset(cbwmprior, mask_image=None,
                            save_data=True, overwrite=False, output_dir=outdir)['result']

cbwmprior = nighres.surface.levelset_to_probability(cbwmsurf, distance_mm=2,
                                    save_data=True, overwrite=False, output_dir=outdir)['result']


recompute=True

# wm proba: fcm + *connected* ridges

# WM = fcm 2,3 + *connected* dark ridges
# GM = fcm 4,5
# CSF = fcm 1 + connected dark ridges

# GM is good, WM includes too much stuff outside -> start from pial and erode?

# start from GM and CSF, then grow interface with probability diffusionon the WM class
# -> WM+GM vs CSF

gmprior = thiomsk.replace('.nii','_gmrawprior.nii')
if recompute or not os.path.exists(gmprior):
    gmproba = nighres.io.load_volume(fcm['memberships'][3]).get_fdata()\
             +nighres.io.load_volume(fcm['memberships'][4]).get_fdata()
    branchpv = nighres.io.load_volume(allbranch).get_fdata()
    nocb = nighres.io.load_volume(nocbprior).get_fdata()
    gmproba = nibabel.Nifti1Image(cbmask*(1.0-numpy.minimum(nocb*(1.0+branchpv),numpy.ones(gmproba.shape)))\
                                *numpy.maximum(gmproba-branchpv,numpy.zeros(gmproba.shape)),
                                img.affine,img.header)
    #gmproba = nibabel.Nifti1Image(cbmask*numpy.maximum(gmproba-branchpv,numpy.zeros(gmproba.shape)),
    #                            img.affine,img.header)
    nighres.io.save_volume(gmprior,gmproba)
    gmproba = gmproba.get_fdata()
else:
    gmproba = nighres.io.load_volume(gmprior).get_fdata()

# for wm as well? or just take the rest?
wmprior = thiomsk.replace('.nii','_wmrawprior.nii')
if recompute or not os.path.exists(wmprior):
    wmproba = nighres.io.load_volume(fcm2['memberships'][4]).get_fdata()\
             +nighres.io.load_volume(fcm2['memberships'][5]).get_fdata()
    # remove csf influence in likely WM region?
    cbwm = nighres.io.load_volume(cbwmprior).get_fdata()
    wmproba = wmproba+cbwm*nighres.io.load_volume(fcm2['memberships'][3]).get_fdata()
             
    branchpv = nighres.io.load_volume(wmbranch).get_fdata()
    nobranch = nighres.io.load_volume(allbranch).get_fdata()
    nocb = nighres.io.load_volume(nocbprior).get_fdata()
    wmproba = nibabel.Nifti1Image(cbmask*(1.0-numpy.minimum(nocb*(1.0+nobranch),numpy.ones(gmproba.shape)))\
                                   *numpy.maximum(numpy.minimum(wmproba+branchpv-gmproba,\
                                    numpy.ones(wmproba.shape)),numpy.zeros(img.shape)),\
                                    img.affine,img.header)
    nighres.io.save_volume(wmprior,wmproba)
    wmproba = wmproba.get_fdata()
else:
    wmproba = nighres.io.load_volume(wmprior).get_fdata()


# same for csf
csfprior = thiomsk.replace('.nii','_csfrawprior.nii')
if recompute or not os.path.exists(csfprior):
    csfproba = nibabel.Nifti1Image(cbmask*numpy.maximum(1.0-wmproba-gmproba,numpy.zeros(img.shape)),
                                    img.affine,img.header)
    nighres.io.save_volume(csfprior,csfproba)
    csfproba = csfproba.get_fdata()
else:
    csfproba = nighres.io.load_volume(csfprior).get_fdata()




recompute=True

# wm proba: fcm + *connected* ridges

# WM = fcm 2,3 + *connected* dark ridges
# GM = fcm 4,5
# CSF = fcm 1 + connected dark ridges

# GM is good, WM includes too much stuff outside -> start from pial and erode?

# start from GM and CSF, then grow interface with probability diffusionon the WM class
# -> WM+GM vs CSF

gmprior = thiomsk.replace('.nii','_gmprior.nii')
if recompute or not os.path.exists(gmprior):
    gmproba = nighres.io.load_volume(fcm['memberships'][3]).get_fdata()\
             +nighres.io.load_volume(fcm['memberships'][4]).get_fdata()
    branchpv = nighres.io.load_volume(allthick).get_fdata()
    nocb = nighres.io.load_volume(nocbprior).get_fdata()
    gmproba = nibabel.Nifti1Image(cbmask*(1.0-numpy.minimum(nocb*(1.0+branchpv),numpy.ones(gmproba.shape)))\
                                *numpy.maximum(gmproba-branchpv,numpy.zeros(gmproba.shape)),
                                img.affine,img.header)
    #gmproba = nibabel.Nifti1Image(cbmask*numpy.maximum(gmproba-branchpv,numpy.zeros(gmproba.shape)),
    #                            img.affine,img.header)
    nighres.io.save_volume(gmprior,gmproba)
    gmproba = gmproba.get_fdata()
else:
    gmproba = nighres.io.load_volume(gmprior).get_fdata()

# for wm as well? or just take the rest?
wmprior = thiomsk.replace('.nii','_wmprior.nii')
if recompute or not os.path.exists(wmprior):
    wmproba = nighres.io.load_volume(fcm2['memberships'][3]).get_fdata()\
             +nighres.io.load_volume(fcm2['memberships'][4]).get_fdata()
    # remove csf influence in likely WM region?
    cbwm = nighres.io.load_volume(cbwmprior).get_fdata()
    wmproba = wmproba+cbwm*nighres.io.load_volume(fcm2['memberships'][3]).get_fdata()

    branchpv = nighres.io.load_volume(wmthick).get_fdata()
    nobranch = nighres.io.load_volume(allthick).get_fdata()
    nocb = nighres.io.load_volume(nocbprior).get_fdata()
    wmproba = nibabel.Nifti1Image(cbmask*(1.0-numpy.minimum(nocb*(1.0+nobranch),numpy.ones(gmproba.shape)))\
                                   *numpy.maximum(numpy.minimum(wmproba+branchpv-gmproba,\
                                    numpy.ones(img.shape)),numpy.zeros(img.shape)),\
                                    img.affine,img.header)
    #wmproba = nibabel.Nifti1Image(cbmask*numpy.maximum(numpy.minimum(wmproba+branchpv-gmproba,\
    #                                numpy.ones(wmproba.shape)),numpy.zeros(img.shape)),\
    #                                img.affine,img.header)
    nighres.io.save_volume(wmprior,wmproba)
    wmproba = wmproba.get_fdata()
else:
    wmproba = nighres.io.load_volume(wmprior).get_fdata()


# same for csf
csfprior = thiomsk.replace('.nii','_csfprior.nii')
if recompute or not os.path.exists(csfprior):
    csfproba = nibabel.Nifti1Image(cbmask*numpy.maximum(1.0-wmproba-gmproba,numpy.zeros(img.shape)),
                                    img.affine,img.header)
    nighres.io.save_volume(csfprior,csfproba)
    csfproba = csfproba.get_fdata()
else:
    csfproba = nighres.io.load_volume(csfprior).get_fdata()

