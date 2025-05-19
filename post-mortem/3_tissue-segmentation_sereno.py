import nighres
import os
import nibabel
import numpy
import glob
import scipy.ndimage
import ants

indir = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/sereno/coreg/'
outdir = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/sereno/tissues/'

t1w = indir+'serenocb20_T1_cr_def-img.nii.gz'
t2w = indir+'serenocb20_T2_cr_def-img.nii.gz'
pdw = indir+'serenocb20_PD_cr_def-img.nii.gz'

cbprior = indir+'serenocb20_T1_cr_cbmask_def-img.nii.gz'


recompute=False

img = nighres.io.load_volume(t1w)
    
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

# N4 correction? not really useful
t1w_n4 = t1w.replace('.nii','_n4.nii').replace(indir,outdir)
if recompute or not os.path.exists(t1w_n4):
    image = ants.image_read(t1w)
    n4img = ants.n4_bias_field_correction(image,verbose=True)
    ants.image_write(n4img,t1w_n4)

t2w_n4 = t2w.replace('.nii','_n4.nii').replace(indir,outdir)
if recompute or not os.path.exists(t2w_n4):
    image = ants.image_read(t2w)
    n4img = ants.n4_bias_field_correction(image,verbose=True)
    ants.image_write(n4img,t2w_n4)

pdw_n4 = pdw.replace('.nii','_n4.nii').replace(indir,outdir)
if recompute or not os.path.exists(pdw_n4):
    image = ants.image_read(pdw)
    n4img = ants.n4_bias_field_correction(image,verbose=True)
    ants.image_write(n4img,pdw_n4)


# possibly some preprocessing, we'll see

t1fcm = nighres.segmentation.fuzzy_cmeans(t1w_n4, clusters=5, max_iterations=50, max_difference=0.01, 
                    smoothing=0.1, fuzziness=2.0, mask_zero=True, map_intensity=True,
                    save_data=True, overwrite=False, output_dir=outdir, 
                    file_name=os.path.basename(t1w_n4).replace('.nii','_cl5.nii'))

t2fcm = nighres.segmentation.fuzzy_cmeans(t2w_n4, clusters=5, max_iterations=50, max_difference=0.01, 
                    smoothing=0.1, fuzziness=2.0, mask_zero=True, map_intensity=True,
                    save_data=True, overwrite=False, output_dir=outdir, 
                    file_name=os.path.basename(t2w_n4).replace('.nii','_cl5.nii'))

pdfcm = nighres.segmentation.fuzzy_cmeans(pdw_n4, clusters=5, max_iterations=50, max_difference=0.01, 
                    smoothing=0.1, fuzziness=2.0, mask_zero=True, map_intensity=True,
                    save_data=True, overwrite=False, output_dir=outdir, 
                    file_name=os.path.basename(pdw_n4).replace('.nii','_cl5.nii'))

# stop and check here

wmbranch = nighres.filtering.recursive_ridge_diffusion(t1w_n4, ridge_intensities='bright', ridge_filter='2D',
                              surface_levelset=None, orientation='undefined',
                              loc_prior=None,
                              min_scale=0, max_scale=0,
                              diffusion_factor=0.95,
                              similarity_scale=0.1,
                              max_iter=100, max_diff=1e-3,
                              threshold=0.5, rescale=False,
                              save_data=True, overwrite=False, output_dir=outdir,
                              file_name=os.path.basename(t1w_n4).replace('.nii','_wm.nii'))['propagation']

csfbranch = nighres.filtering.recursive_ridge_diffusion(t1w_n4, ridge_intensities='dark', ridge_filter='2D',
                              surface_levelset=None, orientation='undefined',
                              loc_prior=None,
                              min_scale=0, max_scale=0,
                              diffusion_factor=0.95,
                              similarity_scale=0.1,
                              max_iter=100, max_diff=1e-3,
                              threshold=0.5, rescale=False,
                              save_data=True, overwrite=False, output_dir=outdir,
                              file_name=os.path.basename(t1w_n4).replace('.nii','_csf.nii'))['propagation']

duraprior = nighres.surface.levelset_to_probability(cbsurf, distance_mm=-2,
                            save_data=True, overwrite=False, output_dir=outdir)['result']

dura = nighres.filtering.recursive_ridge_diffusion(t1w_n4, ridge_intensities='bright', ridge_filter='2D',
                              surface_levelset=cbsurf, orientation='parallel',
                              loc_prior=duraprior,
                              min_scale=0, max_scale=0,
                              diffusion_factor=0.95,
                              similarity_scale=0.1,
                              max_iter=100, max_diff=1e-3,
                              threshold=0.5, rescale=False,
                              save_data=True, overwrite=False, output_dir=outdir,
                              file_name=os.path.basename(t1w_n4).replace('.nii','_dura.nii'))['propagation']
        
# combine all information: T1 bright + ridges for WM, + T1 mid-class away from dura prior

# results are too fine for proper surface reconstruction, we need to add some artificial PV on both WM ans CSF
csfthick = nighres.intensity.intensity_propagation(csfbranch, mask=None, combine='max', distance_mm=0.4,
                      target='lower', scaling=0.5, domain=None,
                      save_data=True, overwrite=False, output_dir=outdir)['result']

wmthick = nighres.intensity.intensity_propagation(wmbranch, mask=None, combine='max', distance_mm=0.4,
                      target='lower', scaling=0.5, domain=None,
                      save_data=True, overwrite=False, output_dir=outdir)['result']

# prior to remove cortical GM: not used so far, let's see what topology correction brings
nocbprior = nighres.surface.levelset_to_probability(cbsurf, distance_mm=-2,
                                    save_data=True, overwrite=False, output_dir=outdir)['result']



recompute=True

# wm proba: fcm + *connected* ridges

# WM = T1 fcm 4,5 + T1 fcm 3 x T2 fcm 3 + *connected* T1 bright ridges - dura
# CSF = T1 fcm 1 + PD fcm 1 + T1 connected dark ridges + dura
# GM is the rest

wmprior = t1w_n4.replace('.nii','_wmrawprior.nii')
if recompute or not os.path.exists(wmprior):
    wmproba_img = nibabel.Nifti1Image(nighres.io.load_volume(t1fcm['memberships'][2]).get_fdata()\
                                     *nighres.io.load_volume(t2fcm['memberships'][2]).get_fdata(),
                                     img.affine, img.header)
    dilated = nighres.intensity.intensity_propagation(wmproba_img, mask=None, combine='mean', distance_mm=0.6,
                      target='lower', scaling=1.0, domain=None,
                      save_data=True, overwrite=True, output_dir=outdir,
                      file_name=os.path.basename(t1w_n4).replace('.nii','_wmdilated.nii'))['result']
    
    closed = nighres.intensity.intensity_propagation(dilated, mask=None, combine='mean', distance_mm=0.6,
                      target='higher', scaling=1.0, domain=None,
                      save_data=True, overwrite=True, output_dir=outdir,
                      file_name=os.path.basename(t1w_n4).replace('.nii','_wmclosed.nii'))['result']
    
    wmproba = nighres.io.load_volume(closed).get_fdata()
    wmproba = wmproba + nighres.io.load_volume(t1fcm['memberships'][3]).get_fdata()\
                      + nighres.io.load_volume(t1fcm['memberships'][4]).get_fdata()
    branchpv = nighres.io.load_volume(wmbranch).get_fdata()
    nobranch = nighres.io.load_volume(csfbranch).get_fdata()
    nocb = nighres.io.load_volume(nocbprior).get_fdata()
    duraroi = nighres.io.load_volume(duraprior).get_fdata()
    durapv = nighres.io.load_volume(dura).get_fdata()
    wmproba = nibabel.Nifti1Image(cbmask*(1.0-numpy.minimum(durapv+duraroi,numpy.ones(durapv.shape)))\
                                        *(1.0-nocb)\
                                        *numpy.minimum(wmproba+numpy.maximum(branchpv-nobranch,\
                                                    numpy.zeros(wmproba.shape)),numpy.ones(wmproba.shape)),\
                                img.affine,img.header)
    nighres.io.save_volume(wmprior,wmproba)
    wmproba = wmproba.get_fdata()
else:
    wmproba = nighres.io.load_volume(wmprior).get_fdata()

# for wm as well? or just take the rest?
csfprior = t1w_n4.replace('.nii','_csfrawprior.nii')
if recompute or not os.path.exists(csfprior):
    csfproba = nighres.io.load_volume(t1fcm['memberships'][0]).get_fdata()\
              +nighres.io.load_volume(pdfcm['memberships'][0]).get_fdata()
    branchpv = nighres.io.load_volume(csfbranch).get_fdata()
    nobranch = nighres.io.load_volume(wmbranch).get_fdata()
    nocb = nighres.io.load_volume(nocbprior).get_fdata()
    duraroi = nighres.io.load_volume(duraprior).get_fdata()
    durapv = nighres.io.load_volume(dura).get_fdata()
    csfproba = nibabel.Nifti1Image(cbmask*numpy.minimum(duraroi+durapv+csfproba+branchpv,numpy.ones(csfproba.shape)),
                                            img.affine,img.header)
    nighres.io.save_volume(csfprior,csfproba)
    csfproba = csfproba.get_fdata()
else:
    csfproba = nighres.io.load_volume(csfprior).get_fdata()


# for gm as well? or just take the rest?
gmprior = t1w_n4.replace('.nii','_gmrawprior.nii')
if recompute or not os.path.exists(gmprior):
    gmproba = nibabel.Nifti1Image(cbmask*(1.0-numpy.minimum(wmproba+csfproba,numpy.ones(wmproba.shape))),
                                    img.affine,img.header)
    nighres.io.save_volume(gmprior,gmproba)
else:
    gmproba = nighres.io.load_volume(gmprior)




recompute=True

wmprior = t1w_n4.replace('.nii','_wmprior.nii')
if recompute or not os.path.exists(wmprior):
    wmproba_img = nibabel.Nifti1Image(nighres.io.load_volume(t1fcm['memberships'][2]).get_fdata()\
                                     *nighres.io.load_volume(t2fcm['memberships'][2]).get_fdata(),
                                     img.affine, img.header)
    dilated = nighres.intensity.intensity_propagation(wmproba_img, mask=None, combine='mean', distance_mm=0.6,
                      target='lower', scaling=1.0, domain=None,
                      save_data=True, overwrite=True, output_dir=outdir,
                      file_name=os.path.basename(t1w_n4).replace('.nii','_wmdilated.nii'))['result']
    
    closed = nighres.intensity.intensity_propagation(dilated, mask=None, combine='mean', distance_mm=0.6,
                      target='higher', scaling=1.0, domain=None,
                      save_data=True, overwrite=True, output_dir=outdir,
                      file_name=os.path.basename(t1w_n4).replace('.nii','_wmclosed.nii'))['result']
    
    wmproba = nighres.io.load_volume(closed).get_fdata()
    wmproba = wmproba + nighres.io.load_volume(t1fcm['memberships'][3]).get_fdata()\
                      + nighres.io.load_volume(t1fcm['memberships'][4]).get_fdata()
    branchpv = nighres.io.load_volume(wmthick).get_fdata()
    nobranch = nighres.io.load_volume(csfthick).get_fdata()
    nocb = nighres.io.load_volume(nocbprior).get_fdata()
    duraroi = nighres.io.load_volume(duraprior).get_fdata()
    durapv = nighres.io.load_volume(dura).get_fdata()
    wmproba = nibabel.Nifti1Image(cbmask*(1.0-numpy.minimum(durapv+duraroi,numpy.ones(durapv.shape)))\
                                        *(1.0-nocb)\
                                        *numpy.minimum(wmproba+numpy.maximum(branchpv-nobranch,\
                                                    numpy.zeros(wmproba.shape)),numpy.ones(wmproba.shape)),\
                                img.affine,img.header)
    nighres.io.save_volume(wmprior,wmproba)
    wmproba = wmproba.get_fdata()
else:
    wmproba = nighres.io.load_volume(wmprior).get_fdata()

# for wm as well? or just take the rest?
csfprior = t1w_n4.replace('.nii','_csfprior.nii')
if recompute or not os.path.exists(csfprior):
    csfproba = nighres.io.load_volume(t1fcm['memberships'][0]).get_fdata()\
              +nighres.io.load_volume(pdfcm['memberships'][0]).get_fdata()
    branchpv = nighres.io.load_volume(csfthick).get_fdata()
    nobranch = nighres.io.load_volume(wmthick).get_fdata()
    nocb = nighres.io.load_volume(nocbprior).get_fdata()
    duraroi = nighres.io.load_volume(duraprior).get_fdata()
    durapv = nighres.io.load_volume(dura).get_fdata()
    csfproba = nibabel.Nifti1Image(cbmask*numpy.minimum(duraroi+durapv+csfproba+branchpv,numpy.ones(csfproba.shape)),
                                            img.affine,img.header)
    nighres.io.save_volume(csfprior,csfproba)
    csfproba = csfproba.get_fdata()
else:
    csfproba = nighres.io.load_volume(csfprior).get_fdata()


# for gm as well? or just take the rest?
gmprior = t1w_n4.replace('.nii','_gmprior.nii')
if recompute or not os.path.exists(gmprior):
    gmproba = nibabel.Nifti1Image(cbmask*(1.0-numpy.minimum(wmproba+csfproba,numpy.ones(wmproba.shape))),
                                    img.affine,img.header)
    nighres.io.save_volume(gmprior,gmproba)
else:
    gmproba = nighres.io.load_volume(gmprior)


