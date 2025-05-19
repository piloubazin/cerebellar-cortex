import nighres
import os
import nibabel
import numpy
import glob
import scipy.ndimage

inpath = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/'
in_sfx = 'nighres/coreg/'
out_sfx = 'nighres/tissues/'

t1map_sfx = '_anat_inv1_s2c10_lcpca-den_qt1map-t1_cr_def-img.nii.gz'

cbprior_sfx = '_anat_inv1_s2c10_lcpca-den_qt1map-t1_cr_cbmask_def-img.nii.gz'


indirs = glob.glob(inpath+'*/'+in_sfx)
for indir in indirs:
    
    subject = indir.replace(inpath,'').replace(in_sfx,'').replace('/','')
    print(subject)
    outdir = indir.replace(in_sfx,out_sfx)
    
    t1map = indir+subject+t1map_sfx
    cbprior = indir+subject+cbprior_sfx
    print('input files:')
    print(t1map)
    print(cbprior)
    
    if os.path.exists(t1map) and os.path.exists(cbprior):

        recompute=False
        
        t1img = nighres.io.load_volume(t1map)
            
        # crop outside region for now (TODO: make levelset, crop from some distance outside)
        cbsurf = nighres.surface.probability_to_levelset(cbprior, mask_image=None,
                                    save_data=True, overwrite=False, output_dir=outdir)['result']
        
        # mask at 20 voxels (4mm) from the p=0.5 boundary of the MNI cb prior 
        # (very generous, but mostly masked as dura later on)
        cbmask = nighres.io.load_volume(cbsurf).get_fdata()<20
        
        
        t1msk = t1map.replace('.nii','_msk.nii').replace(indir,outdir)
        if recompute or not os.path.exists(t1msk):
            t1img = nibabel.Nifti1Image(cbmask*t1img.get_fdata(),t1img.affine,t1img.header)
            nighres.io.save_volume(t1msk,t1img)
        
        # tests: supervoxels not helpful (too much driven by residual noise)
        # tests: FCM works better with 4 classes, to model the spread of GM values and inhomogeneities
        
        # a combination of FCM to get large WM, GM, CSF regions and ridge filtering to get the fine branches
        # seems promising
        
        fcm = nighres.segmentation.fuzzy_cmeans(t1msk, clusters=4, max_iterations=50, max_difference=0.01, 
                            smoothing=0.1, fuzziness=2.0, mask_zero=True, map_intensity=True,
                            save_data=True, overwrite=False, output_dir=outdir, 
                            file_name=os.path.basename(t1msk).replace('.nii','_cl4.nii'))
        
        
        wmbranch = nighres.filtering.recursive_ridge_diffusion(t1msk, ridge_intensities='dark', ridge_filter='2D',
                                      surface_levelset=None, orientation='undefined',
                                      loc_prior=None,
                                      min_scale=0, max_scale=0,
                                      diffusion_factor=0.95,
                                      similarity_scale=0.1,
                                      max_iter=100, max_diff=1e-3,
                                      threshold=0.5, rescale=False,
                                      save_data=True, overwrite=False, output_dir=outdir,
                                      file_name=os.path.basename(t1msk).replace('.nii','_wm095.nii'))['propagation']
        
        # results are too fine for proper surface reconstruction, we need to add some artificial PV on both WM ans CSF
        wmthick = nighres.intensity.intensity_propagation(wmbranch, mask=None, combine='max', distance_mm=0.4,
                              target='lower', scaling=0.5, domain=None,
                              save_data=True, overwrite=False, output_dir=outdir)['result']
        
        csfbranch = nighres.filtering.recursive_ridge_diffusion(t1msk, ridge_intensities='bright', ridge_filter='2D',
                                      surface_levelset=None, orientation='undefined',
                                      loc_prior=None,
                                      min_scale=0, max_scale=0,
                                      diffusion_factor=0.95,
                                      similarity_scale=0.1,
                                      max_iter=100, max_diff=1e-3,
                                      threshold=0.5, rescale=False,
                                      save_data=True, overwrite=False, output_dir=outdir,
                                      file_name=os.path.basename(t1msk).replace('.nii','_csf095.nii'))['propagation']
        
        # results are too fine for proper surface reconstruction, we need to add some artificial PV on both WM ans CSF
        csfthick = nighres.intensity.intensity_propagation(csfbranch, mask=None, combine='max', distance_mm=0.4,
                              target='lower', scaling=0.5, domain=None,
                              save_data=True, overwrite=False, output_dir=outdir)['result']
        
        # vessel detection not doing much, gives spurrious results on branches
        
        # specific detection of dura: use cbmask as surface direction + location prior to find the dark dura band between cb and ctx
        duraprior = nighres.surface.levelset_to_probability(cbsurf, distance_mm=-2,
                                    save_data=True, overwrite=False, output_dir=outdir)['result']
        
        dura = nighres.filtering.recursive_ridge_diffusion(t1msk, ridge_intensities='dark', ridge_filter='2D',
                                      surface_levelset=cbsurf, orientation='parallel',
                                      loc_prior=duraprior,
                                      min_scale=0, max_scale=0,
                                      diffusion_factor=0.95,
                                      similarity_scale=0.1,
                                      max_iter=100, max_diff=1e-3,
                                      threshold=0.5, rescale=False,
                                      save_data=True, overwrite=False, output_dir=outdir,
                                      file_name=os.path.basename(t1msk).replace('.nii','_dur095.nii'))['propagation']
        
        # results are too fine for proper surface reconstruction, we need to add some artificial PV on both WM ans CSF
        dura = nighres.intensity.intensity_propagation(dura, mask=None, combine='max', distance_mm=0.4,
                              target='lower', scaling=0.5, domain=None,
                              save_data=True, overwrite=False, output_dir=outdir)['result']
        
        
        recompute=False
        
        # wm proba: fcm + ridges
        wmprior = t1msk.replace('.nii','_wmprior.nii')
        if recompute or not os.path.exists(wmprior):
            wmproba = nighres.io.load_volume(fcm['memberships'][0]).get_fdata()
            wmpv = nighres.io.load_volume(wmthick).get_fdata()
            duraroi = nighres.io.load_volume(duraprior).get_fdata()
            durapv = nighres.io.load_volume(dura).get_fdata()
            wmproba = nibabel.Nifti1Image(cbmask*(1.0-numpy.minimum(durapv+duraroi,numpy.ones(durapv.shape)))*
                                        numpy.minimum(wmproba+wmpv,numpy.ones(wmproba.shape)),
                                        t1img.affine,t1img.header)
            nighres.io.save_volume(wmprior,wmproba)
            wmproba = wmproba.get_fdata()
        else:
            wmproba = nighres.io.load_volume(wmprior).get_fdata()
        
        # same for csf
        csfprior = t1msk.replace('.nii','_csfprior.nii')
        if recompute or not os.path.exists(csfprior):
            csfproba = nighres.io.load_volume(fcm['memberships'][3]).get_fdata()
            csfpv = nighres.io.load_volume(csfthick).get_fdata()
            csfproba = nibabel.Nifti1Image(cbmask*
                                            numpy.minimum(duraroi+durapv+csfproba+csfpv,numpy.ones(csfproba.shape)),
                                            t1img.affine,t1img.header)
            nighres.io.save_volume(csfprior,csfproba)
            csfproba = csfproba.get_fdata()
        else:
            csfproba = nighres.io.load_volume(csfprior).get_fdata()
        
        
        recompute = False
        
        # for gm as well? or just take the rest?
        gmprior = t1msk.replace('.nii','_gmprior.nii')
        if recompute or not os.path.exists(gmprior):
            gmproba = nibabel.Nifti1Image(cbmask*(1.0-numpy.minimum(wmproba+csfproba,numpy.ones(wmproba.shape))),
                                            t1img.affine,t1img.header)
            nighres.io.save_volume(gmprior,gmproba)
        else:
            gmproba = nighres.io.load_volume(gmprior)
        
        
        
        recompute=False
        
        # wm proba: fcm + ridges
        wmprior = t1msk.replace('.nii','_wmrawprior.nii')
        if recompute or not os.path.exists(wmprior):
            wmproba = nighres.io.load_volume(fcm['memberships'][0]).get_fdata()
            wmpv = nighres.io.load_volume(wmbranch).get_fdata()
            duraroi = nighres.io.load_volume(duraprior).get_fdata()
            durapv = nighres.io.load_volume(dura).get_fdata()
            wmproba = nibabel.Nifti1Image(cbmask*(1.0-numpy.minimum(durapv+duraroi,numpy.ones(durapv.shape)))*
                                        numpy.minimum(wmproba+wmpv,numpy.ones(wmproba.shape)),
                                        t1img.affine,t1img.header)
            nighres.io.save_volume(wmprior,wmproba)
            wmproba = wmproba.get_fdata()
        else:
            wmproba = nighres.io.load_volume(wmprior).get_fdata()
        
        # same for csf
        csfprior = t1msk.replace('.nii','_csfrawprior.nii')
        if recompute or not os.path.exists(csfprior):
            csfproba = nighres.io.load_volume(fcm['memberships'][3]).get_fdata()
            csfpv = nighres.io.load_volume(csfbranch).get_fdata()
            csfproba = nibabel.Nifti1Image(cbmask*
                                            numpy.minimum(duraroi+durapv+csfproba+csfpv,numpy.ones(csfproba.shape)),
                                            t1img.affine,t1img.header)
            nighres.io.save_volume(csfprior,csfproba)
            csfproba = csfproba.get_fdata()
        else:
            csfproba = nighres.io.load_volume(csfprior).get_fdata()
        
        
        recompute = False
        
        # for gm as well? or just take the rest?
        gmprior = t1msk.replace('.nii','_gmrawprior.nii')
        if recompute or not os.path.exists(gmprior):
            gmproba = nibabel.Nifti1Image(cbmask*(1.0-numpy.minimum(wmproba+csfproba,numpy.ones(wmproba.shape))),
                                            t1img.affine,t1img.header)
            nighres.io.save_volume(gmprior,gmproba)
        else:
            gmproba = nighres.io.load_volume(gmprior)
        
        
