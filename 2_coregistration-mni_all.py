import nighres
import os
import nibabel
import numpy
import glob

inpath = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/'
prev_sfx = 'nighres/denoise/'
out_sfx = 'nighres/coreg/'

t1map_sfx = '_anat_inv1_s2c10_lcpca-den_qt1map-t1.nii.gz'
t1w_sfx = '_anat_inv1_s2c10_lcpca-den_qt1map-uni.nii.gz'

t1ref_sfx = '_anat_HEADT1w.nii.gz'

inv1m_sfx = '_anat_inv1_s2c10_lcpca-den.nii.gz'
inv1p_sfx = '_anat_inv1_ph_s2c10_lcpca-den.nii.gz'
inv2m_sfx = '_anat_inv2_s2c10_lcpca-den.nii.gz'
inv2p_sfx = '_anat_inv2_ph_s2c10_lcpca-den.nii.gz'

mni = '/home/pilou/Datasets/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz'
#mnicb = '/home/pilou/Datasets/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_cb_mask-int60.nii.gz'
mnicb = '/home/pilou/Datasets/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_cb_mask-int60_close3.nii.gz'
mnicbl = '/home/pilou/Datasets/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_cb_mask-int60_close3_left.nii.gz'
mnicbr = '/home/pilou/Datasets/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_cb_mask-int60_close3_right.nii.gz'

indirs = glob.glob(inpath+'*/anat/')
for indir in indirs:
    
    subject = indir.replace(inpath,'').replace('/anat/','')
    print(subject)
    outdir = indir.replace('anat/',out_sfx)
    prevdir = indir.replace('anat/',prev_sfx)

    t1ref = indir+subject+t1ref_sfx

    inv1m = prevdir+subject+inv1m_sfx
    inv1p = prevdir+subject+inv1p_sfx
    inv2m = prevdir+subject+inv2m_sfx
    inv2p = prevdir+subject+inv2p_sfx
    
    t1map = prevdir+subject+t1map_sfx
    t1w = prevdir+subject+t1w_sfx
    
    if os.path.exists(t1ref) and os.path.exists(inv1m) and os.path.exists(inv1p) \
            and os.path.exists(inv2m) and os.path.exists(inv2p) \
                and os.path.exists(t1map) and os.path.exists(t1w):

        # coregister sla to whole brain, whole brain to MNI2009c

        slab2wb = nighres.registration.embedded_antspy(t1w, t1ref,
                    run_rigid=True,
                    rigid_iterations=1000,
                    run_affine=False,
                    run_syn=False,
                    scaling_factor=8,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False,smooth_mask=0.0,
					ignore_affine=False, ignore_header=False,
                    save_data=True, overwrite=False, output_dir=outdir)

        wb2mni1 = nighres.registration.embedded_antspy(t1ref, mni,
                    run_rigid=True,
                    rigid_iterations=1000,
                    run_affine=True,
                    affine_iterations=1000,
                    run_syn=True,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
					scaling_factor=8,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False,smooth_mask=0.0,
					ignore_affine=False, ignore_header=False,
                    save_data=True, overwrite=False, output_dir=outdir)

        wb2mni2 = nighres.registration.embedded_antspy(wb2mni1['transformed_source'], mni,
                    run_rigid=False,
                    run_affine=False,
                    run_syn=True,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
					scaling_factor=8,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False,smooth_mask=0.0,
					ignore_affine=False, ignore_header=False,
                    save_data=True, overwrite=False, output_dir=outdir)

        # transfer CB masks
        cb_mask = nighres.registration.apply_coordinate_mappings(mnicb, mapping1=wb2mni2['inverse'],
                        mapping2=wb2mni1['inverse'], mapping3=slab2wb['inverse'],
                        interpolation="linear", padding="closest",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir)['result']

        # crop & rescale the data
        crop = nighres.registration.crop_mapping(cb_mask, boundary=20, save_data=True, overwrite=False, output_dir=outdir)

        rescale = nighres.registration.rescale_mapping(crop['mapping'], scaling_factor=2.0,
                        save_data=True, overwrite=False, output_dir=outdir)

        # re-align for more accurate masking?
        mni_t1c = nighres.registration.apply_coordinate_mappings(mni, mapping1=wb2mni2['inverse'],
                        mapping2=wb2mni1['inverse'], mapping3=slab2wb['inverse'], mapping4=crop['mapping'],
                        interpolation="linear", padding="closest",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name='min2009c2'+os.path.basename(t1w).replace('.nii','_c.nii'))['result']

        t1c = nighres.registration.apply_coordinate_mappings(t1w, mapping1=crop['mapping'],
                        interpolation="linear", padding="closest",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name=os.path.basename(t1w).replace('.nii','_c.nii'))['result']

        # combine with background estimation for better alignment?

        bg = nighres.intensity.background_estimation(inv2m, distribution='exponential', ratio=1e-3,
                          skip_zero=True, iterate=True, dilate=0,
                          threshold=0.5,
                          save_data=True, overwrite=False, output_dir=outdir)['proba']

        bgc = nighres.registration.apply_coordinate_mappings(bg, mapping1=crop['mapping'],
                        interpolation="linear", padding="closest",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name=os.path.basename(bg).replace('.nii','_c.nii'))['result']
        
        bgc = nighres.io.load_volume(bgc).get_fdata()
        bgc = 1.0/(1.0 + numpy.exp( -10.0*(bgc-0.2)))
        t1c_img = nighres.io.load_volume(t1c)
        t1c_img = nibabel.Nifti1Image(bgc*t1c_img.get_fdata(),t1c_img.affine,t1c_img.header)
        nighres.io.save_volume(t1c.replace('.nii','_xbg.nii'),t1c_img)
        
        mni2t1c = nighres.registration.embedded_antspy(mni_t1c, t1c.replace('.nii','_xbg.nii'),
                            run_rigid=False,
                            rigid_iterations=1000,
                            run_affine=False,
                            affine_iterations=1000,
                            run_syn=True,
                            coarse_iterations=60,
                            medium_iterations=80, fine_iterations=60,
                            scaling_factor=8,
                            cost_function='MutualInformation',
                            interpolation='NearestNeighbor',
                            regularization='High',
                            convergence=1e-8,
                            mask_zero=False,smooth_mask=0.0,
                            ignore_affine=False, ignore_header=False,
                            save_data=True, overwrite=False, output_dir=outdir)
        
        
        
        # use denoised & skull-stripped data once available
        t1mapcr = nighres.registration.apply_coordinate_mappings(t1map, mapping1=crop['mapping'],
                                mapping2=rescale['mapping'],
                                interpolation="linear", padding="closest",
                                zero_border=0,
                                save_data=True, overwrite=False, output_dir=outdir, 
                                file_name=os.path.basename(t1map).replace('.nii','_cr.nii'))['result']
        
        cbmaskcr = nighres.registration.apply_coordinate_mappings(cb_mask, mapping1=crop['mapping'],
                                mapping2=mni2t1c['mapping'],
                                mapping3=rescale['mapping'],
                                interpolation="linear", padding="zero",
                                zero_border=0,
                                save_data=True, overwrite=False, output_dir=outdir, 
                                file_name=os.path.basename(t1map).replace('.nii','_cr_cbmask.nii'))['result']
        
        # transfer additional masks
        cb_mask_l = nighres.registration.apply_coordinate_mappings(mnicbl, mapping1=wb2mni2['inverse'],
                                mapping2=wb2mni1['inverse'], mapping3=slab2wb['inverse'],
                                interpolation="linear", padding="closest",
                                zero_border=0,
                                save_data=True, overwrite=False, output_dir=outdir)['result']
        
        cbmasklcr = nighres.registration.apply_coordinate_mappings(cb_mask_l, mapping1=crop['mapping'],
                                mapping2=mni2t1c['mapping'],
                                mapping3=rescale['mapping'],
                                interpolation="linear", padding="zero",
                                zero_border=0,
                                save_data=True, overwrite=False, output_dir=outdir, 
                                file_name=os.path.basename(t1map).replace('.nii','_cr_cbmask_l.nii'))['result']
        
        cb_mask_r = nighres.registration.apply_coordinate_mappings(mnicbr, mapping1=wb2mni2['inverse'],
                                mapping2=wb2mni1['inverse'], mapping3=slab2wb['inverse'],
                                interpolation="linear", padding="closest",
                                zero_border=0,
                                save_data=True, overwrite=False, output_dir=outdir)['result']
        
        cbmaskrcr = nighres.registration.apply_coordinate_mappings(cb_mask_r, mapping1=crop['mapping'],
                                mapping2=mni2t1c['mapping'],
                                mapping3=rescale['mapping'],
                                interpolation="linear", padding="zero",
                                zero_border=0,
                                save_data=True, overwrite=False, output_dir=outdir, 
                                file_name=os.path.basename(t1map).replace('.nii','_cr_cbmask_r.nii'))['result']
