import nighres
import os
import nibabel
import numpy
import glob

inpath = '/home/public/InSitu/Processed/insitu_03_152017_female_75/release/'
outdir = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/ahead2/coreg/'


bf_img = inpath+'Ahead_brain_152017_blockface-image.nii.gz'
thio_img = inpath+'Ahead_brain_152017_thionin-interpolated.nii.gz'
parv_img = inpath+'Ahead_brain_152017_parvalbumin-interpolated.nii.gz'
silv_img = inpath+'Ahead_brain_152017_Bielschowsky-interpolated.nii.gz'


mni09c = '/home/pilou/Datasets/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz'
mnicb = '/home/pilou/Datasets/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_cb_mask-int60_close3.nii.gz'
mnicbwm = '/home/pilou/Datasets/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_cbwm_mask-int80.nii.gz'

mni09b =  '/home/pilou/Datasets/mni_icbm152_nlin_asym_09b/mni_icbm152_t1_tal_nlin_asym_09b_hires.nii.gz'

mni09b2bf = inpath+'mni2009b/Ahead_brain_152017_mapping-mni2009b2blockface.nii.gz'

b2c = nighres.registration.embedded_antspy(mni09b, mni09c,
            run_rigid=True,
            rigid_iterations=1000,
            run_affine=False,
            affine_iterations=1000,
            run_syn=False,
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
cb_mask = nighres.registration.apply_coordinate_mappings(mnicb, mapping1=b2c['inverse'],
                mapping2=mni09b2bf,
                interpolation="linear", padding="closest",
                zero_border=0,
                save_data=True, overwrite=False, output_dir=outdir)['result']

cbwm_mask = nighres.registration.apply_coordinate_mappings(mnicbwm, mapping1=b2c['inverse'],
                mapping2=mni09b2bf,
                interpolation="linear", padding="closest",
                zero_border=0,
                save_data=True, overwrite=False, output_dir=outdir)['result']

# crop & rescale the data
crop = nighres.registration.crop_mapping(cb_mask, boundary=20, save_data=True, overwrite=False, output_dir=outdir)

# use denoised & skull-stripped data once available
bfcr = nighres.registration.apply_coordinate_mappings(bf_img, mapping1=crop['mapping'],
                        interpolation="nearest", padding="closest",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name=os.path.basename(bf_img).replace('.nii','_cr.nii'))['result']

thiocr = nighres.registration.apply_coordinate_mappings(thio_img, mapping1=crop['mapping'],
                        interpolation="nearest", padding="closest",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name=os.path.basename(thio_img).replace('.nii','_cr.nii'))['result']

parvcr = nighres.registration.apply_coordinate_mappings(parv_img, mapping1=crop['mapping'],
                        interpolation="nearest", padding="closest",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name=os.path.basename(parv_img).replace('.nii','_cr.nii'))['result']

silvcr = nighres.registration.apply_coordinate_mappings(silv_img, mapping1=crop['mapping'],
                        interpolation="nearest", padding="closest",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name=os.path.basename(silv_img).replace('.nii','_cr.nii'))['result']

cbmaskcr = nighres.registration.apply_coordinate_mappings(cb_mask, mapping1=crop['mapping'],
                        interpolation="linear", padding="zero",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name=os.path.basename(bf_img).replace('.nii','_cr_cbmask.nii'))['result']

cbwmmaskcr = nighres.registration.apply_coordinate_mappings(cbwm_mask, mapping1=crop['mapping'],
                        interpolation="linear", padding="zero",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name=os.path.basename(bf_img).replace('.nii','_cr_cbwmmask.nii'))['result']
