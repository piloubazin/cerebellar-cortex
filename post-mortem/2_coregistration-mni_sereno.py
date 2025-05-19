import nighres
import os
import nibabel
import numpy
import glob

inpath = '/home/pilou/Datasets/Sereno_cb2020/cerebmap20/mri/'
outdir = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/sereno/coreg/'


T1_img = inpath+'orig.mgz'
T2_img = inpath+'T2.mgz'
PD_img = inpath+'PD.mgz'


mni09c = '/home/pilou/Datasets/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz'
mnicb = '/home/pilou/Datasets/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_cb_mask-int60_close3.nii.gz'

# first crop the cb in MNI template

os.makedirs(outdir, exist_ok=True)

mni09c_cb_file = outdir+'mni09c_t1_cbmasked.nii.gz'
if not os.path.exists(mni09c_cb_file):
    mni09c_cb = nighres.io.load_volume(mni09c)
    cbmask = nighres.io.load_volume(mnicb).get_fdata()
    mni09c_cb = nibabel.Nifti1Image(cbmask*mni09c_cb.get_fdata(),mni09c_cb.affine,mni09c_cb.header)
    nighres.io.save_volume(mni09c_cb_file, mni09c_cb)

crop_mni = nighres.registration.crop_mapping(mnicb, boundary=20, save_data=True, overwrite=False, output_dir=outdir)

mni_t1w_cr = nighres.registration.apply_coordinate_mappings(mni09c_cb_file, mapping1=crop_mni['mapping'],
                        interpolation="nearest", padding="zero",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name=os.path.basename(mni09c).replace('.nii','_cr.nii'))['result']

# make nifti copies of the Sereno data
T1_nii = outdir+'serenocb20_T1.nii.gz'
T2_nii = outdir+'serenocb20_T2.nii.gz'
PD_nii = outdir+'serenocb20_PD.nii.gz'
if not os.path.exists(T1_nii) or not os.path.exists(T1_nii) or not os.path.exists(PD_nii):
    T1_img = nighres.io.load_volume(T1_img)
    T1_img = nibabel.Nifti1Image(T1_img.get_fdata(),T1_img.affine,T1_img.header)
    T1_img.header.set_zooms([0.19,0.19,0.19])
    nighres.io.save_volume(T1_nii,T1_img)
    
    T2_img = nighres.io.load_volume(T2_img)
    T2_img = nibabel.Nifti1Image(T2_img.get_fdata(),T2_img.affine,T2_img.header)
    T2_img.header.set_zooms([0.19,0.19,0.19])
    nighres.io.save_volume(T2_nii,T2_img)
    
    PD_img = nighres.io.load_volume(PD_img)
    PD_img = nibabel.Nifti1Image(PD_img.get_fdata(),PD_img.affine,PD_img.header)
    PD_img.header.set_zooms([0.19,0.19,0.19])
    nighres.io.save_volume(PD_nii,PD_img)
    
    
cb2mni1 = nighres.registration.embedded_antspy(T1_nii, mni_t1w_cr,
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
            ignore_affine=True, ignore_header=False,
            save_data=True, overwrite=False, output_dir=outdir)

cb2mni2 = nighres.registration.embedded_antspy(cb2mni1['transformed_source'], mni_t1w_cr,
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
cb_mask = nighres.registration.apply_coordinate_mappings(mni_t1w_cr, mapping1=cb2mni2['inverse'],
                mapping2=cb2mni1['inverse'], 
                interpolation="linear", padding="closest",
                zero_border=0,
                save_data=True, overwrite=False, output_dir=outdir)['result']

# crop & rescale the data
cb_img = nighres.io.load_volume(cb_mask)
cb_img = nibabel.Nifti1Image(cb_img.get_fdata()>0,cb_img.affine,cb_img.header)
crop = nighres.registration.crop_mapping(cb_img, boundary=20, save_data=True, overwrite=False, output_dir=outdir, file_name=cb_mask)

# use denoised & skull-stripped data once available
t1cr = nighres.registration.apply_coordinate_mappings(T1_nii, mapping1=crop['mapping'],
                        interpolation="nearest", padding="closest",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name=os.path.basename(T1_nii).replace('.nii','_cr.nii'))['result']

t2cr = nighres.registration.apply_coordinate_mappings(T2_nii, mapping1=crop['mapping'],
                        interpolation="nearest", padding="closest",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name=os.path.basename(T2_nii).replace('.nii','_cr.nii'))['result']

pdcr = nighres.registration.apply_coordinate_mappings(PD_nii, mapping1=crop['mapping'],
                        interpolation="nearest", padding="closest",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name=os.path.basename(PD_nii).replace('.nii','_cr.nii'))['result']

cbmaskcr = nighres.registration.apply_coordinate_mappings(cb_mask, mapping1=crop['mapping'],
                        interpolation="linear", padding="zero",
                        zero_border=0,
                        save_data=True, overwrite=False, output_dir=outdir, 
                        file_name=os.path.basename(T1_nii).replace('.nii','_cr_cbmask.nii'))['result']
