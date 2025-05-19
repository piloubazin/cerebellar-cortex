import nighres
import os
import nibabel
import numpy
import glob

inpath = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/'
out_sfx = 'nighres/denoise/'

inv1m_sfx = '_anat_inv1.nii.gz'
inv1p_sfx = '_anat_inv1_ph.nii.gz'
inv2m_sfx = '_anat_inv2.nii.gz'
inv2p_sfx = '_anat_inv2_ph.nii.gz'

mni = '/home/pilou/Data/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz'

indirs = glob.glob(inpath+'*/anat/')
print(indirs)
for indir in indirs:
    
    subject = indir.replace(inpath,'').replace('/anat/','')
    print(subject)
    outdir = indir.replace('anat/',out_sfx)
    
    inv1m = indir+subject+inv1m_sfx
    inv1p = indir+subject+inv1p_sfx
    inv2m = indir+subject+inv2m_sfx
    inv2p = indir+subject+inv2p_sfx
    
    if os.path.exists(inv1m) and os.path.exists(inv1p) and os.path.exists(inv2m) and os.path.exists(inv2p):
        
        # 1. LCPCA denoising (conservative parameters: ngb_size=3 introduces serious smoothing already)
        outputs = [inv1m.replace(indir,outdir).replace('.nii','_s2c10.nii'),
                   inv2m.replace(indir,outdir).replace('.nii','_s2c10.nii'),
                   inv1p.replace(indir,outdir).replace('.nii','_s2c10.nii'),
                   inv2p.replace(indir,outdir).replace('.nii','_s2c10.nii')]

        lcpca = nighres.intensity.lcpca_denoising([inv1m,inv2m], phase_list=[inv1p,inv2p], 
                    ngb_size=2, stdev_cutoff=1.10,
                    min_dimension=0, max_dimension=-1,
                    unwrap=True, rescale_phs=True, process_2d=False, use_rmt=False,
                    save_data=True, overwrite=False, output_dir=outdir, file_names=outputs)

        # sweet spot seems to be ngb_size=2, cutoff=1.10 (cutoff not very sensitive)

        # 2. reconstruction (no further denoising)

        qmri = nighres.intensity.mp2rage_t1_mapping([lcpca['denoised'][0],lcpca['denoised'][2]], 
                      [lcpca['denoised'][1],lcpca['denoised'][3]],
                      inversion_times=[1.0, 2.9], flip_angles=[7.0, 5.0], inversion_TR=5.0,
                      excitation_TR=[0.0062, 0.0062], N_excitations=150, efficiency=0.96,
                      correct_B1=False, B1_map=None, B1_scale=1.0,
                      scale_phase=True,
                      save_data=True, overwrite=False, output_dir=outdir)
