import nighres
import os
import nibabel
import numpy
import glob
import scipy.ndimage

indir = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/sereno/topology/'
prevdir = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/sereno/tissues/'
outdir = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/sereno/surfaces/'

wmrawprior = prevdir+'serenocb20_T1_cr_def-img_n4_wmrawprior.nii.gz'
gmrawprior = prevdir+'serenocb20_T1_cr_def-img_n4_gmrawprior.nii.gz'
csfrawprior = prevdir+'serenocb20_T1_cr_def-img_n4_csfrawprior.nii.gz'

wmtpc_wcs = indir+'serenocb20_T1_cr_def-img_n4_wmlvdiff_wcs_tpc-obj.nii.gz'

os.makedirs(outdir, exist_ok=True)

cruise = nighres.cortex.cruise_cortex_extraction(wmtpc_wcs, wmrawprior, gmrawprior, csfrawprior,
                             vd_image=None, data_weight=0.4,
                             regularization_weight=0.05,
                             max_iterations=800, normalize_probabilities=False,
                             correct_wm_pv=False, wm_dropoff_dist=5.0,
                             topology='wcs',
                             save_data=True, overwrite=False, output_dir=outdir,
                             file_name=os.path.basename(gmrawprior).replace('.nii','_wcs.nii'))


# surface meshes
gwb_surf = nighres.surface.levelset_to_mesh(cruise['gwb'], connectivity="wcs", level=0.0,
                        save_data=True, overwrite=False,
                        output_dir=outdir)

cgb_surf = nighres.surface.levelset_to_mesh(cruise['cgb'], connectivity="wcs", level=0.0,
                        save_data=True, overwrite=False,
                        output_dir=outdir)

# surface inflation?
avg_surf = nighres.surface.levelset_to_mesh(cruise['avg'], connectivity="wcs", level=0.0,
                        save_data=True, overwrite=False,
                        output_dir=outdir)

# distance is more stable than area, step size controls speed, regularization seems to increase pointiness (??)
inf_surf = nighres.surface.surface_inflation(surface_mesh=avg_surf['result'],
                        max_iter=200, max_curv=20.0,
                        regularization=0.0, method='dist', step_size=0.75, 
                        save_data=True,overwrite=False,output_dir=outdir,
                        file_name=os.path.basename(avg_surf['result']).replace('.vtk','_it200.vtk'))

