import nighres
import os
import nibabel
import numpy
import glob
import scipy.ndimage

datadir = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/ahead/coreg/'
indir = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/ahead/surfaces/'
outdir = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/ahead/features/'

cgb = indir+'Ahead_brain_122017_thionin-interpolated_cr_def-img_msk_gmrawprior_wcs_cruise-cgb.nii.gz'
gwb = indir+'Ahead_brain_122017_thionin-interpolated_cr_def-img_msk_gmrawprior_wcs_cruise-gwb.nii.gz'
avg = indir+'Ahead_brain_122017_thionin-interpolated_cr_def-img_msk_gmrawprior_wcs_cruise-avg.nii.gz'

cgb_mesh = indir+'Ahead_brain_122017_thionin-interpolated_cr_def-img_msk_gmrawprior_wcs_cruise-cgb_l2m-mesh.vtk'
gwb_mesh = indir+'Ahead_brain_122017_thionin-interpolated_cr_def-img_msk_gmrawprior_wcs_cruise-gwb_l2m-mesh.vtk'
avg_mesh = indir+'Ahead_brain_122017_thionin-interpolated_cr_def-img_msk_gmrawprior_wcs_cruise-avg_l2m-mesh.vtk'

thick = indir+'Ahead_brain_122017_thionin-interpolated_cr_def-img_msk_gmrawprior_wcs_cruise-thick.nii.gz'
parv = datadir+'Ahead_brain_122017_parvalbumin-interpolated_cr_def-img.nii.gz'



os.makedirs(outdir, exist_ok=True)


# Laplacian embedding
embedding = nighres.shape.spectral_mesh_spatial_embedding(avg_mesh, dims=3,
                    msize=3600, scale=10.0, depth=18, eigengame=True,
                    save_data=True, 
                    overwrite=False, 
                    output_dir=outdir,
                    file_name=os.path.basename(avg_mesh).replace('.vtk','_sc10x3600_egg.vtk'))

# replace coordinates by embedding value
surf = nighres.io.load_mesh(embedding['result'])

surf['points'] = surf['data'][:,0:3]
coord_file = embedding['result'].replace('.vtk','_coordmap.vtk')
nighres.io.save_mesh(coord_file, surf)

# surface mapping
nighres.surface.surface_mesh_mapping(thick, avg_mesh, inflated_mesh=coord_file,
                         mapping_method="closest_point",
                         save_data=True, overwrite=False, output_dir=outdir)

nighres.surface.surface_mesh_mapping(parv, avg_mesh, inflated_mesh=coord_file,
                         mapping_method="closest_point",
                         save_data=True, overwrite=False, output_dir=outdir)

# geoemtric features
curv = nighres.surface.levelset_curvature(avg, distance=1.0,
                            save_data=True, overwrite=False, output_dir=outdir)
