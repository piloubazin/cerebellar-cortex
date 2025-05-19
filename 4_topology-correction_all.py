import nighres
import os
import nibabel
import numpy
import glob
import scipy.ndimage

inpath = '/home/pilou/Projects/Cerebellum/Cerebellar-segmentation-ABC/'
in_sfx = 'nighres/tissues/'
out_sfx = 'nighres/topology/'

wmprior_sfx = '_anat_inv1_s2c10_lcpca-den_qt1map-t1_cr_def-img_msk_wmprior.nii.gz'
gmprior_sfx = '_anat_inv1_s2c10_lcpca-den_qt1map-t1_cr_def-img_msk_gmprior.nii.gz'
csfprior_sfx = '_anat_inv1_s2c10_lcpca-den_qt1map-t1_cr_def-img_msk_csfprior.nii.gz'

cbsurf_sfx = '_anat_inv1_s2c10_lcpca-den_qt1map-t1_cr_cbmask_def-img_p2l-surf.nii.gz'

indirs = glob.glob(inpath+'*/'+in_sfx)
for indir in indirs:
    
    subject = indir.replace(inpath,'').replace(in_sfx,'').replace('/','')
    print(subject)
    outdir = indir.replace(in_sfx,out_sfx)
    os.makedirs(outdir, exist_ok=True)

    wmprior = indir+subject+wmprior_sfx
    gmprior = indir+subject+gmprior_sfx
    csfprior = indir+subject+csfprior_sfx
    
    cbsurf = indir+subject+cbsurf_sfx

    wmimg = nighres.io.load_volume(wmprior)

    wmpost = wmprior.replace(indir,outdir)

    # clean up the WM to be a single connected component without holes, adjust priors accordingly

    recompute=False

    wmthres = wmpost.replace('_wmprior.nii','_wmthres.nii')
    wmthrespv = wmpost.replace('_wmprior.nii','_wmthrespv.nii')
    csfthrespv = wmpost.replace('_wmprior.nii','_csfthrespv.nii')
    if recompute or not os.path.exists(wmthres) or not os.path.exists(wmthrespv) or not os.path.exists(csfthrespv):
        wmproba = nighres.io.load_volume(wmprior).get_fdata()
        wmbg = wmproba<0.5
    
        # use connected components for background
        wmlb,nlb = scipy.ndimage.label(wmbg,scipy.ndimage.generate_binary_structure(3, 3))
    
        wmvol,edges = numpy.histogram(wmlb,bins=(nlb+1))
        nmax=numpy.argmax(wmvol[1:])
        print("main label: "+str(nmax))
        print("volumes: "+str(wmvol))
        wmobj = 1.0-(wmlb==nmax+1)
        
        # use connected components for foreground    
        wmlb,nlb = scipy.ndimage.label(wmobj,scipy.ndimage.generate_binary_structure(3, 1))
    
        wmvol,edges = numpy.histogram(wmlb,bins=(nlb+1))
        nmax=numpy.argmax(wmvol[1:])
        print("main label: "+str(nmax))
        print("volumes: "+str(wmvol))
        wmobj = nibabel.Nifti1Image(1.0*(wmlb==nmax+1),wmimg.affine,wmimg.header)
        nighres.io.save_volume(wmthres,wmobj)
    
        lvl = nighres.surface.probability_to_levelset(wmobj, mask_image=None,
                            save_data=True, overwrite=True, output_dir=outdir)['result']

        lvl_data = nighres.io.load_volume(lvl).get_fdata()
        sigmoid = 1.0/(1.0+numpy.exp(lvl_data/5.0))
    
        wmproba = numpy.maximum(sigmoid,wmproba)*(wmlb==nmax+1) + numpy.minimum(sigmoid,wmproba)*(wmlb!=nmax+1)
        img = nibabel.Nifti1Image(wmproba,wmimg.affine,wmimg.header)
        nighres.io.save_volume(wmthrespv,img)
    
        csfproba = nighres.io.load_volume(csfprior).get_fdata()
        csfproba = numpy.minimum(1.0-wmproba,csfproba)
        img = nibabel.Nifti1Image(csfproba,wmimg.affine,wmimg.header)
        nighres.io.save_volume(csfthrespv,img)
    

    recompute = False

    # update GM estimate for diffusion step
    gmthrespv = wmpost.replace('_wmprior.nii','_gmthrespv.nii')
    if recompute or not os.path.exists(gmthrespv):
        wmproba = nighres.io.load_volume(wmthrespv).get_fdata()
        csfproba = nighres.io.load_volume(csfthrespv).get_fdata()
    
        cbmask = nighres.io.load_volume(cbsurf).get_fdata()<20
        gmproba = nibabel.Nifti1Image(cbmask*(1.0-numpy.minimum(wmproba+csfproba,numpy.ones(wmproba.shape))),
                                    wmimg.affine,wmimg.header)
        nighres.io.save_volume(gmthrespv,gmproba)
    else:
        gmproba = nighres.io.load_volume(gmthrespv)


    # diffusion: define the midpoint between WM and GM as the better basis for topology correction
    # parameters have limited impact, use defaults
    cpd = nighres.segmentation.competing_probability_diffusion([wmthrespv,csfthrespv], gmthrespv,
                            ratio=0.1, neighbors=4, maxdiff=0.01, maxiter=100,
                            save_data=True, overwrite=False, output_dir=outdir)

    recompute=False

    # merge probability and mask distances for better topology correction basis
    wmthdiff = wmpost.replace('_wmprior.nii','_wmthdiff.nii')
    wmlvdiff = wmpost.replace('_wmprior.nii','_wmlvdiff.nii')
    if recompute or not os.path.exists(wmlvdiff):
        probas = nighres.io.load_volume(cpd['posteriors']).get_fdata()
        wmproba = nibabel.Nifti1Image(numpy.maximum(0.0, probas[:,:,:,0]),
                                        wmimg.affine,wmimg.header)
        wmthr = nibabel.Nifti1Image(wmproba.get_fdata()>=0.5,
                                        wmimg.affine,wmimg.header)
        nighres.io.save_volume(wmthdiff,wmthr)

        lvl = nighres.surface.probability_to_levelset(wmthdiff, mask_image=None,
                                save_data=True, overwrite=False, output_dir=outdir)['result']

        lvl_data = nighres.io.load_volume(lvl).get_fdata()
        sigmoid = 1.0/(1.0+numpy.exp(lvl_data/5.0))
    
        wmproba = numpy.maximum(sigmoid,2.0*sigmoid*wmproba.get_fdata())*(wmproba.get_fdata()>=0.5) \
                 +numpy.minimum(sigmoid,2.0*sigmoid*wmproba.get_fdata())*(wmproba.get_fdata()<0.5)
        img = nibabel.Nifti1Image(wmproba,wmimg.affine,wmimg.header)
        nighres.io.save_volume(wmlvdiff,img)

    orig = nighres.io.load_volume(wmlvdiff).get_fdata()

    # topology correction with other connectivities introduce singularities, use WCS
    correct = nighres.shape.topology_correction(wmlvdiff, shape_type='probability_map',
                        connectivity='wcs', propagation='object->background',
                        minimum_distance=0.0001, topology_lut_dir=None,
                        save_data=True, overwrite=False, output_dir=outdir,
                        file_name=wmlvdiff.replace('.nii','_wcs.nii'))

    topo = nighres.io.load_volume(correct['corrected']).get_fdata()
    print("WCS correction: "+str(numpy.sum((orig>0)*(orig-topo))))
