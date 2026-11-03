# cerebellar-cortex
Specialized pipelines for the segmentation of cerebellar 
surfaces at ultra-high resolution in-vivo and post-mortem
used in the article:

"Mesoscale imaging of the human cerebellum reveals converging
regional specialization of its morphology, vasculature, and 
cytoarchitecture." Priovoulos, N., Bazin, P.-L., Brouwer, 
E.J.P., Mejias, J.F., De Buck, M.H.S., Alkemade, A., Van Der Zwaag, 
W., Caan, M.W.A., 2026.Proc. Natl. Acad. Sci. U.S.A. 123, e2516608123. 
https://doi.org/10.1073/pnas.2516608123

The processing follows the same general pipeline for all 
data sets, but some steps have been customized in order
to work with different types of data sources in the 
post-mortem data sets (customized scripts are included
under the post-mortem/ sub-folder).

The scripts may be adapted to new data by changing the
input paths and data type specifications.

An example unified script for cerebellar segmentation
based on this pipeline is also to be integrated in the
next release of the Nighres toolbox.
