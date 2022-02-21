# TRG-ImageProcessing
Neural network-based application for automated detection and measurement of tree-ring boundaries from coniferous species. It takes core images in a TIFF format as an input. Final measurements for each core are exported as JSON and POS files (compatible with Coo Recorder).
Processing of full core images can take long and GPU is recommended to achieve reasonable processing time.

Detailed information can be found in our publication:
Poláček, M., Arizpe, A., Hüther, P., Weidlich, L., Steindl, S., & Swarts, K. (2022). Automation of tree-ring detection and measurements using deep learning (p. 2022.01.10.475709). bioRxiv. https://doi.org/10.1101/2022.01.10.475709

## Run as a Docker container (using Singularity) – the easiest way
1. Install Singularity https://sylabs.io/guides/3.0/user-guide/installation.html
2. Download the container from repository (2.4 GB):
‘singularity pull docker://quay.io/treeringgenomics/image-processing:master’
If you encounter low space related error despite plenty of space available, try to specify singularity cache and tmp:
‘export SINGULARITY_CACHEDIR=/preferred/folder/cache’
‘export SINGULARITY_TMPDIR=/preferred/folder/tmp’
3. Run detection on single TIFF image or folder of images modifying example script:
‘singularity run --nv image-processing_master.sif \
  --dpi=13039 \ # provided by microscope software or derived from size standard
  --run_ID=RUNID \ # subfolders in output will be named by run ID
  --input=/folder/with/images \
  --output_folder=/desired/output/folder
