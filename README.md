# TRG-ImageProcessing
Neural network-based application for automated detection and measurement of tree-ring boundaries from coniferous species. It takes core images in a TIFF format as an input. Final measurements for each core are exported as JSON and POS files (compatible with Coo Recorder).
Processing of full core images can take long and GPU is recommended to achieve reasonable processing time.

Detailed information can be found in our publication:
Poláček, M., Arizpe, A., Hüther, P., Weidlich, L., Steindl, S., & Swarts, K. (2023). Automation of tree-ring detection and measurements using deep learning (p. 2022.01.10.475709). Methods in Ecology and Evolution. https://doi.org/10.1111/2041-210X.14183

## Run as a Docker container (using Singularity) – the easiest way
1. Install Singularity https://sylabs.io/guides/3.0/user-guide/installation.html
2. Download the container from repository (2.4 GB):
```
singularity pull docker://quay.io/treeringgenomics/image-processing:master
```
If you encounter low space related error despite plenty of space available, try to specify singularity cache and tmp:
```
export SINGULARITY_CACHEDIR=/preferred/folder/cache
export SINGULARITY_TMPDIR=/preferred/folder/tmp
```
3. Run detection on single TIFF image or folder of images modifying example script:
```
singularity run --nv image-processing_master.sif \
  --dpi=13039 \ # provided by microscope software or derived from size standard
  --run_ID=RUNID \ # subfolders in output will be named by run ID
  --input=/folder/with/images \
  --output_folder=/desired/output/folder
```
## Arguments
### Compulsary
```
--dpi
```
DPI (Dots per Inch) is necessary to convert distance from pixels to mm. This should be part of microscope metadata or derived manually from a size standard.
```
--run_ID
```
Run_ID is dataset specific identifier, a subfolder with this ID will be created in the output_folder to store all associated results.
```
--input
```
Input can be either path to an individual core image or a folder containing images for batch processing.
```
--output_folder
```
Output_folder specifies the location where to save results.

```
--weightRing
```
Retrained weights in a form of .h5 file can be passed using this argument.

### Optional
```
--cropUpandDown
```
Proportion of the top and bottom of image height of the full-size core image that should be cropped out to remove the sloping edge of the increment core or background.
```
--n_detection_rows
```
Number or rows of sliding window to use to extract squared sections for detection. More will extract smaller squares which can improve detection of narrower rings in some cases but processing time is increasing significantly. Values bigger than 4 does not seem to make sense. Default is 1.
```
--sliding_window_overlap
```
Overlap of squared sections on horizontal line. Default is 0.75
```
--min_mask_overlap
```
How many overlapping masks should be considered a good detection. Default is 3
```
--print_detections=yes
```
Saves images with detected masks and distances in “pngs” subfolder within output folder. Default is no.
