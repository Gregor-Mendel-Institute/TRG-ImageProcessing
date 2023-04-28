# cd /Users/miroslav.polacek/Dropbox\ \(VBC\)/'Group Folder Swarts'/Research/CNNRings/Mask_RCNN/preprocessImages

from PIL import Image
import skimage
from skimage import exposure, img_as_ubyte
import os.path
#import matplotlib
#matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from shapely.geometry import box
#matplotlib.rcParams['interactive'] == True

pathin = "/Volumes/Storage/TestForPosterTRACE/Quercus/training"
pathout = "/Volumes/Storage/TestForPosterTRACE/Quercus/training/Quercus_cropped"
#pathin = "/Volumes/swarts/lab/DendroImages/00008JPEG"
#pathout = "/Volumes/swarts/lab/DendroImages/ToAnnotate"

def CutTrain(pathin = pathin, pathout = pathout, overlap = 0):

    for f in os.listdir(pathin):
        supported_extensions = ['.tif', '.tiff']
        file_extension = os.path.splitext(f)[1]
        if file_extension in supported_extensions:

            print(f)
            im = skimage.io.imread(os.path.join(pathin,f))
            """
            # check number of channels and if 4 assume rgba and convert to rgb
            # conversion if image is not 8bit convert to 8 bit
            if im.dtype == 'uint8' and im.shape[2] == 3:
                print("Image was 8bit and RGB")
            elif im.shape[2] == 4:
                print("Image has 4 channels, assuming RGBA, trying to convert")
                im = img_as_ubyte(skimage.color.rgba2rgb(im))
            elif im.dtype != 'uint8':
                print("Image converted to 8bit")
                im = img_as_ubyte(exposure.rescale_intensity(im))  # with rescaling should be better
            """
            imgheight, imgwidth = im.shape[:2]
            print("Image dimensions", imgheight, imgwidth)
            for i in range(0, imgwidth, int(imgheight-(imgheight*overlap))):
                print("i", i)
                #b = (i, 0, (i+imgheight), imgheight)

                """
                bplot = box(*b) #next 3 rows from here i just try to plot box on the image
                x,y = bplot.exterior.xy
                print(x,y)
                plt.plot(x,y)
                plt.imshow(im) #to plot an image that is being processed
                """
                a = im[:, i:i+imgheight, :]

                # conversion if image is not 8bit convert to 8 bit
                if a.dtype == 'uint8' and a.shape[2] == 3:
                    print("Image was 8bit and RGB")
                elif a.shape[2] == 4:
                    print("Image has 4 channels, assuming RGBA, trying to convert")
                    a = img_as_ubyte(skimage.color.rgba2rgb(a))
                elif a.dtype != 'uint8':
                    print("Image converted to 8bit")
                    a = img_as_ubyte(exposure.rescale_intensity(a))  # with rescaling should be better

                skimage.io.imsave(fname=os.path.join(pathout, str(i) + '_' + f), arr=a)

if __name__ == "__main__":
    CutTrain()
