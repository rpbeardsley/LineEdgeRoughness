

import cv2 #(OpenCV3)
import argparse
from LER import edge_roughness
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--Image", required=True,
	help="path to the first image")
args = vars(ap.parse_args())

image = cv2.imread(args["Image"])

#image = cv2.imread("C:/Users/ppzrb/Documents/Quantum technology/Code/line_edge_roughness/Spec_test.bmp")

LER = edge_roughness()
(Xcln, Ycln, freq, FourierPow) = LER.LER_analysis(image, 50e-6, 2) #image, im_wdth, thresh


plt.plot(Xcln, Ycln, '-')
plt.plot(Xcln, Ycln, 'r.')
plt.show()

plt.plot(freq[2:100], FourierPow[2:100],'-')
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
plt.xlabel('cycles (m$^{-1}$)')
plt.ylabel('Fourier power (au)')
plt.title('Fourier power spectrum')
plt.show() 
#stitcher = Stitcher()
#(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

cv2.imshow('Origional Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()