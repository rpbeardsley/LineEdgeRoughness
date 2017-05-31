
import numpy as np
import cv2 #(OpenCV3)
from scipy.interpolate import interp1d
from sklearn import linear_model

class edge_roughness:
    
    #analyse an image
    def LER_analysis(self, image, im_wdth, thresh): #origional image, width in meters of image (ie the scale), threshold for outliers
        
        bin_im = self.binarization(image) #binarize the image
        
        cv2.imwrite('binary_test.jpg', bin_im) #write the binary image to a file
        
        X = np.linspace(0, im_wdth, num=bin_im.shape[1], endpoint=True) #create the X data array, width of image and number pixels
        Y = self.binary_imsearch(255, bin_im) #find the line edge values
        #plt.plot(X, Y, 'r')
        #plt.show()
        
        Xcln, Ycln = self.extract_clean(X, Y, thresh) #extract the cleaned data from the raw edge data
        
        freq, FourierPow = self.fourier_power(Xcln, Ycln)
        
        return Xcln, Ycln, freq, FourierPow
        
        
    # binarize the image    
    def binarization(self, image):
       
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #Convert the loaded image to greyscale, only required for openCV version 2 I am using 3 (confusingly also imoprts as cv2, as it's in beta)
        
        #Binarise image and compare methods
        blur = cv2.GaussianBlur(image,(5,5),0) #gaussian filter to remove noise
        ret1,th1 = cv2.threshold(blur,15,255,cv2.THRESH_BINARY) #manual threshold value
        ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #Otsu method
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #Otsu method following filtering
        retBin,binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #repeat above, oddly can't just copy above to new variable for storage
        
        #==============================================================================
        # #plot all the binarisation methods and histograms for comparison
        # images = [imageA, 0, th1,
        #           imageA, 0, th2,
        #           blur, 0, th3]
        # titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
        #           'Original Noisy Image','Histogram',"Otsu's Thresholding",
        #           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
        # 
        # for i in range(3):
        #     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        #     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        #     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        #     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        #     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        #     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
        # plt.show()
        #==============================================================================
                                     
        cv2.imwrite('binary_test.jpg', th1) #write the binary image to a file
        
        return th1
        
        
    #search an image of pixel values for the black white transition
    def binary_imsearch(self, x, arr): 
    
        cl = 0
        fin = np.size(arr,1)
        Out = []
        
        for cl in range(0,fin,1):
            V = self.binary_Search(255,arr[:,cl])
            Out.append(V)
        
        Out = np.asarray(Out)
        Out = Out.astype(float)
        
        return Out
            
    
    #search a column of pixel values for the black white transition
    def binary_Search(self, x, arr): 
            
        lo = 0
        hi = np.size(arr,0)    
        while lo < hi:
            mid = (lo + hi) // 2
    
            if arr[mid] < x:
                lo = mid + 1
            elif arr[mid] > x:
                hi = mid
            elif mid > 0 and arr[mid-1] == x:
                hi = mid
            else:
                return mid   
        return -1
   
    
    #remove any outliers form the data
    def remove_outliers(self, Xdata, Ydata, n):
        
        #remove the outliers
        d = np.abs(Ydata - np.median(Ydata))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        YdataClean = Ydata[s<n]
        XdataClean = Xdata[s<n]
        
        #Interpolate the data to put it on an even mesh
        finterp = interp1d(XdataClean,YdataClean)
        xnew = np.linspace(0, max(Xdata), num=600, endpoint=True)
        ynew = finterp(xnew)  
        
        return xnew, ynew
    
    
    #remove the background with a regression model
    def remove_background(self, Xdata, Ydata): #x data, ydata
        
        Xdata = Xdata[:, np.newaxis]  #add axis for the regression algorithm
        regr = linear_model.LinearRegression() #instance of lin regr class
        regr.fit(Xdata,Ydata) #fit to the data
        Ymod = regr.predict(Xdata) #predict Y in the Xdata range using the model 
        
        #print some paramteres for the regression model
        #print("Coefficients: \n" + str(regr.coef_)) #coefficients
        #print("Mean squared error: %.2f" % np.mean((regr.predict(Xdata) - Ydata) ** 2)) #mean squared error
        #print('Variance score: %.2f' % regr.score(Xdata, Ydata)) #explained variance score: 1 is perfect prediction
        
        Ydata = np.subtract(Ydata,Ymod)     
        
        return Ydata
    
    
    #clean the data
    def extract_clean(self, X, Y, stds): #x data, y data, threshold for outliers

        Xclean, Yclean = self.remove_outliers(X,Y,stds) #remove the outliers
        #plt.plot(Xclean, Yclean, 'r')
        #plt.show()

        Ytrans = self.remove_background(Xclean,Yclean) #remove background with a regression model
        #plt.plot(Xclean, Ytrans, 'r')
        #plt.show()
        
        return Xclean, Ytrans
    
    
    #Fourier transform the data and convert to power spectrum 
    def fourier_power(self, Xdata, Ydata):
        
        sp = np.fft.rfft(Ydata) #FFT
        freq = np.fft.rfftfreq(Xdata.shape[0]) / (max(Xdata)/Xdata.shape[0]) #(distance/pixel) / (maxdistance/Pixel) i.e. normalised                     
        FourierPower = np.abs(sp) / (Xdata.shape[0])*2 #convert to the power spectrum
        
        #plot the fourier spectra
        #plt.plot(freq[2:100], sp.real[2:100]) #real part
        #plt.show()
        #plt.plot(freq[2:100], FourierPower[2:100],'-')
        #ax = plt.gca()
        #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
        #plt.xlabel('cycles (m$^{-1}$)')
        #plt.ylabel('Fourier power (au)')
        #plt.title('Fourier power spectrum')
        #plt.show()    
        
        return freq, FourierPower
        
        
        
        
        
        
        
        
        
        
        
        
        