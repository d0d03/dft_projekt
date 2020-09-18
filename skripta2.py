import sys
from termcolor import colored
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main(argv):
	help()
	filepath1,filepath2 = argv if argv==0 else ('./images/house.png','./images/house2.png')	
		
	I = cv.imread(filepath1, cv.IMREAD_GRAYSCALE)
	I2 = cv.imread(filepath2,cv.IMREAD_GRAYSCALE)
	
	if I is None or I2 is None:
		print("error opening image")
		return -1
	
	plt.subplot(121)
	plt.imshow(mixMagPha(I,I2),cmap='gray'),plt.title("Re(I1) + Im(I2)"),plt.xticks([]),plt.yticks([])
	plt.subplot(122)
	plt.imshow(mixMagPha(I2,I),cmap='gray'),plt.title("Re(I2) + Im(I1)"),plt.xticks([]),plt.yticks([])
	plt.show()
	
def help():
	print(colored('''Ovaj program demonstrira kako je faza slike bitna za rekonstrukciju iako prikazivanje faze ne daje previše informacija.
	
	Upute: skripta1.py putanja_slike1, putanja_slike2 -- default house.png house2.png ''','green'))	

def mixMagPha(img1,img2):

	dft1 = cv.dft(np.float32(img1),flags = cv.DFT_COMPLEX_OUTPUT) 
	dft2 = cv.dft(np.float32(img2),flags = cv.DFT_COMPLEX_OUTPUT)
	
	
	shift1 = np.fft.fftshift(dft1)
	shift2 = np.fft.fftshift(dft2)
	re1 = cv.split(shift1)	#razdvajanje realnih i imagniarnih vrijednosti prve slike
	re2 = cv.split(shift2)	#razdvajanje realnih i imaginarnih vrijednosti druge slike
	mix = cv.merge([re1[0],re2[1]]) #kombinacija realnog dijela jedne te imaginarnog dijela druge slike
	ishift = np.fft.ifftshift(mix)
	recon = cv.dft(ishift,flags=cv.DFT_REAL_OUTPUT | cv.DFT_INVERSE)
	return recon
		
if __name__=="__main__":
	main(sys.argv[1:])
