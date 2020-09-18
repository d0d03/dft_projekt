import sys
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import signal


def main(argv):
	help()
	filepath = argv[0] if len(argv)>0 else './images/tigers.jpeg'
	size = np.int(argv[1]) if len(argv)> 1 else np.int(11)
	sigma1 = np.float(argv[2]) if len(argv)> 2 else np.float(5)
	sigma2 = np.float(argv[3]) if len(argv)> 3 else np.float(0.005)	
	
	I = cv.imread(filepath,cv.IMREAD_GRAYSCALE)
	if I is None:
		print('error loading image')
		return -1	
	
	plt.imshow(I,cmap='gray'),plt.xticks([]),plt.yticks([])
	plt.show()
	
	gauss_k1 = np.outer(cv.getGaussianKernel(size,sigma1),cv.getGaussianKernel(size,sigma1)) #kreiranje prvog gaussovog kernela dimenzija sizeXsize sa standardnom devijacijom sigma1
	gauss_k2 = np.outer(cv.getGaussianKernel(size,sigma2),cv.getGaussianKernel(size,sigma2)) #kreiranje drugog gaussovog kernela
	DoG = gauss_k1 - gauss_k2 #izračunavanje razlike guassa
	
	#prikaz kernela
	plt.subplot(131)
	plt.imshow(gauss_k1,cmap='gray'),plt.title("first kernel"),plt.xticks([]),plt.yticks([])
	plt.subplot(132)
	plt.imshow(gauss_k2,cmap='gray'),plt.title("second kernel"),plt.xticks([]),plt.yticks([])
	plt.subplot(133)
	plt.imshow(DoG,cmap='gray'), plt.title("DoG"),plt.xticks([]),plt.yticks([])
	plt.show()
	
	out = signal.fftconvolve(I,DoG,mode='same') #primjena kernela na sliku
	dft = cv.dft(np.float32(out),flags= cv.DFT_COMPLEX_OUTPUT)
	shift = np.fft.fftshift(dft)
	
	#prikaz rezultante slike
	plt.subplot(121)
	plt.imshow(out,cmap='gray'),plt.title("output image"),plt.xticks([]),plt.yticks([])
	plt.subplot(122)
	plt.imshow(20*np.log(cv.magnitude(shift[:,:,0],shift[:,:,1]))),plt.title("magnitude of output image"),plt.xticks([]),plt.yticks([])
	plt.show()
	

def help():
	print(colored('''Ovaj program demonstrira primjenu pojasnopropusnog filtera (BPF), pri čemu su korišteni Gaussovi kerneli te je uzeta njihova razlika (ovo se još naziva DoG (difference of gaussians)). Na ovaj način propuštaju se frekvencije u određenom pojasu odnosno između dvije granične frekvencije
	
	Upute: skripta5.py putanja_slike veličina_gaussovih_kernela signam1 sigma2 -- default tigers.jpeg 11  5 0.005''','green'))		
		
if __name__=="__main__":
	main(sys.argv[1:])
