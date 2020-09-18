import sys
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import signal


def main(argv):
	help()
	filepath = argv[0] if len(argv)>0 else './images/lenna.bmp'
	co = np.int(argv[1]) if len(argv)> 1 else np.int(10)
	
	I = cv.imread(filepath,cv.IMREAD_GRAYSCALE)
	if I is None:
		print('error loading image')
		return -1	
	
	plt.imshow(I,cmap='gray'),plt.xticks([]),plt.yticks([])
	plt.show()

	def HPF(image, cut_off_value):
		freq = cv.dft(np.float32(image),flags=cv.DFT_COMPLEX_OUTPUT)
		w,h = freq.shape[0],freq.shape[1] 
		half_w, half_h = int(w/2), int(h/2)
		co = cut_off_value
		freq1 = np.copy(freq)
		freq2 = np.fft.fftshift(freq1)
		
		plt.imshow(20*np.log(cv.magnitude(freq2[:,:,0],freq2[:,:,1])))
		plt.show()

		freq2[half_w - co:half_w+co+1,half_h-co:half_h+co+1]=0 #zadržavamo sve osim niskih frekvencija ispod granične vrijednosti
		plt.imshow(20*np.log(cv.magnitude(freq2[:,:,1],freq2[:,:,0]))),plt.xticks([]),plt.yticks([])
		plt.show()

		im1 = cv.dft(np.fft.ifftshift(freq2),flags = cv.DFT_REAL_OUTPUT | cv.DFT_INVERSE)
		plt.imshow(im1,cmap='gray'),plt.xticks([]),plt.yticks([])
		plt.show()
	
	HPF(I,co)

def help():
	print(colored('''Ovaj program demonstrira primjenu High-Pass filtera (HPF) sa zadanom graničnom vrijednosti, na taj način dobivamo izlazne slike na kojima su izraženiji rubovi.
	
	Upute: skripta5.py putanja_slike cutoff_vrijednost -- default lenna.bmp 10 ''','green'))	
		
if __name__=="__main__":
	main(sys.argv[1:])
