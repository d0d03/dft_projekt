import sys
from termcolor import colored
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main(argv):
	help()
	filepath = argv[0] if len(argv)>0 else './images/house.png'
	I = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
	if I is None:
		print('error opening image')
		return -1

	dft = cv.dft(np.float32(I),flags = cv.DFT_COMPLEX_OUTPUT)#frekvencijska domena je puno veća od prostorne pa je dobro koristiti float, zastavica označava, zastavica označava da želimo izlaz u obliku kompleksne matrice istih dimenzija kao i ulaz
	dft_shift = np.fft.fftshift(dft)#premiještanje 0-freq. (DC) u sredinu slike 

	mag = cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]) # apsolutna vrijednost spektra (np.abs(dft) = np.sqrt(np.square(Re(I)) + np.square(Im(I))))
	pha = cv.phase(dft_shift[:,:,0],dft_shift[:,:,1]) # faza = np.arctan(Im(I)/Re(I))

	back_shift = np.fft.ifftshift(dft_shift)#inverz od fftshift
	recon = cv.dft(back_shift, flags= cv.DFT_REAL_OUTPUT | cv.DFT_INVERSE)#zastavice označavaju da želimo inverznu DFT te da izlaz treba biti realna matrica iste veličine kao i ulazna kompleksna matrica

	#prikaz pomoću matplotlib
	plt.subplot(221)
	plt.imshow(I,cmap='gray'),plt.title('Input image'),plt.xticks([]),plt.yticks([])
	plt.subplot(222)
	plt.imshow(np.log(mag), cmap='gray'),plt.title('Magnitude'),plt.xticks([]),plt.yticks([])
	plt.subplot(223)
	plt.imshow(pha,cmap='gray'),plt.title('Phase'),plt.xticks([]),plt.yticks([])
	plt.subplot(224)
	plt.imshow(recon,cmap='gray'),plt.title('Output image(reconstructed)'),plt.xticks([]),plt.yticks([])
	plt.show()
	
	
def help():
	print(colored('''Ovaj program demonstrira primejnu DFT, prikazuje originalnu sliku, njenu magnitudu i fazu te rekonstuiranu sliku koristeći inverznu DFT.
	
	Upute: skripta1.py putanja_slike -- default house.png ''','green'))		
		
if __name__=="__main__":
	main(sys.argv[1:])
