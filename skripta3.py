import sys
from termcolor import colored
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imutils


def main(argv):
	help()
	filepath = argv[0] if len(argv)>0 else './images/imageTextR.png'
	I = cv.imread(filepath,cv.IMREAD_GRAYSCALE)
	if I is None:
		print('error opening image')
		return -1
		
	cv.imshow("rotated image",I)
	cv.waitKey(0)

	x,y = np.shape(I)

	dft = cv.dft(np.float32(I-I.mean()),flags = cv.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	mag = (cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

	plt.imshow(np.log(mag),cmap='gray'),plt.xticks([]),plt.yticks([])
	plt.show()

	indices = np.where(mag>=np.mean(mag)) #izdvajanje samo magnituda većih ili jednakih srednjoj vrijednosti
	
	maxx = (indices[0][0] - x/2)/x	#određivanje koordinata najutjecajnijih frekvencija kako bi se odredio kut
	maxy = (indices[1][0] - y/2)/y

	alpha = np.arctan(maxy/maxx)*180/np.pi #izračun kuta
	print("rotation angle = ", alpha)

	recon = imutils.rotate_bound(I,alpha)	#rotacija originalne slike za dobiveni kut
	cv.imshow("rotated back",recon)	#prikaz
	cv.waitKey(0)
	cv.destroyAllWindows()

def help():
	print(colored('''Ovaj program demonstrira kako odrediti kut za koji je slika zarotirana te rotira ulaznu sliku za dobiveni kut, prikazujući sliku u pravilnoj orijentaciji.
	
	Upute: skripta3.py putanja_slike -- default imageTextR.png ''','green'))		
		
if __name__=="__main__":
	main(sys.argv[1:])


