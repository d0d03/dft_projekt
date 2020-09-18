from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import signal

print(colored('''Ovaj program demonstrira primjenu pojasne brane kao filtera. Učitana je slika parrot.png te joj je dodan periodični šum, nakon čega je primjenjena pojasna brana kako bi se on uklonio. ''','green'))
I = np.mean(cv.imread("./images/parrot.png"),axis=2)

dft = cv.dft(np.float32(I),flags=cv.DFT_COMPLEX_OUTPUT)
shift = np.fft.fftshift(dft)
magnitude = cv.magnitude(shift[:,:,0],shift[:,:,1])

noiseIm = np.copy(I)

for n in range(noiseIm.shape[1]):
	np.add(noiseIm[:,n],100*np.cos(0.1*np.pi*n),out=noiseIm[:,n],casting='unsafe') #dodavanje periodičnog šuma slici
	
noiseDft = cv.dft(np.float32(noiseIm),flags= cv.DFT_COMPLEX_OUTPUT)
noiseDftShift = np.fft.fftshift(noiseDft)
noiseMag = cv.magnitude(noiseDftShift[:,:,0],noiseDftShift[:,:,1])

 
plt.subplot(221),plt.imshow(I,cmap='gray'),plt.title("original image"),plt.axis('off')
plt.subplot(222),plt.imshow(20*np.log(magnitude),cmap='gray'),plt.title("original magnitude")
plt.xticks(np.arange(0,I.shape[1],50))
plt.yticks(np.arange(0,I.shape[0],25))
plt.subplot(223),plt.imshow(noiseIm,cmap='gray'),plt.title("Noisy image"),plt.axis('off')
plt.subplot(224),plt.imshow(20*np.log(noiseMag),cmap='gray'),plt.title("Noisy magnitude")
plt.xticks(np.arange(0,noiseIm.shape[1],50))
plt.yticks(np.arange(0,noiseIm.shape[0], 25))
plt.show()

noiseDftShift[170:176,:220] = noiseDftShift[170:176,230:] = 0 #blokiranje frekvencija za koje se pretpostavlja da predstavljaju šum
recon = cv.dft(np.fft.ifftshift(noiseDftShift),flags= cv.DFT_REAL_OUTPUT | cv.DFT_INVERSE)
plt.imshow(recon,cmap='gray'),plt.title("Reconstructed image"),plt.axis('off')
plt.show()




