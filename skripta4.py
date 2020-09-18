import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import signal
import timeit

print(colored('''Ovaj program efektivnost prostorne i frekvencijske konovlucije kako bi se utvrdilo koja ima brže vrijeme izvođenja. Fukcije su sadržane u SciPy.signal biblioteci''','green'))

im = cv.imread('./images/baboon.bmp',cv.IMREAD_GRAYSCALE)

gauss_kernel = np.outer(cv.getGaussianKernel(11,3),cv.getGaussianKernel(11,3)) #2D gaussov kernel dimenzija 11x11 sa standardnom devijaciom 3
plt.imshow(gauss_kernel),plt.title("gaussian kernel"),plt.xticks([]),plt.yticks([])
plt.show()

im_blurredR = signal.convolve(im,gauss_kernel,mode='same',method='direct') #izvodi konovlucije dviju N-dimenzionalnih matrica, mode='same' označava da će izlaz biti dimenzija prvog argumenta, dok method='direct' označava da želimo prostornu konovluciju, ako nije zadana, funkcija sama odabire metodu ovisno o pretpostavki koja će biti brža
im_blurredC = signal.fftconvolve(im,gauss_kernel,mode='same') #izvodi frekvencijsku konovluciju dviju N-dimenzionalnih matrica

plt.subplot(131), plt.imshow(im,cmap='gray'),plt.title("original image"),plt.xticks([]),plt.yticks([])

plt.subplot(132), plt.imshow(im_blurredR,cmap='gray'), plt.title("convolution"),plt.xticks([]),plt.yticks([])

plt.subplot(133), plt.imshow(im_blurredC,cmap = 'gray'),plt.title("DFT convolution"),plt.xticks([]),plt.yticks([])
plt.show()

f1 = cv.dft(np.float32(im),flags=cv.DFT_COMPLEX_OUTPUT)
shift = np.fft.fftshift(f1)

f2 = cv.dft(np.float32(im_blurredR),flags = cv.DFT_COMPLEX_OUTPUT)
shift2 = np.fft.fftshift(f2)

f3 = cv.dft(np.float32(im_blurredC),flags = cv.DFT_COMPLEX_OUTPUT)
shift3 = np.fft.fftshift(f3)

plt.subplot(131),plt.imshow(20*np.log(cv.magnitude(shift[:,:,0],shift[:,:,1])),cmap='gray'),plt.title("Original image"),plt.xticks([]),plt.yticks([])

plt.subplot(132),plt.imshow(20*np.log(cv.magnitude(shift2[:,:,0],shift2[:,:,1])),cmap='gray'),plt.title("Convolution"),plt.xticks([]),plt.yticks([])

plt.subplot(133), plt.imshow(20*np.log(cv.magnitude(shift3[:,:,0],shift3[:,:,1])),cmap='gray'),plt.title("DFT convolution"),plt.xticks([]),plt.yticks([])
plt.show()

#defininranje wrapera kako bi se funkcije lakše mogle predavati kao argumenti te pokrenuti nekoliko puta da se odredi prosječno vrijeme izvršavanja
def wrapper_convolve(func):
	def wrapped_convolve():
		return func(im,gauss_kernel, mode='same')
	return wrapped_convolve

wrapped_convolve = wrapper_convolve(signal.convolve)
wrapped_fftconvolve = wrapper_convolve(signal.fftconvolve)

times1 = timeit.repeat(wrapped_convolve, number=1, repeat=100) #pokretanje prostorne konvolucije 100 puta te bilježenje vremena izvršavanja
times2 = timeit.repeat(wrapped_fftconvolve, number=1, repeat=100) #pokretanje frekvencijske konovlucije 100 puta te bilježenje vremena izvršavanja

#prikaz kutijastog dijagrama
data=[times1,times2]
plt.figure(figsize=(8,6))
box = plt.boxplot(data)

plt.xticks(np.arange(3),('','convolve','fftconvolve'),size=15)
plt.yticks(fontsize=15)
plt.xlabel('convolution methods', size=15)
plt.ylabel('time taken to run', size=15)
plt.show()

