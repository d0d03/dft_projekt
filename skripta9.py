import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np
from matplotlib.image import imread
import cv2 as cv

print(colored('''Ovaj program demonstrira primjenu DFT pri kompresiji slike. Prikazane su kompresije u iznosu od 90%, 95%, 99% te 99.8% ''','green'))
I = cv.imread('./images/house2.png',cv.IMREAD_GRAYSCALE)
cv.imshow("",I)
cv.waitKey(0)

freq = np.fft.fft2(I) #preslik u frekvencijsku domenu pomoću ugrađene brze fourierove transformacije unutar NumPy biblioteke
freqSort = np.sort(np.abs(freq)) #pretvaramo frekvencijsku matricu u vektor koji poredamo prema veličinama kako bi lakše bilo odbaciti nepotrebne frekvencije

for keep in (0.1,0.05,0.01,0.002):
	tresh = freqSort[int(np.floor((1-keep)*len(freqSort)))] #inicijalizacija tresholda
	ind = np.abs(freq)>tresh #zadržavanje samo frekvencija koje su iznad trehsolda
	freqlow = freq * ind #pošto matrica sadrži samo 0 (frekvencije koje odbacujemo) i 1(frekvencije koje zadržavamo) množimo ju sa frekvencijskm matricom kako bi zadržali željene vrijednosti
	Alow = np.fft.ifft2(freqlow).real #inverz brze fourierove transformacije
	plt.imshow(Alow,cmap='gray'), plt.title("compressed image: keep= " + str(keep*100) + "%") #prikaz kompresirane slike
	plt.axis("off")
	plt.show()
cv.destroyAllWindows()

