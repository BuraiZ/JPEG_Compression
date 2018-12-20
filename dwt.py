import numpy as np
import matplotlib.pyplot as plt
import cv2

def h_dwt(data):
    f1 = np.zeros((data.shape[0],int(data.shape[1]/2),data.shape[2])).astype(int)
    fh = np.zeros((data.shape[0],int(data.shape[1]/2),data.shape[2])).astype(int)
    for i in range(0,data.shape[0]):
        k=0
        for j in range(0, data.shape[1],2):
            f1[i,k]=(data[i,j]+data[i,j+1])/2
            fh[i,k]=(data[i,j]-data[i,j+1])/2
            k=k+1
        
    return { 'f1':f1, 'fh':fh }

def v_dwt(data):
    f1 = np.zeros((int(data.shape[0]/2),data.shape[1],data.shape[2])).astype(int)
    fh = np.zeros((int(data.shape[0]/2),data.shape[1],data.shape[2])).astype(int)
    for j in range(0,data.shape[1]):
        k=0
        for i in range(0, data.shape[0],2):
            f1[k,j]=(data[i,j]+data[i+1,j])/2
            fh[k,j]=(data[i,j]-data[i+1,j])/2
            k=k+1
        
    return { 'f1':f1, 'fh':fh }

def dwt(data, recurs=1):
    #image finale, compression horizontale, compression verticale, compression double
    print("First DWT Folding")
    res = h_dwt(data)
    res1_ = v_dwt(res['f1']) #f1 = f11, fh = f1h
    resh_ = v_dwt(res['fh']) #f1 = fh1, fh = fhh

    total = {'f11':res1_['f1'], 'fh1':resh_['f1'], 'f1h':res1_['fh'], 'fhh':resh_['fh']}

    current = total
    for i in range (1, recurs):
        print("Folding level")
        print(i+1)
        r = h_dwt(current['f11'])
        r1_ = v_dwt(r['f1']) #f1 = f11, fh = f1h
        rh_ = v_dwt(r['fh']) #f1 = fh1, fh = fhh

        current['f11'] = {'f11':r1_['f1'], 'fh1':rh_['f1'], 'f1h':r1_['fh'], 'fhh':rh_['fh']}
        current = current['f11']
    return total

def inverse_dwt(data):
    #trouver le niveau de recursion
    done = False
    recurs = 0
    check = data
    while (not done):
        if (type(check) == type({})):
            #un autre etage de recursion
            recurs += 1
            check = check['f11']
        else:
            #plus de dictionnaires
            done = True

    scale = 2**recurs
    orig = np.zeros((int(check.shape[0] * scale),int(check.shape[1] * scale),check.shape[2])).astype(int)

    #compressed image
    for i in range(check.shape[0]):
        for j in range(check.shape[1]):
            orig[scale*i:scale*i+scale, scale*j:scale*j+scale] += check[i, j]

    #detailing
    for r in range(1, recurs + 1):
        print("Unfolding level");print(r)
        step = 2**r
        hstep = 2**(r-1)
        for i in range(data['fhh'].shape[0]):
            for j in range(data['fhh'].shape[1]):
                #TOP-LEFT (+x error, +y error)
                orig[step*i:step*i+hstep, step*j:step*j+hstep] += data['fhh'][i, j] + data['f1h'][i, j] + data['fh1'][i, j]
                #TOP-RIGHT (-x error, +y error)
                orig[step*i+hstep:step*i+step, step*j:step*j+hstep] += - data['fhh'][i, j] - data['f1h'][i, j] + data['fh1'][i, j]
                #BOT-LEFT (+x error, -y error)
                orig[step*i:step*i+hstep, step*j+hstep:step*j+step] += - data['fhh'][i, j] + data['f1h'][i, j] - data['fh1'][i, j]
                #BOT-RIGHT (-x error, -y error)
                orig[step*i+hstep:step*i+step, step*j+hstep:step*j+step] += data['fhh'][i, j] - data['f1h'][i, j] - data['fh1'][i, j]
                
        data = data['f11']
    return orig
        

'''
image = (cv2.imread('RGB.jpg')).astype(int) * 255
image = np.float32(image) / np.max(image) #reajuster selon le maximum

img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

#sampling 4:2:0
for i in range(0, len(img_yuv), 2):
    for j in range (0, len(img_yuv[0]), 2):
        moyU = (img_yuv[i,j,1] + img_yuv[i+1,j,1] +img_yuv[i,j+1,1] +img_yuv[i+1,j+1,1]) /4
        img_yuv[i,j,1] = moyU
        img_yuv[i+1,j,1] = moyU
        img_yuv[i,j+1,1] = moyU
        img_yuv[i+1,j+1,1] = moyU
        moyV = (img_yuv[i,j,2] + img_yuv[i+1,j,2] +img_yuv[i,j+1,2] +img_yuv[i+1,j+1,2]) /4
        img_yuv[i,j,2] = moyV
        img_yuv[i+1,j,2] = moyV
        img_yuv[i,j+1,2] = moyV
        img_yuv[i+1,j+1,2] = moyV

y, u, v = cv2.split(img_yuv)
cv2.imshow('img32', img_yuv)
cv2.imshow('y', y)
cv2.imshow('u', u)
cv2.imshow('v', v)

img = img_yuv

f1 = np.zeros((len(img),int(len(img[0])/2)))
for i in range(0,len(img)):
    k=0
    for j in range(0, len(img[0]),2):
        f1[i,k]=(img[i,j]+img[i,j+1])/2
        k=k+1;

plt.imshow(v, cmap = plt.get_cmap('gray'))

plt.show()
fh = (image[:,::2] - image[:,1::2])/2  #Avec indexation avancée plutôt qu'avec des for. Wow!!!!

plt.imshow(fh, cmap = plt.get_cmap('gray'))
f11 = (f1[::2,:] + f1[1::2,:])/2

plt.imshow(f11, cmap = plt.get_cmap('gray'))
plt.show()

f1h = np.zeros((int(len(image)/2),int(len(image[0])/2)))
k=0
for i in range(0,len(f1),2):
    for j in range(0, len(f1[0])):
        f1h[k,j]=(f1[i,j]-f1[i+1,j])/2
    k=k+1;

plt.imshow(f1h, cmap = plt.get_cmap('gray'))
plt.show()

fh1 = np.zeros((int(len(image)/2),int(len(image[0])/2)))
k=0
for i in range(0,len(fh),2):
    for j in range(0, len(fh[0])):
        fh1[k,j]=(fh[i,j]+fh[i+1,j])/2
    k=k+1;

plt.imshow(fh1, cmap = plt.get_cmap('gray'))
plt.show()

fhh = np.zeros((int(len(image)/2),int(len(image[0])/2)))
k=0
for i in range(0,len(fh),2):
    for j in range(0, len(fh[0])):
        fhh[k,j]=(fh[i,j]-fh[i+1,j])/2
    k=k+1;

plt.imshow(fhh, cmap = plt.get_cmap('gray'))
plt.show()

original = np.zeros((len(image),len(image[0])))
#copie des pixels de f1. Met les mêmes valeurs de f1 pour i et i+1
for i in range(0,len(f1)):
    for j in range(0, len(f1[0])):
        original[i,2*j]=f1[i,j]
        original[i,2*j+1]=f1[i,j]


plt.imshow(original, cmap = plt.get_cmap('gray'))
plt.show()

#Ajout des hautes fréquences
for i in range(0,len(fh)):
    for j in range(0, len(fh[0])):
        original[i,2*j] += fh[i,j]
        original[i,2*j+1] -= fh[i,j]


plt.imshow(original, cmap = plt.get_cmap('gray'))
plt.show()

np.max(image - original)
'''
