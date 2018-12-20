import dwt
import lzw

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-r", type=int, nargs='?', default=1)
parser.add_argument("-s", type=int, nargs='?', default=1)
parser.add_argument("-d", type=int, nargs='?', default=0)
parser.add_argument("-i", type=str, nargs='?', default="RGB.jpg")
parser.add_argument("-o", type=str, nargs='?', default=None)
args = parser.parse_args()

def RGBtoYUV(img):
    res = np.copy(img)
    res[:,:,[0]] = (img[:,:,[0]] + img[:,:,[1]] + img[:,:,[1]] + img[:,:,[2]])/4
    res[:,:,[1]] = img[:,:,[2]] - img[:,:,[1]]
    res[:,:,[2]] = img[:,:,[0]] - img[:,:,[1]]
    return res

def YUVtoRGB(img):
    res = np.copy(img)
    res[:,:,[1]] = img[:,:,[0]] - (img[:,:,[1]] + img[:,:,[2]])/4
    res[:,:,[0]] = img[:,:,[2]] + res[:,:,[1]]
    res[:,:,[2]] = img[:,:,[1]] + res[:,:,[1]]
    return res

def quant(data,dead,step):
    #Dead-zone: 
    result = np.clip((data - dead), 0, 255)*255/(255 - dead)

    #Step-sizing
    result = (result/step).astype(int) * step
    return result

def to_1d(data):
    vec = np.array([0, 0, 0], dtype=int)
    #trouver le niveau de recursion
    done = False
    recurs = 0
    check = data
    while (not done):
        if (type(check) == type({})):
            #un autre etage de recursion
            recurs += 1
            vec = np.append(vec, np.reshape(check['f1h'],-1))
            vec = np.append(vec, np.reshape(check['fh1'],-1))
            vec = np.append(vec, np.reshape(check['fhh'],-1))
            check = check['f11']
        else:
            #plus de dictionnaires
            vec = np.append(vec, np.reshape(check,-1))
            done = True

    vec[0] = recurs
    vec[1] = check.shape[0] * 2**recurs
    vec[2] = check.shape[1] * 2**recurs
    
    return vec.astype(int)

def from_1d(vec):
    recurs = vec[0]
    w = vec[1]
    h = vec[2]
    data = {}
    offset = 3
    if (recurs == 0):
        data = np.reshape(vec[offset:],(w, h, -1))
    else:
        data = {'f11':0, 'f1h':0, 'fh1':0, 'fhh':0}
        current = data
        for i in range(1, recurs+1):
            scale = 2**i
            s_w = int(w/scale)
            s_h = int(h/scale)
            length = int(s_w * s_h * 3)
            current['f1h'] = np.reshape(vec[offset:length+offset:1],(s_w, s_h, -1))
            offset += length
            current['fh1'] = np.reshape(vec[offset:length+offset:1],(s_w, s_h, -1))
            offset += length
            current['fhh'] = np.reshape(vec[offset:length+offset:1],(s_w, s_h, -1))
            offset += length
            if (i == recurs):   
                current['f11'] = np.reshape(vec[offset:length+offset:1],(s_w, s_h, -1))
                offset += length
            else:
                current['f11'] = {'f11':0, 'f1h':0, 'fh1':0, 'fhh':0}
                current = current['f11']
            print(offset)

        
    return data

def main(rec, step, dead, img, out):
    image = (cv2.imread(img, 1)).astype(int)
    w = image.shape[0]
    h = image.shape[1]
    
    image[:,:,[0,1,2]] = image[:,:,[2,1,0]] #To RGB
    
    cv2.imshow('img',256*image[:,:,[2,1,0]]) #Il faut inverser, car cv2 interprète RGB dans l'ordre BGR
    #somehow aussi il faut multiplier par 256

    img_yuv = RGBtoYUV(image)
    
    #cv2.imshow('yuv',256*img_yuv)
    #sampling 4:2:0
    for i in range(0, len(img_yuv), 2):
        for j in range (0, len(img_yuv[0]), 2):
            moyU = (img_yuv[i,j,1]/4 + img_yuv[i+1,j,1]/4 +img_yuv[i,j+1,1]/4 +img_yuv[i+1,j+1,1]/4)
            img_yuv[i,j,1] = moyU
            img_yuv[i+1,j,1] = moyU
            img_yuv[i,j+1,1] = moyU
            img_yuv[i+1,j+1,1] = moyU
            moyV = (img_yuv[i,j,2]/4 + img_yuv[i+1,j,2]/4 +img_yuv[i,j+1,2]/4 +img_yuv[i+1,j+1,2]/4) 
            img_yuv[i,j,2] = moyV
            img_yuv[i+1,j,2] = moyV
            img_yuv[i,j+1,2] = moyV
            img_yuv[i+1,j+1,2] = moyV
        
    #cv2.imshow('yuv_s',256*img_yuv)
    
    result = dwt.dwt(img_yuv, rec)
    # fonction dwt travaille recursivement
    # result est un dictionnaire avec ces valeurs:
    # / /f11  f1h\          \
    # | |        |      f1h |
    # | \fh1  fhh/          |
    # |                     |
    # | fh1             fhh |
    # \                     /
    # f11 se "deroule" selon le niveau de recursion souhaite

    # LZW Compression
    v1d = to_1d(result)
    info = v1d[0:3]
    #Quantification
    q1d = quant(v1d[3:], dead, step)
    print("Compression LZW")
    print("length original : ", len(q1d))
    vecLZW = lzw.encode(q1d)

    # LZW Decompression
    print("Decompression LZW")
    vecDeLZW = lzw.decode(vecLZW)
    print("length reconstructed : ", len(vecDeLZW))
    v3d = from_1d(np.append(info,vecDeLZW))

    reconstructed = dwt.inverse_dwt(v3d)

    #cv2.imshow('rec',256*reconstructed)

    recolored = YUVtoRGB(reconstructed)
    cv2.imshow('col',256*recolored[:,:,[2,1,0]])

    cv2.imshow('qnt',256*quant(recolored[:,:,[2,1,0]], dead, step))

    if (out is not None):
        cv2.imwrite(out, recolored[:,:,[2,1,0]])

    cv2.waitKey(10000)
    
if __name__ == "__main__":
    main(args.r, args.s, args.d, args.i, args.o)
