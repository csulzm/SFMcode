from tokenize import maybe
import cv2
import pylab as plt
from matplotlib import pyplot as plt

from utils.matchSift import matchSIFTdescr
from utils.loadDatacsv import load

import numpy as np

sift = cv2.SIFT_create(contrastThreshold=0.04)

def drawMatches(img1, kp1, img2, kp2, indicesI,indicesJ ):

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage /  r = y-axis  c = x-axis
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros( (max([rows1,rows2]),cols1+cols2,3 ), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for i in range(len(indicesI)):
        (i,j) = indicesI[i],indicesJ[i]

        # x - columns
        # y - rows
        if type(kp1[0]) == np.ndarray:
            (x1, y1) = kp1[i]
            (x2, y2) = kp2[j]
            (y1, x1) = kp1[i]
            (y2, x2) = kp2[j]
        else:
            (x1,y1) = kp1[i].pt
            (x2,y2) = kp2[j].pt

        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

def algoritmo1Match(desA,desB):
    #Make sure that the format of Desa and Debs is NP floating point 32
    desA = desA.astype(np.float32)
    desB = desB.astype(np.float32)

    # BFMatcher with default params
    
    bf = cv2.BFMatcher()
    
    matches = bf.knnMatch(desA, desB, k=2) #Return k = 2 matches for each feature point

    # Apply ratio test
    
    salidaI = []
    salidaJ = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            salidaI.append(m.queryIdx)
            salidaJ.append(m.trainIdx)
            
    return salidaI,salidaJ

def algoritmo2Match(desA,desB):
    
    indicesI, indicesJ = matchSIFTdescr(desA,desB) #Serial number I and J of feature points
    
    #I assume that the distance between frames is 1 (may change?)
    minNeighboringMatching = 15

    toleranciaRelajada = [2, 0.999, 0.90, 0.8, 0.7]

    # Filtering is at least mineigboringmatching or error
    while (len(indicesI) < minNeighboringMatching):
        if len(toleranciaRelajada) == 0:
            raise Exception(" The minimum amount of Minneapolis examination is not reached", len(indicesI),
                            " The minimum value is ", minNeighboringMatching)
        nuevaTol = toleranciaRelajada.pop()
        indicesI, indicesJ = matchSIFTdescr(desA, desB, nuevaTol)
        print ("Too few matching points will relax the right", nuevaTol, " habian ", len(indicesI))
        
    print ("Realized ", len(indicesI), " Matching point")
    return indicesI,indicesJ

def getPairSIFT(imageA,imageB,show=False):

    algoritmoMatch = algoritmo2Match

    # calcula descriptores sift
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    kpA, desA = sift.detectAndCompute(grayA, None)

    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    kpB, desB = sift.detectAndCompute(grayB, None)
    
    print ("Sift generated for A and B ",desA.shape[0],desB.shape[0]," descriptores")

    # Test code
    # kpA, desA = load("data/descriptLocI.csv"),load("data/descI.csv")
    # kpB, desB = load("data/descriptLocJ.csv"),load("data/descJ.csv")

    if show:
        imga = cv2.drawKeypoints(grayA, kpA)
        imgB = cv2.drawKeypoints(grayB, kpB)
        cv2.imshow('ImageWindowA', imga)
        cv2.imshow('ImageWindowB', imgB)
        cv2.waitKey()

    indI,indJ = algoritmoMatch(desA,desB)
    #indI is the number of the points

    if type(kpA[0]) != np.ndarray:
        rev = lambda l: [l[-1]] + ( rev(l[:-1]) if len(l) > 1 else [] )
        # kpA = np.hstack([[p.pt[0] for p in kpA],[p.pt[1] for p in kpA]])
        # kpB = np.hstack([[p.pt[0] for p in kpB],[p.pt[1] for p in kpB]])
        kpA = [rev(list(p.pt)) for p in kpA]
        kpB = [rev(list(p.pt)) for p in kpB]
        kpA = np.array(kpA)
        kpB = np.array(kpB)

    #kpA is the pt of the points


    # If you want to scale it, remember to scale the descriptor position as well
    
    show = True
    if show:
        img3 = drawMatches(grayA, kpA, grayB, kpB, indI, indJ)


    selectedI = kpA[np.array(indI)]
    selectedJ = kpB[np.array(indJ)]

    #It moves to cords centered at the center of x,and of image
    selectedI = -1 * (selectedI - (np.array(imageA.shape[0:2])*0.5))
    selectedJ = -1 * (selectedJ - (np.array(imageA.shape[0:2])*0.5))

    #selectedI is the pt of the point in Camera coordinate system ___[  64.31452942 -166.56546021]

    return selectedI,selectedJ









