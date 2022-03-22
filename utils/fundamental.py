
import cv2
import numpy as np

def default(pointsA,pointsB):
    return cv2.findFundamentalMat(pointsA, pointsB,param1=0.002)


def implementacionRansac(pointsA,pointsB):
    
    #The distance considered to be an outlier
    t = 0.002 

    F, inliers = ransacfitfundmatrix(pointsA, pointsB, t)
    print ("embedded  ",len(inliers))
    print ('Total score ',len(pointsA))
    print ('Internal percentage ', len(inliers)*1.0 / len(pointsA))

    return F,inliers

def ransacfitfundmatrix(pA,pB,tolerancia):
    assert(pA.shape == pB.shape)

    #Normalization method of making the origin cnetroide and the average distance from the origin sqrt (2)
    #Also, make sure that the scaling parameter is 1.
    na,Ta = normalizeHomogeneous(pA)
    
    nb,Tb = normalizeHomogeneous(pB)

    #Key points of basic matrix estimation
    s = 8

    #RANSAC scheduling algorithm (using MAS embedded implementation model)
    modeloF = fundamentalFit
    distFun = distanceModel
    isdegenerate = lambda x : False #Nada es degenerado

    #Add 6 elements to hstack in each line x1, X2 and 3 + 3
    dataset = np.hstack([na,nb])
    embedded , M = ransac(dataset,modeloF,distFun,isdegenerate,s,tolerancia)

    F = fundamentalFit(np.hstack([na[embedded ,:],nb[embedded ,:]]))

    F = np.dot(np.dot(Tb, F), np.transpose(Ta))

    return F,embedded 

def fundamentalFit(data):
    assert(data.shape[1] == 6 )

    p1,p2 = data[:,0:3],data[:,3:]
    n, d = p1.shape

    na,Ta = normalizeHomogeneous(p1)
    nb,Tb = normalizeHomogeneous(p2)

    p2x1p1x1 = nb[:,0] * na[:,0]
    p2x1p1x2 = nb[:,0] * na[:,1]
    p2x1 = nb[:, 0]
    p2x2p1x1 = nb[:,1] * na[:,0]
    p2x2p1x2 = nb[:,1] * na[:,1]
    p2x2 = nb[:,1]
    p1x1 = na[:,0]
    p1x2 = na[:,1]
    ones = np.ones((1,p1.shape[0]))

    A = np.vstack([p2x1p1x1,p2x1p1x2,p2x1,p2x2p1x1,p2x2p1x2,p2x2,p1x1,p1x2,ones])
    A = np.transpose(A)

    u, D, v = np.linalg.svd(A)
    vt = v.T

    F = vt[:, 8].reshape(3,3) #Get the vector with the smallest eigenvalue, that is, F.

    #Because the basic matrix is rank 2, singular value decomposition and reconstruction must be carried out again.
    #From range 2
    u, D, v = np.linalg.svd(F)
    F=np.dot(np.dot(u, np.diag([D[0], D[1], 0])), v)

    F=np.dot(np.dot(Tb,F),np.transpose(Ta))

    return F

    pass

def distanceModel(F, x, t):
    p1, p2 = x[:, 0:3], x[:, 3:]

    x2tFx1 = np.zeros((p1.shape[0],1))

    x2ftx1 = [np.dot(np.dot(p2[i], F), np.transpose(p1[i])) for i in range(p1.shape[0])]

    ft1 = np.dot(F,np.transpose(p1))
    ft2 = np.dot(F.T,np.transpose(p2))

    bestInliers = None
    bestF = None

    sumSquared = (np.power(ft1[0, :], 2) +
                  np.power(ft1[1, :], 2)) + \
                 (np.power(ft2[0, :], 2) +
                  np.power(ft2[1, :], 2))
    d34 = np.power(x2ftx1, 2) / sumSquared
    bestInliers = np.where(np.abs(d34) < t)[0]
    bestF = F
    return bestInliers,bestF

def ransac(x, fittingfn, distfn, degenfn, s, t):
    
    maxTrials = 2000
    maxDataTrials = 200
    p=0.99

    bestM = None
    trialCount = 0
    maxembedded_Yet = 0
    N=1
    maxN = 120
    n, d = x.shape

    M = None
    bestembedded  = None
    while N > trialCount:
        degenerate = 1
        degenerateCount = 1
        while degenerate:
            inds = np.random.choice(range(n),s,replace=False)
            sample = x[inds,:]
            degenerate = degenfn(sample)

            if not degenerate:
                M = fittingfn(sample)
                if M is None:
                    degenerate = 1
            degenerateCount +=1
            if degenerateCount > maxDataTrials:
                raise Exception("Error: multiple degraded samples exit")
        #Evaluar modelo
        embedded ,M = distfn(M,x,t)
        nembedded  = len(embedded )

        if maxembedded_Yet < nembedded :
            maxembedded_Yet = nembedded 
            bestM = M
            bestembedded  = embedded 

            #Probability estimation test
            eps = 0.000001
            fractIn = nembedded *1.0/n
            pNoOut = 1 - fractIn*fractIn
            pNoOut = max(eps,pNoOut) #Evitar division por 0
            N = np.log(1-p) / np.log(pNoOut)
            N = max(N,maxN)

        trialCount +=1
        if trialCount > maxTrials:
            print("Maximum number of iterations reached by exiting")
            break
    if M is None:
        raise Exception("Error model not found")
    print ("realization ",trialCount,' items')
    return bestembedded ,bestM

def normalizeHomogeneous(points):
    
    normPoints = []
    if points.shape[1] == 2:
        #Add scale factor (join columns with 1)
        points = np.hstack([points, np.ones((points.shape[0],1))])

    n = points.shape[0]
    d = points.shape[1]
    
    #Leave at level 1
    factores = np.repeat((points[:, -1].reshape(n, 1)), d, axis=1)
    points = points / factores ##Note that this is by element

    prom = np.mean(points[:,:-1],axis=0)
    newP = np.zeros(points.shape)
    #Set the average of all dimensions to 0 (minus scale)
    newP[:,:-1] = points[:,:-1] - np.vstack([prom for i in range(n)])

    #Calculate average distance
    dist = np.sqrt(np.sum(np.power(newP[:,:-1],2),axis=1))
    meanDis = np.mean(dist)
    scale = np.sqrt(2)*1.0/ meanDis

    T = [ 
         [scale, 0    , -scale*prom[0]   ],
         [0    , scale, -scale * prom[1] ],
         [0    ,0     ,  1               ]
        ]
    #This is the original version using t * points
    #This assumes that DXN points are used when using points in NXD format.
    #Use transposition
    T = np.transpose(np.array(T))
    transformedPoints = np.dot(points,T)
    return transformedPoints,T

    pass
