#!/usr/bin/env python

import numpy as np

"""
Implementation of Dynamic Time Warping
"""

def dtw(st, f=None, D_full=False): 
    """
    Given the function dist, compute the matrice distance of two sequences.
    
    input:
        a and b: lists to match
        dist: function computing the distance between points (takes two floats, return one)
        D_full: True to have the full matrix in output, False to have the last value.
    output:
        Matrix of distances or last value
    """
    a, b = st
    a=np.array(a)
    b=np.array(b)
    N=len(a)
    M=len(b)
    D=[[0. for j in range(M)] for i in range(N)]
    for j in range(1,M):
        cost=f(a[0],b[j])
        D[0][j]=cost+D[0][j-1]
    for i in range(1,N):
        cost=f(a[i],b[0])
        D[i][0]=cost+D[i-1][0]
    for i in range(1,N):
        for j in range(1,M):
            cost=f(a[i],b[j])
            D[i][j]=cost+min(D[i-1][j],D[i][j-1],D[i-1][j-1])
    if not D_full:
        return D[N-1][M-1]
    else:
        return D

def dtw_path(st, D):
    """
    Given distance matrix, compute the best match between objects of the list.
    input:
        a and b: lists to match
        D: distance matrix
    output:
        list of tuples, each of them a couple of index realted to the lists a an b.
    
    """
    a, b = st
    a=np.array(a)
    b=np.array(b)
    N=len(a)
    M=len(b)
    p=[(N-1,M-1)]  # There is another way of doing this, from the beginning instead of the end. Gives similar results (degenerated paths)
    while p[0]!=(0,0):
        n=p[0][0]
        m=p[0][1]
        if n==0:
            p.insert(0,(0,m-1))
        elif m==0:
            p.insert(0,(n-1,0))
        else:
            am=np.argmin([D[n-1][m-1],D[n-1][m],D[n][m-1]])
            if am==0:
                p.insert(0,(n-1,m-1))
            elif am==1:
                p.insert(0,(n-1,m))
            elif am==2:
                p.insert(0,(n,m-1))

    return p
