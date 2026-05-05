"""
Funtions used to create combined data structure

"""

import numpy as np
import os
#import re

#---------------------------------------------------------------------
def starchild(ws_idmap,xstar,ystar, starID):
    # Inputs:
    #  ws_idmap = watershed ID map
    #  xstar = x pixel position 
    #  ystar = y pixel position
    #  starID = ID number for stars

    n_reg = np.max(ws_idmap)
    ws_id = range(1,n_reg+1)
    star_ids =  [[] for i in range(n_reg)] #empty list of lists

    for n in range(len(xstar)):
        region = ws_idmap[(np.round(ystar[n])).astype(int),(np.round(xstar[n])).astype(int)]
        #print(region,n)
        #find ws_region
        if (region != 0):
            star_ids[region-1].append(starID[n])
        
    return(star_ids)



#---------------------------------------------------------------------
def regchild(ws_idmap_p,ws_idmap_c):
    #Inputs
    #  ws_idmap_p = parent ID map
    #  ws_idmap_c = child ID map
    
    from tqdm import trange

    n_regp = np.max(ws_idmap_p)
    ws_idp = range(1,n_regp+1)
    n_regc = np.max(ws_idmap_c)
    ws_idc = range(1,n_regc+1)
    
    child_ids = [[] for i in range(n_regp)] #empty list of lists
    child_noverlap = [[] for i in range(n_regp)] #empty list of lists
    child_npix = [[] for i in range(n_regp)] #empty list of lists
    nchild = [[] for i in range(n_regp)] #empty list of lists
    
    parent_id = [[] for i in range(n_regc)] 
    parent_noverlap = [[] for i in range(n_regc)] #empty list of lists
    parent_npix = [[] for i in range(n_regc)] #empty list of lists

    #n_reg1
    for n in trange(n_regp):
        parent_region = np.where(ws_idmap_p == ws_idp[n])
        #find children
        child_ids_tmp,child_noverlap_tmp =np.unique(ws_idmap_c[parent_region],return_counts=True)
        #remove unique elements=zero (first element of array)
        #print(child_ids_tmp[0])
        if (child_ids_tmp[0]== 0):
            child_ids_tmp = child_ids_tmp[1:]
            child_noverlap_tmp = child_noverlap_tmp[1:]
        child_ids[n] =child_ids_tmp
        child_noverlap[n] =child_noverlap_tmp
        #print(child_ids[n])
        nchild[n] = len(child_ids[n])
        
        if (nchild[n] > 0):
            for i in range(nchild[n]):
                child_region = np.where(ws_idmap_c == child_ids[n][i])
                child_size = len(child_region[0])
                child_npix[n].append(len(child_region[0]))
        
        #star_ids[region-1].append(n) 
        
    return(child_ids, child_noverlap, child_npix,nchild)
