#wshierarchy/
#Functions to run the watershed hierarchy of stellar positions
#

import numpy as np
import os
from astropy.io import fits
from scipy import ndimage
import pandas as pd

import glob#for sorting
from scipy.ndimage import rotate
from astropy import units as u

#packages of functions created by klarson for PHANGS-hierarchy pipline 
#import wsregions #functions to create DS9 region files


def saveimg(img,header,filename):
    #will OVERWRITE the current version if it exists
    #os.system('rm '+filename)
    hdu=fits.PrimaryHDU(img,header=header)
    hdu.writeto(filename,overwrite=True)
 #----------------------------------
    
def scale_levels(min_scale,nscale,filter_step=0):
    
    """
    Compute the scale levels of the hierarchy

    Parameters:
    ----------
    min_scale : float 
        The FHWM in pixels for the minimum scale length
    nscale : int
        The number of scale levels desired in the hierarchy
    filter_step : int, optional
        Default value is zero.
        The number of hierarchy steps that will be used for high-pass filtering.
        The total number of scales levels needed for computation of the hierarchy
        is equal to nscale+filter_step.
    """
    
    pix_scales = []
    for i in range (nscale+filter_step):
        scale_tmp = min_scale* 2**i
        pix_scales=pix_scales+[scale_tmp]
    return np.array(pix_scales)
 #----------------------------------
   

def gauss_param(nscales, pix_scales):
    """
    Calculate the peak and value at FWHM for a single smoothed gaussian
    at all scale levels.
    These are the default values used as inputs for the Watershed Function.

    Parameters:
    ----------
    nscales: int
        The number of scale levels desired in the hierarchy
    pix_scales: numpy array
        Pixel scale levels of the hierarchy.
        Output from scale_levels function.

    Outputs:
    ----------
    peak_max: numpy array
        Peak value of a smoothed gaussian 
        = 1.5* single gaussian peak

    Edge Threshold: numpy array
        Value at FWHM of a single gaussian.
        
    
    Example:
    ----------
       peak_max, threshold = gauss_params(3,pix_scales)

    """
    #convert FWHM to Sigma
    sigma_scales = pix_scales / 2.358
    
    peak_max = np.zeros(nscales,dtype=float)
    threshold= np.zeros(nscales,dtype=float)

    #create a point souce 
    test_star = np.zeros([50,50])
    test_star[25,25]=1
    test_center = 25
    
    for i in range(nscales):
        test_smooth1 = ndimage.gaussian_filter(test_star, sigma=sigma_scales[i])
        #print('peak_max=1.5*single gaussian peak',1.5*test_smooth1[25,25])
        peak_max1=1.5*test_smooth1[test_center,test_center]
        peak_max[i] = peak_max1
        # -------- value at FWHM -----------
        FWHM_position=(np.round(test_center+pix_scales[i]/2.)).astype(np.int) 
        #print('FWHM position',test_center,test_center+pix_scales[i]/2.)
        #print('value at FWHM',test_smooth1[test_center,FWHM_position])
        threshold[i]=test_smooth1[test_center,FWHM_position]

    return(peak_max, threshold)
#----------------------------------

def create_position_map(xpos,ypos,refimg):

    img_positions_idx = np.zeros_like(refimg)
    img_positions = np.zeros_like(refimg)
    
    ntracer = len(ypos)
    tracer_idx = np.arange(ntracer)+1
    
    # Position of tracer stars with idx values
    img_positions_idx[(np.round(ypos)).astype(np.int), (np.round(xpos)).astype(np.int)] = tracer_idx

    # Position of tracer stars with value=1 -> for image smoothing
    img_positions[(np.round(ypos)).astype(np.int), (np.round(xpos)).astype(np.int)] = 1

    
    return(img_positions_idx)
#----------------------------------


def gauss_kdmap(img_positions,pix_scale,header,outfile):
    #convert FWHM to Sigma
    sigma_scale = pix_scale / 2.358
    smooth = ndimage.gaussian_filter(img_positions, sigma=sigma_scale)
    saveimg(smooth,header,outfile)
#----------------------------------

def hpfilter(nlevels,filter_step,header,file_start):
    #high pass filter of image
    for i in range(nlevels):
        smooth1_file = file_start+str(i)+'.fits'
        smooth2_file = file_start+str(i+int(filter_step))+'.fits'
    
        smooth1 = fits.getdata(smooth1_file)
        smooth2 = fits.getdata(smooth2_file)
        diff_img = smooth1 - smooth2
        #filter_step = 0  means skip subtraction step
        if ( int(filter_step) == 0):
            diff_img = smooth1
        
        outfile=file_start+str(i)+'_hpfilt'+str(filter_step)+'.fits'
        saveimg(diff_img,header,outfile)
        print(outfile,' file written')
#----------------------------------

def find_markers(img,min_dist,peak_max):
    
    from skimage.feature import peak_local_max

    local_max = peak_local_max(img, min_distance=np.round(min_dist).astype(np.int),threshold_abs=peak_max,indices=False)
    markers = ndimage.label(local_max)[0] #
    marker_positions = np.where(markers>0)
    num_markers=len(marker_positions[0])
    #randomize markers
    random_arr = np.arange(num_markers)+1 #start count at 1. 
    np.random.choice(random_arr,len(random_arr), replace=False)
    markers[marker_positions]=random_arr #randomize idx numbers
    marker_positions = [random_arr,marker_positions[0],marker_positions[1]] #random_idx,yarray,xarray
    
    return(markers,marker_positions)

def compute_ws(img,markers,marker_positions,threshold,wcs_header=None,file_start=None):
    """
    if file_start and wcs_header : then save idmask image
    """
    
    import skimage.morphology.watershed as watershed

    mask2 = np.zeros_like(img)
    mask2[img > threshold]=1 #use fwhm of gauss as cut-off threshold
    #
    #---------------------------------------------------------------- 
    #calculate region contours 
    #Calculate watershed with space between regions
    #  -space between regions is necessary to create region contours
    #---------------------------------------------------------------- 
    ws_line = watershed(-img, markers,mask=mask2,watershed_line=True)
    ws= np.copy(ws_line)
    if file_start and wcs_header:
        ws_line_file = file_start+'_idmask.fits'
        saveimg(ws_line,wcs_header,ws_line_file)
    elif (file_start):
        print('Must provide WCS header to save file')
    elif (wcs_header):
        print('Must provide file name to save file')
        
    return(ws_line)


def compute_info(ws,marker_positions,img_positions_idx,refw):
    #def compute(img,refw, min_dist,peak_max, threshold,file_start):
    """
    img : img to watershed regions on
    refw : WCS info for image
    
    threshold : cut-off threshold for watershed algorithm
    file_start : file name for outputs. End of file name will be added.

    Outputs:
    wstable : data table
        region info for image
        size in pixels

    Saves:
    marker region file: file_start_peaks.reg
 
    ws region file: file_start.reg

    idmask: file_start_idmask.fits

    """

    #----------------------------------------------------------------
    #Calculate basic region properties
    # - could add region photometry here -
    #----------------------------------------------------------------

    from astropy.table import Table

    
    #create arrays
    num_markers=len(marker_positions[0])
    reg_size = np.zeros(num_markers)
    reg_area = np.zeros(num_markers)
    ntracer_in_reg = np.zeros(num_markers)
    ws_marker_num = np.zeros(num_markers)
    ws_marker_xpos = np.zeros(num_markers)
    ws_marker_ypos = np.zeros(num_markers)
    ws_marker_ra = np.zeros(num_markers)
    ws_marker_dec = np.zeros(num_markers)
    
    #peak_ct = np.zeros(num_markers)
    #median_ct = np.zeros(num_markers)

    marker_positions_wcs = refw.all_pix2world(marker_positions[2]+0.5,marker_positions[1]+0.5,1)

    #find region properties
    for n in range(1,(num_markers+1)):
        region = np.where(ws == n) #where mask img = marker num
        ws_marker_num[n-1] = n
        marker_idx = np.where(marker_positions[0] == n)#where marker position = marker num
        ws_marker_xpos[n-1] = marker_positions[2][marker_idx]
        ws_marker_ypos[n-1] = marker_positions[1][marker_idx]
        #xpos_tmp = marker_positions[1][n]+0.5
        #ypos_tmp = marker_positions[0][n]+0.5
        ws_marker_ra[n-1] = marker_positions_wcs[0][marker_idx]
        ws_marker_dec[n-1] = marker_positions_wcs[1][marker_idx]    

        reg_size[n-1] = np.sqrt(len(region[0])/(np.pi)) #* pcpix
        reg_area[n-1] = len(region[0])#* pcpix**2
        #peak_ct[n-1] = np.max(subimg[region])
        #median_ct[n-1] = np.median(subimg[region])
        
        tracer_in_reg = np.where(img_positions_idx[region] != 0)
        ntracer_in_reg[n-1] = tracer_in_reg[0].shape[0]

        ##print(n)
        
    #--------------
    #return data
    #--------------
    wstab = Table([ws_marker_num,ws_marker_xpos,ws_marker_ypos,ws_marker_ra,ws_marker_dec,reg_area,reg_size,ntracer_in_reg
              ], 
                names=('regID','peak_xpos','peak_ypos','peak_ra','peak_dec','area_pix','rad_pix','ntracer_reg',
                      ) )
    #            meta={'ws_data':gal+' smoothing scale [pc]:'+str(pc_scales[i].astype(np.int)) })
    return(wstab)

#----------------------------------
def create_circle_mask(xsize, ysize,center,radius=20):

        Y,X = np.ogrid[:ysize,:xsize]
        dist_from_center = np.sqrt( (X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
        #print('circle mask shape:', mask.shape)
        return mask


#----------------------------------
#def mask_stars(refimg, brightstar_file):
#    brightstar = pd.read_csv(brightstar_file,header=0)
#    #starmask = np.empty_like(refimg)
#    for i in range(len(brightstar.xstar)):
#        tmpstar = create_circle_mask(7400,7400,(brightstar.xstar.values[i],brightstar.ystar.values[i]),brightstar.spikerad.values[i])
#        refimg[tmpstar] = 0
#    return refimg
#----------------------------------
#----------------------------------
def create_square_mask(img_xsize, img_ysize,center,rect_size,angle=0):
#def mask_square(img, r0, size, angle=0):
    """Square: 1 inside, 0 outside

    Parameters:
        img_xsize
        img_ysize
        r0 (float, float): center of square
        rect_size (float, float) or (float): full length of slit to mask
        angle (float): angle of rotation in *degrees*

    Example:

    test=square(img,r0=(0 * um, 0 * um), size=(250 * \
                 um, 120 * um), angle=0 * degrees)
    """

    # get image size from input img.
    #imgshape=np.shape(img)
    img_x = np.linspace(0,img_xsize-1,img_xsize,dtype=int)# linear interp. of x
    img_y = np.linspace(0,img_ysize-1,img_ysize,dtype=int)# linear interp. of y
    img_X, img_Y = np.meshgrid(img_x, img_y) #get meshgrid
    #img_Y, img_X = np.meshgrid(img_y, img_x) #get meshgrid
    #print('img_Y shape, ',img_Y.shape)
    if isinstance(rect_size, (float, int)):
        #size of masked rectangle
        #if one number given, x and y same
        sizex, sizey = rect_size, rect_size
    else:
        sizex, sizey = rect_size

    x0, y0 = center

    # Definition of square/rectangle
    xmin = -sizex / 2
    xmax = +sizex / 2
    ymin = -sizey / 2
    ymax = +sizey / 2

    # Rotate the square/rectangle
    Xrot = (img_X - x0) * np.cos(angle) + (img_Y - y0) * np.sin(angle)
    Yrot = -(img_X - x0) * np.sin(angle) + (img_Y - y0) * np.cos(angle)

    # Translate masked points
    mask = (Xrot < xmax) & (Xrot > xmin) & (Yrot < ymax) & (Yrot > ymin)
    #img[index_mask] = 1
    
    return(mask)

#----------------------------------

#----------------------------------
def mask_stars(refimg, brightstar_file, xlen, ylen,maskvalue,angle=0):
    brightstar = pd.read_csv(brightstar_file,header=0)
    #starmask = np.empty_like(refimg)
    #print(refimg.shape)
    for i in range(len(brightstar.xstar)):
        #mask circle
        center = (brightstar.xstar.values[i],brightstar.ystar.values[i])
        #print(center)
        
        tmpstar = create_circle_mask(xlen,ylen,center,brightstar.innerrad.values[i])
        #print(tmpstar.shape)
        
        refimg[tmpstar] = maskvalue#=0 for ws pipeline
        #mask difraction spike
        tmp_spike1 = create_square_mask(xlen,ylen,center, (brightstar.spikerad.values[i]*2,brightstar.spikewidth.values[i]), angle*u.degree)
        #print('spike shape,',tmp_spike1.shape)
        refimg[tmp_spike1] = maskvalue#=0 for ws pipeline
        tmp_spike2 = create_square_mask(xlen,ylen,center, (brightstar.spikerad.values[i]*2,brightstar.spikewidth.values[i]), angle*u.degree+ 90*u.degree)
        refimg[tmp_spike2] = maskvalue#=0 for ws pipeline
        
        #for VERY bright stars need a second horizonal mask ->makes spikewidth=11.5
        if (brightstar.spikewidth.values[i]==11.5):
            tmp_spike3 = create_square_mask(xlen,ylen,center, (brightstar.spikerad.values[i]*2,brightstar.spikewidth.values[i]), angle*u.degree+ 87.3*u.degree)
            refimg[tmp_spike3] = maskvalue#=0 for ws pipeline
        
    
        
    return refimg
#----------------------------------
