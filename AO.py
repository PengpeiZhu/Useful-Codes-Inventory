#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling import models, fitting
from matplotlib.colors import LogNorm
from astropy.io.fits import getdata


# In[ ]:


#open the file and return star data
#center x and center y are coord of the center pixel of the star
#n is the number of pixels included for the star image

def star_data(data, center_x,center_y,n=11):
    star  = data[center_y-n:center_y+n+1,center_x-n:center_x+n+1]
    return star


# In[ ]:


#define gain
gain = fits.getheader('fixed_ao/no_dist/with_ao.fit')['EGAIN']


# In[ ]:


#define all the data needed
nodis_ao = star_data(getdata('fixed_ao/no_dist/with_ao.fit'),185,315)

cyl_ao = star_data(getdata('fixed_ao/cyl(1000mm)/with_ao.fit'),266,322)

cyl_noao = star_data(getdata('fixed_ao/cyl(1000mm)/no_ao.fit'),224,305)

bicon_ao = star_data(getdata('fixed_ao/bi_convex(1000mm)/with_ao.fit'),247,305)

bicon_noao = star_data(getdata('fixed_ao/bi_convex(1000mm)/no_ao.fit'),202,287)

background_1 = getdata('fixed_ao/no_dist/with_ao.fit')[0:30]


# In[ ]:


#a function that gives the fwhm and error of the data
#fitted with a gaussian

#y is a list of values of the column or row that goes through the star's center
#name is the name of the figure
def fit_g(y, name, savefig=True):

    x = np.linspace(0, 23, 23)
    #rms is the background value
    rms = np.sqrt(np.sum(np.square(background_1)) / background_1.size)
    #err is the error on each pixel, caculated from rms and pixel value
    err = np.sqrt(rms**2 + (y - rms) / gain)
    #plot the error bar
    plt.errorbar(x, y - rms, yerr=err, fmt='o', c='r', markersize=2)
    #perform the gaussian fitting
    g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, x, y)

    #caculate the fwhm
    fwhm = 2 * g.stddev * np.sqrt(2 * np.log(2))

    plt.plot(x, g(x))

    plt.title(f'FWHM:{fwhm}, MEAN_ERROR:{np.mean(err)}')
    plt.xlabel('pixel number')
    plt.ylabel('flux')

    if savefig:
        plt.savefig(f'{name}.png')

    return fwhm, err, rms


#a function that plot the image of the star
def imshow(data, savename):
    plt.imshow(data, norm=LogNorm())
    plt.colorbar()

    plt.savefig(f'{savename}.png')


# In[ ]:


imshow(nodis_ao,'nodis_ao')


# In[ ]:


fit_g((nodis_ao[12]), 'no_dis_ao', savefig=True)


# In[ ]:


imshow(cyl_ao,'cyl_ao_im')


# In[ ]:


fit_g(cyl_ao[:,12], 'cyl_ao')


# In[ ]:


imshow(cyl_noao,'cyl_noao_im')


# In[ ]:


fit_g(cyl_noao[12], 'cyl_noao')


# In[ ]:


imshow(bicon_ao,'bicon_ao_im')


# In[ ]:


fit_g(bicon_ao[12],'bicon_ao')


# In[ ]:


imshow(bicon_noao,'bicon_noao_im')


# In[ ]:


fit_g(bicon_noao[:,10],'bicon_noao')

