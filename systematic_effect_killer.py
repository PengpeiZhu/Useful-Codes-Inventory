#!/usr/bin/env python
# coding: utf-8

#-------------------------------------
#systematic_effect_killer.py
# Pengpei Zhu (UCSC) - Oct 10th, 2021
#                   - Oct 21th, 2021: modified input parameters
# last modified     - Nov 7th, 2021: fixed some saving issues
#-------------------------------------
"""
+
 NAME:
   SYSTEMATIC_EFFECT_KILLER

 PURPOSE:
   Eliminate the linear systematic effects on the light curves of the objects dected by consecutive_apphoto.py, that might be associated, for example, with time, temperature or position on the CCD. Based on methods described in 'Tamuz et al. (2005)' https://ui.adsabs.harvard.edu/abs/2005MNRAS.356.1466T

 CALLING SEQUENCE:
   python systenatic_effect_killer.py

 INPUT PARAMETERS:
    <file_name>  - .pkl file generated by consecutive_apphoto.py that contains detected objects which are found in every single frame. i.e., df_catalog*_18.pkl
    <output_dir> - directory where the output image and catalogs will be created
    (If not given, the current working directory will be set)
        
 OUTPUT PARAMETER:
    a new .pkl file named df_catalog*_18_reduced.pkl that added a row named 'reduced_flux_5pix', whcihc is the 'flux_5pix' substracted from systematic effects caculated by this algorithm.
"""

# import a bunch of useful modules
import numpy as np
import pickle
import pandas as pd
import os, sys


"""Parameters"""
N = 4  # N is the number of linear effects to be subtracted
K = 7  # K is the iteration time, default value is 7


def main(input_file, output_path):
    """This is the mean process"""

    print('Input file is: {}'.format(input_file))
    print('Output directory is: {}'.format(output_path))
    print('Times of iteration per factor: {}'.format(K))
    print('Number of linear factor substracted: {}'.format(N))

    data_frame = open_data(input_file)

    r_ij, err_ij, mean_flux, seg_ids = get_residual(data_frame)

    corrected_flux = correct_flux(r_ij, err_ij, mean_flux, N, K)

    new_df = corrected_df(seg_ids, corrected_flux, data_frame)
    #save to a new pickle file
    new_df.to_pickle(output_path + '/' + f'{input_basename}_reduced.pkl')


"""Functions"""


# a function that opens data file
def open_data(data_file):
    df = pickle.load(open(data_file, 'rb'))
    return df

##################################################################################

# a function to get the residual for caculation
def get_residual(data_frame):

    df_frame1 = data_frame.groupby('frames').get_group(
        1)  #define all the deteced targets in frame 1

    #find all targets for iteration from frame 1
    seg_ids = df_frame1['seg_id'].reset_index(drop=True)
    I = len(seg_ids)
    J = len(data_frame.groupby('frames'))

    # make empty 2d arries for residuals and errors
    r_ij = np.zeros((I, J))
    err_ij = np.zeros((I, J))
    mean_flux = np.zeros((I))
    flux = np.zeros((I, J))
    for i in range(I):
        #seg id of the star, i.e., i
        seg_id = seg_ids[i]
        #data of the ith star in all frames
        df_i = data_frame[data_frame['seg_id'] == seg_id].reset_index(
            drop=True)

        mean_flux_i = sum(df_i['flux_5pix']) / len(
            df_i['flux_5pix'])  #mean_flux of the ith star
        flux_i = df_i['flux_5pix']  #flux of the ith star
        err_i = np.array(df_i['fluxerr_5pix']
                         )  #Uncertainty from measurement for the ith star
        #residual of the ith star
        r_i = np.array(flux_i - mean_flux_i)

        #write the result into 2darray with shape (I,J)
        r_ij[i] = r_i
        err_ij[i] = err_i
        mean_flux[i] = mean_flux_i
    return r_ij, err_ij, mean_flux, seg_ids

##################################################################################

#r_ij and err_ij are 2d arraies with shape (I,J)
#c_i and a_j are 1d arraies with len I and len J

#from an initial a_j, caculate c_i and then iterate to get more a_j and c_i,
#until converges to a set of unchanging c_i and a_j

##################################################################################


# a function to caculate the correct flux
def correct_flux(r_ij, err_ij, mean_flux, N,
                 K):  # K is the iteration time, default value is 7
    # N is the number of linear effects to be subtracted
    #define the dimension of residuals
    I = r_ij.shape[0]
    J = r_ij.shape[1]

    #define shape of the corrected flux
    corr_flux = np.zeros((I, J))

    #define the function used for residual correction for a sigle time
    #returns: r_ij(N) =  r_ij(N-1) - ciaj(N)
    def correct_residual_single(r_ij):

        #set an initial value for airmass a_j
        a_j_initial = np.full(shape=18, fill_value=1, dtype=np.float64)

        #define a function that caculate c_i from a_j
        def get_ci(a_j):
            c_i = []
            for i in range(I):
                c_i.append(
                    sum(r_ij[i][:] * a_j / err_ij[i][:]**2) /
                    sum(a_j**2 / err_ij[i][:]**2))

            c_i = np.array(c_i)

            return c_i

        #define a function that caculate a_j from c_i
        def get_aj(c_i):
            a_j = []
            for j in range(J):
                a_j.append(
                    sum(r_ij[..., j].ravel() * c_i / err_ij[..., j].ravel()**2)
                    / sum(c_i**2 / err_ij[..., j].ravel()**2))

            a_j = np.array(a_j)

            return a_j

        #iterate the two functions of c_i and of a_j, until converges
        for k in range(K):
            a_j_initial = get_aj(get_ci(a_j_initial))
            c_i = get_ci(a_j_initial)
        a_j = a_j_initial

        #get a 2d array of c_i times a_j
        ciaj = np.zeros((I, J))

        for i in range(I):
            ci_times_a = c_i[i] * a_j
            ciaj[i] = ci_times_a

        #apply the correction to residual
        corr_rij_single = r_ij - ciaj

        return corr_rij_single

    # use the corrected residual to perform multiple times of correction
    # N denote to N sets of parameter ciaj substracted from residuals
    for n in range(N):
        r_ij = correct_residual_single(r_ij)

    for i in range(I):
        corr_flux[i] = r_ij[i][:] + mean_flux[i]

    return corr_flux


##################################################################################


# function that write the corrected_flux to corresponding stars and frames
def corrected_df(seg_ids, corrected_flux, data_frame):
    new_df = pd.DataFrame()
    for n in range(len(seg_ids)):
        df_seg_nth = data_frame.groupby('seg_id').get_group(seg_ids[n])
        df_seg_nth['reduced_flux_5pix'] = corrected_flux[n]
        new_df = new_df.append(df_seg_nth)
    return new_df

##################################################################################

if __name__ == '__main__':
    # argument settings
    args = sys.argv
    if len(args) == 3:
        input_file = sys.argv[1]
        input_basename = os.path.basename(input_file).split('.', 1)[0]
        output_path = sys.argv[2]
        if not os.path.isfile(input_file):
            print('No such file exists')
            sys.exit()
        main(input_file, output_path)

    if len(args) == 2:
        input_file = sys.argv[1]
        input_basename = os.path.basename(input_file).split('.', 1)[0]
        # Set the current directory as the output directory.
        cwd = os.getcwd()
        output_path = cwd + '/output'
        os.makedirs(output_path, exist_ok=True)
        if not os.path.isfile(input_file):
            print('No such file exists')
            sys.exit()
        main(input_file, output_path)

    elif len(args) < 2:
        print('need at least one <input_file> argument')
        print('$ systemic_effect_killer.py <input_file> <output_path>')
        sys.exit()
    elif len(args) > 3:
        print('too many arguments ({}) are given'.format(len(args) - 1))
        print('$ systemic_effect_killer.py <input_file> <output_path>')
        sys.exit()
