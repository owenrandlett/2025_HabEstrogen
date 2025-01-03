#%% Import libraries and setup fodlders:
import os
import sys
import pickle
import glob
import warnings

import numpy as np
import gspread
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import natsort

from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import median_filter, uniform_filter1d

# Update system paths for local modules if needed
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
sys.path.append(os.path.realpath(os.path.join(current_dir, 'ExtraFunctions', 'glasbey-master')))
from glasbey import Glasbey
import HabTrackFunctions


root_dir = os.path.realpath(r'Z:\2025_EstrogenPaper\BigRigData')

# assume all subdirectories are two levels down

exp_dirs = glob.glob( os.path.realpath( root_dir + '\\*\\*'))

#%% Scan through raw data files and extract responses, etc. 
# parameters:

for exp_dir in tqdm(exp_dirs):

    if "_BR3" in exp_dir:
        big_rig3 = True
        print("BigRig 3!")
    else:
        big_rig3 = False
    graph_dir = os.path.join(exp_dir, 'graphs')

    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)



    n_blocks = 4    
    n_stim_blocks = 60  
    CurvatureStdThresh = 1.7 # max std in curvature trace that is acceptable. non-tracked fish have noisy traces
    SpeedStdThresh = 3.5
    AngVelThresh = np.pi/2.5 # the max angular velocity per frame we will accept as not an artifact

    OBendThresh = 3; # radians of curvature to call an O-bend
    CBendThresh = 1; # radians of curvature to call an C-bend

    sav_ord = 3 # parameters for the sgolayfilt of curvature
    sav_sz = 15
    SpeedSmooth = 5; # the kernel of the medfilt1 filter for the speed


    plot_tracking_results = False # for debugging, will plot each fish that is tracked
    cont_id = 0 # index of the control group, usually 0



    os.chdir(exp_dir)
    with open('exp_data.pkl', 'rb') as f:
        exp_data = pickle.load(f)

    n_fish = np.max(exp_data['im_rois'])

    gc = gspread.oauth()
    if big_rig3:
        sh_str = '1s4Ga1y04dhXehxpoK9JZk7MeFrTOy8_vwkNz021Yw7Y'
    else:
        sh_str = '1YbSu9YZSB-gUrskkQn57ANPkuZJFkafUa9SHnCkkCz4'
    sh = gc.open_by_key(sh_str)
    worksheet = sh.get_worksheet(0)
    df = pd.DataFrame(worksheet.get_all_records())

    path = os.path.normpath(exp_dir)
    ExpDate = path.split(os.sep)[-1][:8]

    for plate in range(2):
        rows = df.loc[(df['Date Screened'] == int(ExpDate)) & (df['Plate (0 or 1)'] == plate)]

        
        #%
        n_groups = rows.shape[0]
        #%
        if n_groups == 0:
            print('\a')
            warnings.warn('didnt find any entries for ' + ExpDate + ', plate number: ' + str(plate))
            continue

        # note that ROIs are 1 indexed in the spreadsheet

        rois = []
        names = []

        for i in range(n_groups):
            roi_str = rows['ROIs'].iloc[i]
            if not roi_str=='[]' and not roi_str=='': # make sure it isnt empty
                names.append(rows['Group Name'].iloc[i])
                rois.append(HabTrackFunctions.convert_roi_str(roi_str))

        #% get burst trials:

        trials = natsort.natsorted(glob.glob(os.path.join(exp_dir, '_plate_' + str(plate) + '*BurstTracks.pkl')))
        n_trials = len(trials)

        if n_trials == 0:
            print('\a\n\a\n\a\n\a\n\a\n\a\n\a\n\a')
            print('didnt find any track files for ' + ExpDate + ', plate number: ' + str(plate))
            print('... skipping')
            continue


        #%
        stim_given = []
        stim_frame = []
        for i in range(n_trials):
            ind = trials[i].find('stim_type')+10
            stim_str = trials[i][ind:ind+2]
            stim_given.append(stim_str)
        stim_given = np.array(stim_given)
        stim_given[stim_given == 'df'] = 1 # training dark flashes == 1
        stim_given[stim_given == 'om'] = 0
        stim_given[stim_given == 'tp'] = 2 # taps = 2

        stim_given[-n_stim_blocks:] = 3 # re-test block = 3
        stim_given = stim_given.astype(int)



        track_data = {
            "ProbabilityOfResponse":np.zeros((n_trials, n_fish)),
            "LatencyOfResponse":np.zeros((n_trials, n_fish)),
            "SecondResponses":np.zeros((n_trials, n_fish)),
            "Reorientation":np.zeros((n_trials, n_fish)),
            "Displacement":np.zeros((n_trials, n_fish)),
            "MovementDuration":np.zeros((n_trials, n_fish)),
            "BendAmplitude":np.zeros((n_trials, n_fish)),
            "CompoundBendResponse":np.zeros((n_trials, n_fish)),
            "C1Length":np.zeros((n_trials, n_fish)),
            "C1AngularVelocity":np.zeros((n_trials, n_fish)),
            "TiffFrameInds":[],
            "names":names,
            "rois":rois,
            "spreadsheet":rows,
            "stim_given":stim_given }

        #%
        for trial in tqdm(range(n_trials)):

            trial_file = trials[trial]
            burst_frame = int(trial_file[trial_file.find('burst_frame_')+12:trial_file.find('_time_')])
            track_data['TiffFrameInds'].append(burst_frame)
            tail_coords, orientations, heading_dir, bend_amps = HabTrackFunctions.load_burst_pkl(trial_file)
            frame_rate = bend_amps.shape[1]

            # % fish are considered not to be tracked properly if they are
            # not found in more than 5% of frames in the movie,
            # or if the curvature or speed trace is too noisy

            
            delta_orient_trace = np.vstack((np.full(n_fish, np.nan), np.diff(orientations.T, axis=0)))

            delta_orient_trace[delta_orient_trace > np.pi] = 2*np.pi-delta_orient_trace[delta_orient_trace > np.pi]
            delta_orient_trace[delta_orient_trace < -np.pi] = delta_orient_trace[delta_orient_trace < -np.pi] + 2*np.pi
            delta_orient_trace[abs(delta_orient_trace) > AngVelThresh] = np.nan
            curve = bend_amps.T
            curve_smooth = savgol_filter(HabTrackFunctions.ffill_cols(curve), sav_sz, sav_ord, axis=0)

            # calculate speed

            x_coors = tail_coords[0,:,0,:].T
            y_coors = tail_coords[1,:,1,:].T

            # diff_x = np.diff(savgol_filter(HabTrackFunctions.ffill_cols(x_coors), sav_sz, sav_ord, axis=0), axis=0)
            # diff_y = np.diff(savgol_filter(HabTrackFunctions.ffill_cols(y_coors), sav_sz, sav_ord, axis=0), axis=0)

            diff_x = np.diff(median_filter(HabTrackFunctions.ffill_cols(x_coors), size=(11,1)), axis=0)
            diff_y = np.diff(median_filter(HabTrackFunctions.ffill_cols(y_coors), size=(11,1)), axis=0)


            speed = savgol_filter(np.sqrt(np.square(diff_x) + np.square(diff_x)), sav_sz, sav_ord, axis=0)
            speed = np.vstack((np.zeros(n_fish), speed))

            obend_start = np.full([n_fish], np.nan)
            obend_happened =  np.full([n_fish], np.nan)
            obend_dorient =  np.full([n_fish], np.nan)
            obend_disp =  np.full([n_fish], np.nan)
            obend_dur =  np.full([n_fish], np.nan)
            obend_max_curve =  np.full([n_fish], np.nan)
            obend_second_counter =  np.full([n_fish], np.nan)
            obend_multibend =  np.full([n_fish], np.nan)
            obend_ang_vel =  np.full([n_fish], np.nan)
            obend_c1len = np.full([n_fish], np.nan)

            #fish_not_tracked = (np.mean(np.isnan(tail_coords[0,:,0,:]), axis=1) > 0.10) | (np.nanstd(bend_amps, axis=1) > CurvatureStdThresh) | (np.nanstd(speed, axis=0) > 3)
            fish_not_tracked = (np.mean(np.isnan(tail_coords[0,:,0,:]), axis=1) > 0.5) #| (np.nanstd(bend_amps, axis=1) > CurvatureStdThresh) | (np.nanstd(speed, axis=0) > SpeedStdThresh)
            
            for fish in range(n_fish):
                peakind_curve_pos = find_peaks(curve_smooth[:,fish], width=5)[0]
                peak_curve_pos = curve_smooth[peakind_curve_pos,fish]

                peakind_curve_neg = find_peaks(-curve_smooth[:,fish], width=5)[0]
                peak_curve_neg = curve_smooth[peakind_curve_neg,fish]

                peakinds_curve = np.hstack((peakind_curve_pos, peakind_curve_neg))
                peaks_curve = abs(np.hstack((peak_curve_pos, peak_curve_neg)))

                I = np.argsort(peakinds_curve)
                peakinds_curve = peakinds_curve[I]
                peaks_curve = peaks_curve[I]

                #plt.plot(curve_smooth[:,fish])
                #plt.plot(peakinds_curve, peaks_curve, 'x')

                # find the first peak the crosses the curvature threshold

                if stim_given[trial]==2:
                    curve_thresh = CBendThresh
                else:
                    curve_thresh = OBendThresh

                obend_peaks = np.where(peaks_curve > curve_thresh)[0]

                # max curvature exibited during movie
                max_curve = np.max(abs(curve[:, fish]))


                # now get the kinematic aspects of the response
                if len(obend_peaks) > 0:
                    start_o = np.nan
                    end_o = np.nan
                    obend_happened[fish] = 1
                    obend_peak = obend_peaks[0]
                    obend_peak_ind = peakinds_curve[obend_peak]
                    obend_peak_val = curve[obend_peak_ind, fish]
                    
                    # determine where the fish is not moving based on speed trace and curvature trace being below a threshold after smoothing    
                    not_moving = np.where((uniform_filter1d(speed[:,fish], 5, mode='nearest')<0.3) & (uniform_filter1d(abs(curve_smooth[:,fish]), 5, mode='nearest') < 0.3))[0]

                    still_before = not_moving[not_moving < obend_peak_ind]

                    # if we cant find the start, stop analysis here
                    if len(still_before) > 0:
                        start_o = still_before[-1]

                        obend_start[fish] = start_o*1000/frame_rate

                        # get the angular velocity of the C1 movement in radians per msec
                        # not sure this is right, copying from Matlab code...
                        obend_ang_vel[fish] = obend_peak_val/(1000/frame_rate*(obend_peak_ind - start_o))

                        # use when the speed and curvature returns to near 0 as the end of the movement to find end of moevement

                        still_after = not_moving[not_moving > obend_peak_ind]
                        

                        # if we cant find the end, the movie cut of the end of the movement. can do this downstream analysis
                        if len(still_after) > 0:
                            end_o = still_after[0]

                            obend_dur[fish] = (end_o - start_o)*1000/frame_rate
                            obend_disp[fish] = np.sqrt(np.square(x_coors[start_o, fish] - x_coors[end_o, fish]) + np.square(y_coors[start_o, fish] - y_coors[end_o, fish]))
                            obend_dorient[fish] = HabTrackFunctions.subtract_angles(orientations.T[end_o, fish], orientations.T[start_o, fish])
                            obend_max_curve[fish] = np.max(abs(curve[start_o:end_o, fish]))

                            # if obend_disp[fish] < 3: # fish needs to move at least 3 pixels, or else assume its a tracking error
                            #     fish_not_tracked[fish] = 1


                            # determine if this is a "multibend o bend" based on if the local minima after the C1 peak is below 0 (normal o-bend) or above 0 (multibend obend)
                            peak_curve = curve[peakinds_curve[obend_peak], fish]
                            if len(peakinds_curve) > (obend_peak + 1):
                                trough_curve = curve[peakinds_curve[obend_peak+1], fish]
                                obend_multibend[fish] = np.sign(peak_curve) == np.sign(trough_curve)
                                # use the difference between peak and trough as c1 length
                                obend_c1len[fish] = (peakinds_curve[obend_peak+1] - peakinds_curve[obend_peak])*1000/frame_rate

                            # now look for a second O-bend
                            if max(peakinds_curve[obend_peaks]) > end_o:
                                obend_second_counter[fish] = 1
                            else:
                                obend_second_counter[fish] = 0


                    # else:
                    #     fish_not_tracked[fish] = 1
                    
                else:
                    obend_happened[fish] = 0
                


            obend_happened[fish_not_tracked] = np.nan
            track_data["ProbabilityOfResponse"][trial, :] = obend_happened
            obend_start[fish_not_tracked] = np.nan
            track_data["LatencyOfResponse"][trial, :] = obend_start
            obend_second_counter[fish_not_tracked] = np.nan
            track_data["SecondResponses"][trial, :] = obend_second_counter
            obend_dur[fish_not_tracked] = np.nan
            track_data["MovementDuration"][trial, :] = obend_dur
            obend_disp[fish_not_tracked] = np.nan
            track_data["Displacement"][trial,:] = obend_disp
            obend_max_curve[fish_not_tracked] = np.nan
            track_data["BendAmplitude"][trial,:] = obend_max_curve
            obend_dorient[fish_not_tracked] = np.nan
            track_data["Reorientation"][trial,:] = obend_dorient
            obend_multibend[fish_not_tracked] = np.nan
            track_data["CompoundBendResponse"][trial,:] = obend_multibend
            obend_c1len[fish_not_tracked] = np.nan
            track_data["C1Length"][trial,:] = obend_c1len
            obend_ang_vel[fish_not_tracked] = np.nan
            track_data["C1AngularVelocity"][trial,:] = obend_ang_vel

        stim_times = np.array(track_data['TiffFrameInds'])
        stim_times = stim_times - stim_times[0]
        stim_times = stim_times/(frame_rate*60*60)
        track_data['stim_times'] = stim_times
        
        os.chdir(graph_dir)

        save_name =trial_file[:trial_file.find('_frame_')] + '_trackdata_twoMeasures.pkl'
        with open(save_name,"wb") as f:
            pickle.dump(track_data,f)


        gb = Glasbey()
        p = gb.generate_palette(size=n_groups+2)
        col_vec = gb.convert_palette_to_rgb(p)
        col_vec = np.array(col_vec[1:], dtype=float)/255
        

        treat_ids = np.arange(len(names))
        non_treat = []
        for t in treat_ids:
                if not t==cont_id and not isinstance(rois[t], int):
                    non_treat.append(t)
                    if len(stim_times) >=330:
                        HabTrackFunctions.plot_cum_diff(track_data, t, cont_id, os.path.split(save_name)[-1].replace('.pkl', '__')+names[t]+'_CumulDiff', ylim=0.2)
                    HabTrackFunctions.plot_burst_data_all(track_data, t, 0, col_vec, os.path.split(save_name)[-1].replace('.pkl', '__')+names[t], smooth_window=15, plot_taps=True, plot_retest=True, stim_times=stim_times)

        HabTrackFunctions.plot_burst_data_all(track_data, non_treat, 0, col_vec, os.path.split(save_name)[-1].replace('.pkl', '_'), smooth_window=15, plot_taps=True, plot_retest=True, stim_times=stim_times)
        
