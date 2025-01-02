#%%

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
            "OBendEvents":np.zeros((n_trials, n_fish)),
            "OBendLatencies":np.zeros((n_trials, n_fish)),
            "DidASecondOBend":np.zeros((n_trials, n_fish)),
            "DeltaOrientPerOBend":np.zeros((n_trials, n_fish)),
            "DispPerOBend":np.zeros((n_trials, n_fish)),
            "OBendDurations":np.zeros((n_trials, n_fish)),
            "MaxCurvatureOBendEvents":np.zeros((n_trials, n_fish)),
            "DidAMultiBendOBend":np.zeros((n_trials, n_fish)),
            "C1LengthOBendEvents":np.zeros((n_trials, n_fish)),
            "C1AngVelOBendEvents":np.zeros((n_trials, n_fish)),
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
            track_data["OBendEvents"][trial, :] = obend_happened
            obend_start[fish_not_tracked] = np.nan
            track_data["OBendLatencies"][trial, :] = obend_start
            obend_second_counter[fish_not_tracked] = np.nan
            track_data["DidASecondOBend"][trial, :] = obend_second_counter
            obend_dur[fish_not_tracked] = np.nan
            track_data["OBendDurations"][trial, :] = obend_dur
            obend_disp[fish_not_tracked] = np.nan
            track_data["DispPerOBend"][trial,:] = obend_disp
            obend_max_curve[fish_not_tracked] = np.nan
            track_data["MaxCurvatureOBendEvents"][trial,:] = obend_max_curve
            obend_dorient[fish_not_tracked] = np.nan
            track_data["DeltaOrientPerOBend"][trial,:] = obend_dorient
            obend_multibend[fish_not_tracked] = np.nan
            track_data["DidAMultiBendOBend"][trial,:] = obend_multibend
            obend_c1len[fish_not_tracked] = np.nan
            track_data["C1LengthOBendEvents"][trial,:] = obend_c1len
            obend_ang_vel[fish_not_tracked] = np.nan
            track_data["C1AngVelOBendEvents"][trial,:] = obend_ang_vel

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
        



#%%

for k, name in enumerate(names):
    print('group number ' + str(k) + ' is = ' + name)

#%%
cont_id = 2
treat_id = [3]

treat_name = ''
for id in treat_id:
    print(id)
    treat_name = treat_name + names[id]

#%
HabTrackFunctions.plot_burst_data_all(track_data, treat_id, cont_id, col_vec, save_name.replace('burst_trackdata_twoMeasures.pkl', '__')+names[cont_id] + '---vs---' + treat_name, smooth_window=15, plot_taps=True, plot_retest=True, stim_times=stim_times)

#%%

#%%
#HabTrackFunctions.plot_cum_diff(track_data, 3, 2, save_name.replace('.pkl', '__')+names[t]+'_CumulDiff', ylim=0.2)

comps = list(track_data.keys())[:10]

treat_to_plot = np.arange(n_groups)
cont_ID = 1
for comp in comps:
    HabTrackFunctions.plot_cum_diff_oneComp(track_data, comp, treat_to_plot, cont_ID, col_vec, '__cumDiff_' + comp, n_norm = 3, ylim=0.3)

#%%

track_names = glob.glob('*trackdata*.pkl')
cont_id = 0 # the index of the controls
col_vec = np.array(((0,0,0),(1,0,0)))
#%%

        #plt.plot(mu, color = col_str)


for file_name in track_names:

    with open(file_name, "rb") as f:
            track_data = pickle.load(f)
    stim_given = track_data['stim_given']
    rois = track_data['rois']
    names = track_data['names']
    data = track_data['OBendEvents']
    data_train = data[stim_given==1, :]
    #%
    if "TiffTimeInds" in track_data.keys():
        stim_times = []
        if track_data["TiffTimeInds"]:
            for i in range(len(track_data["TiffTimeInds"])):
                stim_times.append((track_data["TiffTimeInds"][i] - track_data["TiffTimeInds"][0]).total_seconds()/60/60)
            stim_times = np.array(stim_times)

    else:
        stim_times = np.array(track_data['TiffFrameInds'])
        stim_times = stim_times - stim_times[0]
        frame_rate = int(np.median(np.diff(track_data['TiffFrameInds']))/60) # assume 60 hz time rate. 
        stim_times = stim_times/(frame_rate*60*60)

    


    non_treat = np.array(non_treat)
    gb = Glasbey()
    p = gb.generate_palette(size=len(non_treat)+2)
    col_vec_all = gb.convert_palette_to_rgb(p)
    col_vec_all = np.array(col_vec_all[1:], dtype=float)/255
    HabTrackFunctions.plot_burst_data_all(track_data, non_treat, 0, col_vec_all, file_name.replace('.pkl', ''), smooth_window=15, plot_taps=True, plot_retest=True)

    ssmd_name = file_name.replace('trackdata', 'ssmddata')
    with open(ssmd_name, "rb") as f:
        ssmd_data = pickle.load(f)

    fingerprint = ssmd_data['fingerprint']
    col_map = seaborn.diverging_palette(300, 120, s=100, l=50, sep=30, as_cmap=True, center="dark")
    plt.figure(figsize=(10,40))

    heatmap_scale = 2
    hmap = seaborn.heatmap(fingerprint, vmax=heatmap_scale, vmin=-heatmap_scale, cmap=col_map, 
        yticklabels=ssmd_data['fingerprint_order'], xticklabels=ssmd_data['names'][1:])
    plt.savefig(file_name.replace('trackdata_twoMeasures.pkl', 'fingerprint') + '.svg')
    plt.savefig(file_name.replace('trackdata_twoMeasures.pkl', 'fingerprint') + '.png')


#%%
# load tracking data: 
plate = 0

# track_file = glob.glob('_plate_'+str(plate)+'*trackdata_*.pkl')[0]

# with open(track_file, 'rb') as f:
#     track_data = pickle.load(f)

online_tracking = h5py.File(os.path.join(exp_dir, 'online_tracking.hdf5'), 'r')

frames_plate = np.array(online_tracking['plate']) == plate
rois = track_data['rois']
names = track_data['names']

#%%
tstamps = np.array(online_tracking['time_stamp'])
frame_inds = np.array(online_tracking['frame_index'])
online_framerate = 1/np.median(np.diff(tstamps))
#%%
plate_data = np.where(frames_plate)[0]
plate_coords = online_tracking['tail_coords'][plate_data,:,:,0]
speed = HabTrackFunctions.get_speeds(plate_coords)

#%%
t_speeds = tstamps[plate_data]
frames_speeds = frame_inds[plate_data]
tiff_frames = track_data['TiffFrameInds']
stim_given = track_data['stim_given'] 

frame_stim1 = np.where(frames_speeds >= tiff_frames[0])[0][0]

t_speeds = t_speeds[1:] - t_speeds[frame_stim1]

plt.figure(figsize=(20,20))
x_epoch = []
savgol_window = int(np.ceil(online_framerate*30) // 2 * 2 + 1)

for i in range(len(rois)):
    y = np.nanmean(speed[:, rois[i]], axis=1)
    plt.plot(t_speeds/(60*60), savgol_filter(y, savgol_window, 2), label=names[i], color=col_vec[i])

plt.xlim((8, 12))
plt.legend()


# #%%
# st = 10000
# end = st + 100
# for i in range(6):
#     plt.plot(speed[st:end, i], label=i)
# plt.ylim((-0.5, 9))
# plt.legend()

# #%% use fish 5
# tail_pts_all = online_tracking['tail_coords'][plate1_data,:,:,:]
# plt.plot(speed[st:end, 5])
# plt.show()

# plt.plot(plate1_coords[st:end, 0, 5], plate1_coords[st:end, 1, 5])


# # #%%
exp_defs = exp_data['exp_defs']
stim_data = {}
stim_data['time'] = exp_defs[1:,0].astype('float')

stim_data['stim'] = exp_defs[1:,1]
stim_data['plate'] = exp_defs[1:,2].astype('int')
stim_data['frame_index'] = (stim_data['time'] * frame_rate).astype('int')
#%%
# df_1 = np.where(frame_inds > stim_data['frame_index'][122])[0][0]
# near_df1 = online_tracking['bend_amps'][df_1:df_1+40, :]
# plt.plot(near_df1[:,:], alpha=0.5)
#%%
omr_start = np.where((stim_data['stim'] == 'omr') & (stim_data['plate']==0))[0][0]
omr_start_fr = stim_data['frame_index'] [omr_start]

omr_end_fr = stim_data['frame_index'][omr_start+1]
omr_start_record = np.where(frame_inds >= omr_start_fr)[0][0]
omr_end_record = np.where(frame_inds <= omr_end_fr)[0][-1]
omr_tstamps = tstamps[omr_start_record:omr_end_record]
omr_tstamps_delta = omr_tstamps - omr_tstamps[0]
omr_frame_indexes = frame_inds[omr_start_record:omr_end_record]
#%
orients = np.array(online_tracking['orientations'])
orient_omr = orients[omr_start_record:omr_end_record, :]
#orient_omr[abs(orient_omr) > np.pi] = np.nan
#%%

#%%

orients = online_tracking['orientations']
coords  = online_tracking['tail_coords']
#%%
fish = 299
frame  = 1000000
orient = orients[frame, fish]
coord = coords[frame, :,fish,:]
print(orient)
import math


def compute_average_angle(x_coords, y_coords):
    # Find indices of valid values (non-NaN)
    valid_idxs = ~np.isnan(x_coords) & ~np.isnan(y_coords)
    
    # Only consider valid coordinates for computing angles
    x_valid = x_coords[valid_idxs]
    y_valid = y_coords[valid_idxs]
    
    angles = []
    for i in range(len(x_valid) - 1):
        dx = x_valid[i+1] - x_valid[i]
        dy = y_valid[i+1] - y_valid[i]
        angles.append(math.atan2(dy, dx))
    
    if len(angles) == 0:
        return np.nan
    
    avg_angle = sum(angles) / len(angles)
    return avg_angle


orient_2 = compute_average_angle(coord[0], coord[1])
print(orient_2)

#%%
plt.figure(figsize=(10,10))
omr_lr = np.rad2deg(np.arcsin(np.sin(np.pi/2 - orient_omr)))
omr_lr[np.isnan(orient_omr)] = np.nan
for i in range(len(rois)):
    y = np.nanmedian(omr_lr[:, rois[i]], axis=1)

    plt.plot(omr_tstamps_delta/60, savgol_filter(y, 11, 2) , label=names[i], color=col_vec[i])

plt.xlim((10,50))
plt.legend()
#%%
from scipy.interpolate import interp1d

f_omr = interp1d(
    omr_frame_indexes, 
    omr_lr,
    axis=0)
#%%
interp_fr = 20
interp_jumps = int(frame_rate/interp_fr)
interp_frames = np.arange(omr_frame_indexes[0], omr_frame_indexes[-1]+1, interp_jumps)
omr_lr = f_omr(interp_frames)
plt.plot(np.nanmean(omr_lr, axis=1))
# omr_lr = medfilt(omr_lr, kernel_size=(int(interp_fr)+1,1))
# plt.plot(np.nanmean(omr_lr, axis=1))

#%%
nfr_per_flip = int(np.round(interp_fr*30))
nfr_per_cyc = nfr_per_flip*2
n_fr_OMR = omr_lr.shape[0]
n_cyc_OMR = int(n_fr_OMR/(nfr_per_cyc))
acc_LR = np.zeros((nfr_per_flip, n_fish)) # array to sum up the left/right angles
n_obs = np.zeros((nfr_per_flip, n_fish)) # keep track of if the fish was actually tracked or not

for i in range(n_cyc_OMR):
    trace_start = i*nfr_per_cyc
    trace_mid = trace_start+nfr_per_flip
    trace_end = trace_mid+nfr_per_flip
    acc_LR = np.nansum(np.stack((acc_LR, omr_lr[trace_start:trace_mid]), axis=2), axis=2)
    acc_LR = np.nansum(np.stack((acc_LR, omr_lr[trace_end:trace_mid:-1]), axis=2), axis=2)
    n_obs = np.nansum(np.stack((n_obs, ~np.isnan(omr_lr[trace_start:trace_mid])), axis=2), axis=2)
    n_obs = np.nansum(np.stack((n_obs, ~np.isnan(omr_lr[trace_end:trace_mid:-1])), axis=2), axis=2)

acc_LR = acc_LR/n_obs

for i in range(len(rois)):
    plt.plot(np.nanmean(acc_LR[:,rois[i]], axis=1),label=names[i], color=col_vec[i])


#%%
# remove fish that arent tracked in at least half of the stimlulus fiips
not_omrtracked = np.mean(n_obs, axis=0) < (n_cyc_OMR)
acc_LR[:, not_omrtracked] = np.nan

# remove first and last observation, since flipping can create artifacts here. 
acc_LR = acc_LR[1:-2]

#%
from scipy.stats import linregress

omr_slopes = np.zeros((n_fish, 1))
omr_slopes[:] = np.nan

for i in range(n_fish):
    y = acc_LR[:,i]
    x = np.arange(acc_LR.shape[0])
    slope, intercept, r_value, p_value, std_err  = linregress(x, y)
    omr_slopes[i] = slope

omr_slopes = omr_slopes*online_framerate # change units to degrees per second
plt.hist(omr_slopes, 50)
#%%
fish_plot = np.arange(8, 300, 50)

plt.plot(acc_LR[:,fish_plot])
#%%
plt.figure(figsize=(40,5))
x = np.arange(omr_lr.shape[0])/(60*online_framerate)
plt.plot(x, np.nanmedian(omr_lr, axis=1))
#%% re-analyze orientations

tail_coords = online_tracking['tail_coords']


#%%
omr_inds = np.arange(omr_start_record,omr_end_record)
orient_omr = np.zeros((len(omr_inds), n_fish))
k = 0
for fr in tqdm(omr_inds):
    orient_omr[k, ], ba = HabTrackFunctions.get_bendamps(tail_coords[fr, :,:,:])
    k+=1

#%%

omr_lr_2 =np.arcsin(np.sin(orient_omr_2+np.pi/2))
omr_lr =np.arcsin(np.sin(orient_omr+np.pi/2))

plt.figure(figsize=(50,5))
plt.plot(np.nanmean(omr_lr, axis=1))
plt.plot(np.nanmean(omr_lr_2, axis=1))
plt.show()

#%%
fish = 55
plt.plot(orient_omr_2[2000:3000, fish])
plt.plot(orient_omr[2000:3000, fish])
#%%
test_frame = 10000

orients, b_a = HabTrackFunctions.get_bendamps(tail_coords[test_frame, :,:,:])

np.array_equal(orients, orientations[test_frame, :,], equal_nan=True)
#%%
orient_lr = medfilt(orient_lr, kernel_size=(base_frame_rate+1,1))
# plt.plot(np.nanmean(orient_lr, axis=1))
# plt.show()
#%
# % now we loop through each phase of the transition, and create
# % a single averaged trace of the orientation that will have a positive
# % slope if the fish is performing OMR (ie reorientating right
# % when the motion is to the right, left when the motion is to
# % the left
nfr_per_flip = base_frame_rate*30
nfr_per_cyc = nfr_per_flip*2
n_fr_OMR = orient_lr.shape[0]
n_cyc_OMR = int(n_fr_OMR/(nfr_per_cyc))
acc_LR = np.zeros((nfr_per_flip, n_fish)) # array to sum up the left/right angles
n_obs = np.zeros((nfr_per_flip, n_fish)) # keep track of if the fish was actually tracked or not

for i in range(n_cyc_OMR):
    trace_start = i*nfr_per_cyc
    trace_mid = trace_start+nfr_per_flip
    trace_end = trace_mid+nfr_per_flip
    acc_LR = np.nansum(np.stack((acc_LR, orient_lr[trace_start:trace_mid]), axis=2), axis=2)
    acc_LR = np.nansum(np.stack((acc_LR, orient_lr[trace_end:trace_mid:-1]), axis=2), axis=2)
    n_obs = np.nansum(np.stack((n_obs, ~np.isnan(orient_lr[trace_start:trace_mid])), axis=2), axis=2)
    n_obs = np.nansum(np.stack((n_obs, ~np.isnan(orient_lr[trace_end:trace_mid:-1])), axis=2), axis=2)

acc_LR = acc_LR/n_obs


#plt.plot(np.nanmean(acc_LR, axis=1))



# remove fish that arent tracked in at least half of the stimlulus fiips
not_omrtracked = np.mean(n_obs, axis=0) < (n_cyc_OMR/2)
acc_LR[:, not_omrtracked] = np.nan

# remove first and last observation, since flipping can create artifacts here. 
acc_LR = acc_LR[1:-2]

#%
from scipy.stats import linregress

omr_slopes = np.zeros((n_fish, 1))
omr_slopes[:] = np.nan

for i in range(n_fish):
    y = acc_LR[:,i]
    x = np.arange(acc_LR.shape[0])
    slope, intercept, r_value, p_value, std_err  = linregress(x, y)
    omr_slopes[i] = slope

omr_slopes = omr_slopes*base_frame_rate # change units to degrees per second


#%%
orient_omr = np.unwrap(orient_omr)
#%
omr_lr =np.sin(orient_omr+np.pi/2)

plt.plot(np.nanmedian(omr_lr, axis=1)[50000:80000])


#%%
per = np.arange(50000,70000)
plt.plot(np.rad2deg(orient_omr[per, 15]))
#%%
plt.plot([orient_omr[:,0]])
#%%
online_track_files = glob.glob('*online_tracking*.pkl')

with open(online_track_files[-1], 'rb') as f:
    online_track_data = pickle.load(f)
print('done')
#%%
tail_coords = np.array(online_track_data['tail_coords'])
orientations = np.array(online_track_data['orientations'])
bend_amps = np.array(online_track_data['bend_amps'])
stats_fish = np.array(online_track_data['stats_fish'])
frame_index = np.array(online_track_data['stats_fish'])
time_stamp = np.array(online_track_data['time_stamp'])
plate = np.array(online_track_data['plate'])

#%%


#%% convert from pickle to hdf5
import h5py
online_track_files = glob.glob('*online_tracking*.pkl')


with h5py.File('online_tracking.hdf5', 'w') as f_h5:
    with open(online_track_files[0], 'rb') as f:
        online_track_data = pickle.load(f)
        tail_coords = np.array(online_track_data['tail_coords'])
        orientations = np.array(online_track_data['orientations'])
        bend_amps = np.array(online_track_data['bend_amps'])
        stats_fish = np.array(online_track_data['stats_fish'])
        frame_index = np.array(online_track_data['stats_fish'])
        time_stamp = np.array(online_track_data['time_stamp'])
        plate = np.array(online_track_data['plate'])

        n_frames, xy, n_fish, n_points = tail_coords.shape
        dset_tail_coords = f_h5.create_dataset('tail_coords', data=tail_coords,chunks=True, maxshape = (None, xy, n_fish, n_points))
        dset_orientations = f_h5.create_dataset('orientations', data=orientations,chunks=True, maxshape = (None, n_fish))
        dset_bend_amps = f_h5.create_dataset('bend_amps', data=bend_amps,chunks=True, maxshape = (None, n_fish))
        dset_stats_fish = f_h5.create_dataset('stats_fish', data=stats_fish,chunks=True, maxshape = (None, n_fish, 5))
        dset_frame_index = f_h5.create_dataset('frame_index', data=frame_index,chunks=True, maxshape = None)
        dset_time_stamp = f_h5.create_dataset('time_stamp', data=time_stamp,chunks=True, maxshape = None)
        dset_plate = f_h5.create_dataset('plate', data=plate, chunks=True, maxshape = None)    
        
        for i in tqdm(range(1, len(online_track_files))):
            with open(online_track_files[i], 'rb') as f:
                online_track_data = pickle.load(f)
                
                tail_coords = np.array(online_track_data['tail_coords'])
                n_frames, xy, n_fish, n_points = tail_coords.shape
                n_frames_already = dset_tail_coords.shape[0]
                dset_tail_coords.resize(n_frames_already+n_frames, axis=0)
                dset_tail_coords[-n_frames:, :, :, :] = tail_coords

                orientations = np.array(online_track_data['orientations'])
                dset_orientations.resize(n_frames_already+n_frames, axis=0)
                dset_orientations[-n_frames:, :] = orientations

                bend_amps = np.array(online_track_data['bend_amps'])
                dset_bend_amps.resize(n_frames_already+n_frames, axis=0)
                dset_bend_amps[-n_frames:, :] = bend_amps

#%%
f_h5 = h5py.File('online_tracking.hdf5', 'r')
tail_coords_load = f_h5['tail_coords']
orients = f_h5['orientations']
bend_amps = f_h5['bend_amps']
plate = f_h5['plate']
print(tail_coords_load.shape)
plt.plot(tail_coords_load[0::100000, 0,:,0], tail_coords_load[0::100000, 1,:,0])
plt.show()
x = np.arange(100000, 100500)
plt.plot((orients[x, 10]))
plt.plot(bend_amps[x, 10])
plt.show()
plt.plot(plate)


#%%
np.savez(online_track_files[-1].replace('.pkl', '.npz'), online_track_data)

#%%
np_load_data = np.load(online_track_files[-1].replace('.pkl', '.npz'), allow_pickle=True)['arr_0']
#%%
tail_coords_online = np.array(online_track_data['tail_coords'])

for i in range(300):
    plt.plot(tail_coords[:, 0, i, 0], tail_coords[:, 1, i, 0])
#%%
HabTrackFunctions.plot_burst_data(track_data["OBendEvents"], 'probability of response', rois, stim_times, names, stim_given, 15, col_vec, save_name = 'test', plot_taps=True)
#%%
plt.plot(fish_not_tracked)
plt.show()
for fish in np.where(fish_not_tracked)[0]:
    plt.title(fish)
    plt.plot(curve_smooth[:,fish])
    plt.plot(speed[:,fish])
    if obend_happened[fish] == 1:
        plt.plot(start_o, curve_smooth[start_o, fish], 'o', label='start')
        plt.plot(obend_peak_ind, obend_peak_val, 'o', label='peak')
        plt.plot(end_o, curve_smooth[end_o, fish], 'o', label='end')
    plt.legend()
    plt.show()
    print("multibend =")
    print(obend_multibend[fish])
    print("second o bend =")
    print(obend_second_counter[fish])
    print("max curve = ")
    print(obend_max_curve[fish])
    print("dorient =")
    print(obend_dorient[fish])
    print("disp =")
    print(obend_disp[fish])
    print("dur = ")
    print(obend_dur[fish])
    print("c1 length =")
    print(obend_c1len[fish])
    print("ang vel = ")
    print(obend_ang_vel[fish])


#%%


#%
plot_burst = False


stimFrame = 1; # the frame that the stimulus was delivered in
nStimInBlocks = 60; # the number of stimuli in a block
ContGroup = 1

RestMin = 62; # the number of minutes of rest, this is used if the timestamps on the tracked files are missing
TestRestMin = 62*4; # length of rest between training and re-test, this is used if the timestamps on the tracked files are missing

TrainLast = 240;
TapIndStart = 270; # the indexes of where the taps happen (0 indexed based on the tracked file naming, not the sequence of indexes, in case some tracked data is missing)
TapIndEnd = 300;

OMRStimStart = 300;   # the indexes of where the OMR happens and we will analyze the resulting changes in heading
OMRStimEnd = 359;

RetestStart = 360; # the indexes where the retest flashes happen. (0 indexed based on the tracked file naming, not the sequence of indexes, in case some tracked data is missing)
RetestEnd = 420;

frame_rate = 560; # speed camera is recording frames, frame rate of the burst recorded movies

OBendThresh = 3; # radians of curvature to call an O-bend
CBendThresh = 1; # radians of curvature to call an C-bend

sav_ord = 3; # parameters for the sgolayfilt of curvature
sav_sz = 15;
SpeedSmooth = 5; # the kernel of the medfilt1 filter for the speed

AreaMin = 100; # min size of an acceptable fish
AreaMax = 600; # max size of an acceptable fish


AngVelThresh = np.pi/2.5; # the max angular velocity per frame we will accept as not an artifact

XYCoorSmoothMulti = 2; # smoothing window for MultiTracker coordinate data moving average
SpeedSmoothMulti = 5; # smoothing window for the speed and orientation traces
SpeedThreshMulti = 1; # the speed threshold for the multitracker data to be considered movement

for exp_dir in tqdm(dirs):
    print(exp_dir)
    try: # lazy!!!
        for plate in range(2):
            #exp_dir = os.path.realpath(exp_dir)
            os.chdir(exp_dir)

            if plate == 0: # % load in the tracked file names for the relevant plate
                Trials = natsort.natsorted(glob.glob('*_plate_0_0*_0000_Track.mat'))
            else:
                Trials = natsort.natsorted(glob.glob('*_plate_1_0*_0000_Track.mat'))



            if len(Trials) == 0:
                print('\a\n\a\n\a\n\a\n\a\n\a\n\a\n\a')
                print('Did not find any trials for plate ' + str(plate) + ' in folder ' + exp_dir)
                continue
            #%

            gc = gspread.oauth()

            sh = gc.open_by_key('1YbSu9YZSB-gUrskkQn57ANPkuZJFkafUa9SHnCkkCz4')
            worksheet = sh.get_worksheet(0)
            df = pd.DataFrame(worksheet.get_all_records())
            #%
            path = os.path.normpath(exp_dir)
            ExpDate = '20'+ path.split(os.sep)[-1][:6]
            

            rows = df.loc[(df['Date Screened'] == int(ExpDate)) & (df['Plate'] == plate)]

            nTreat = rows.shape[0]

            if nTreat == 0:
                print('\a\n\a\n\a\n\a\n\a\n\a\n\a\n\a')
                print('didnt find any entries for ' + ExpDate + ', plate number: ' + str(plate))
                break
            # note that ROIs are 1 indexed in the spreadsheet

            rois = []
            names = []

            for i in range(nTreat):
                rois.append(np.arange(rows['ROI Start'].iloc[i]-1, rows['ROI End'].iloc[i], 1 ))
                if len(rows["Other ROIs"].iloc[i]) > 0:
                    other_rois = HabTrackFunctions.convert_roi_str(rows["Other ROIs"].iloc[i])
                    rois[i] = np.hstack((rois[i], other_rois))

                names.append(str(rows['Product Name'].iloc[i]))
            #%

            names.insert(0, rows['Control Name'].iloc[0])
            rois.insert(0, HabTrackFunctions.convert_roi_str(rows['Vehicle ROIs'].iloc[0]))


            #%

            def load_track_burst_data(track_name):

                BurstData = loadmat(track_name)
                x_coors = np.array(BurstData["HeadX"])
                y_coors = np.array(BurstData["HeadY"])
                orient = np.array(BurstData["Orientation"])
                curve = np.array(BurstData["Curvature"])
                area = np.array(BurstData["Areas"])
                tiff_date = BurstData["TiffDate"]

                return x_coors, y_coors, orient, curve, area, tiff_date

            x_coors, y_coors, orient, curve, area, tiff_date = load_track_burst_data(Trials[0])


            #%

            n_fish = np.shape(x_coors)[1]
            n_trials = len(Trials)
            stim_given = np.zeros((n_trials))

            # set the stim indicies where taps and retests happen. 1 = training flash, 2 = tap, 3 = retest flash
            stim_given[:TrainLast] = 1
            stim_given[TapIndStart:TapIndEnd] = 2
            stim_given[RetestStart:RetestEnd] = 3
            stim_given
            

            # %
            n_blocks = np.floor(n_trials/nStimInBlocks)

            #%
            track_data = {
                "OBendEvents":np.zeros((n_trials, n_fish)),
                "OBendLatencies":np.zeros((n_trials, n_fish)),
                "DidASecondOBend":np.zeros((n_trials, n_fish)),
                "DeltaOrientPerOBend":np.zeros((n_trials, n_fish)),
                "DispPerOBend":np.zeros((n_trials, n_fish)),
                "OBendDurations":np.zeros((n_trials, n_fish)),
                "MaxCurvatureOBendEvents":np.zeros((n_trials, n_fish)),
                "DidAMultiBendOBend":np.zeros((n_trials, n_fish)),
                "C1LengthOBendEvents":np.zeros((n_trials, n_fish)),
                "C1AngVelOBendEvents":np.zeros((n_trials, n_fish)),
                "TiffTimeInds":[],
                "names":names,
                "rois":rois,
                "spreadsheet":rows,
                "stim_given":stim_given }


            #%

            for trial in range(n_trials):

                x_coors, y_coors, orient, curve, area, tiff_date = load_track_burst_data(Trials[trial])

                track_data["TiffTimeInds"].append(datetime.strptime(tiff_date[0], '%d-%b-%Y %H:%M:%S'))


                # %

                MeanArea = np.nanmean(area, axis=0)

                # % fish are considered not to be tracked properly if they are
                #             % not found in the first frame 'isnan(XCoors(1,:)', if the area
                #             % of the blob is too big or two small ' MeanArea < AreaMin |
                #             % MeanArea > AreaMax', or if the curvature trace is too noisy,
                #             % ' nanstd(Curvature, 1) > CurvatureStdThresh'

                fish_not_tracked = (np.isnan(x_coors[0,:])) | (MeanArea < AreaMin) | (MeanArea > AreaMax) | (np.nanstd(curve, axis=0) > CurvatureStdThresh)


                delta_orient_trace = np.vstack((np.full(n_fish, np.nan), np.diff(orient, axis=0)))

                # % remove single frame big jumps - these happen when the head and tail
                #             % get confused by the tracking program, or other noisy reasons

                delta_orient_trace[delta_orient_trace > np.pi] = 2*np.pi-delta_orient_trace[delta_orient_trace > np.pi]
                delta_orient_trace[delta_orient_trace < -np.pi] = delta_orient_trace[delta_orient_trace < -np.pi] + 2*np.pi
                delta_orient_trace[abs(delta_orient_trace) > AngVelThresh] = np.nan




                # remove nan values to avoid errors, assume 0 if not tracking from beginning, otherwise fill with previous value





                # savgol filter
                curve_smooth = savgol_filter(HabTrackFunctions.ffill_cols(curve), sav_sz, sav_ord, axis=0)
                diff_x = np.diff(savgol_filter(HabTrackFunctions.ffill_cols(x_coors), sav_sz, sav_ord, axis=0), axis=0)
                diff_y = np.diff(savgol_filter(HabTrackFunctions.ffill_cols(y_coors), sav_sz, sav_ord, axis=0), axis=0)

                # calculate speed
                speed = np.sqrt(np.square(diff_x) + np.square(diff_x))
                speed = np.vstack((np.zeros(n_fish), speed))

                #%

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
                #%



                for fish in range(n_fish):
                    peakind_curve_pos = find_peaks(curve_smooth[:,fish], width=5)[0]
                    peak_curve_pos = curve[peakind_curve_pos,fish]

                    peakind_curve_neg = find_peaks(-curve_smooth[:,fish], width=5)[0]
                    peak_curve_neg = curve[peakind_curve_neg,fish]

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
                        obend_happened[fish] = 1
                        obend_peak = obend_peaks[0]
                        obend_peak_ind = peakinds_curve[obend_peak]
                        obend_peak_val = curve[obend_peak_ind, fish]

                        # find the start of the movement as the local minima before the peak of the obend
                        start_o = find_peaks(-abs(curve_smooth[:obend_peak_ind, fish]))[0]

                        # if we cant find the start, the fish is moving at the start of the movie, ignore this fish
                        if len(start_o) > 0:
                            start_o = start_o[-1]

                            obend_start[fish] = start_o*1000/frame_rate


                            # get the angular velocity of the C1 movement in radians per msec
                            # not sure this is right, copying from Matlab code...
                            obend_ang_vel[fish] = obend_peak_val/(1000/frame_rate*(obend_peak_ind - start_o))

                            # use when the speed and curvature returns to near 0 as the end of the movement, beginning at l

                            end_o = obend_peak_ind + np.where((speed[obend_peak_ind:,fish]<0.1) & (abs(curve_smooth[obend_peak_ind:,fish]) < 0.1))[0]

                            # if we cant find the end, the movie cut of the end of the movement. can do this downstream analysis
                            if len(end_o) > 0:
                                end_o = end_o[0]

                                obend_dur[fish] = (end_o - start_o)*1000/frame_rate
                                obend_disp[fish] = np.sqrt(np.square(x_coors[start_o, fish] - x_coors[end_o, fish]) + np.square(y_coors[start_o, fish] - y_coors[end_o, fish]))
                                obend_dorient[fish] = HabTrackFunctions.subtract_angles(orient[end_o, fish], orient[start_o, fish])
                                obend_max_curve[fish] = np.max(abs(curve[start_o:end_o, fish]))

                                if obend_disp[fish] < 5: # fish needs to move at least 5 pixels, or else assume its a tracking error
                                    fish_not_tracked[fish] = 1

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

                        else:
                            fish_not_tracked[fish] = 1
                    else:
                        obend_happened[fish] = 0


                    if plot_tracking_results and not fish_not_tracked[fish]:
                        plt.title(fish)
                        plt.plot(curve_smooth[:,fish])
                        plt.plot(speed[:,fish])
                        if obend_happened[fish] == 1:
                            plt.plot(start_o, curve_smooth[start_o, fish], 'o', label='start')
                            plt.plot(obend_peak_ind, obend_peak_val, 'o', label='peak')
                            plt.plot(end_o, curve_smooth[end_o, fish], 'o', label='end')
                        plt.legend()
                        plt.show()
                        print("multibend =")
                        print(obend_multibend[fish])
                        print("second o bend =")
                        print(obend_second_counter[fish])
                        print("max curve = ")
                        print(obend_max_curve[fish])
                        print("dorient =")
                        print(obend_dorient[fish])
                        print("disp =")
                        print(obend_disp[fish])
                        print("dur = ")
                        print(obend_dur[fish])
                        print("c1 length =")
                        print(obend_c1len[fish])
                        print("ang vel = ")
                        print(obend_ang_vel[fish])

                    # nan out non-tracked fish and save arrays

                obend_happened[fish_not_tracked] = np.nan
                track_data["OBendEvents"][trial, :] = obend_happened
                obend_start[fish_not_tracked] = np.nan
                track_data["OBendLatencies"][trial, :] = obend_start
                obend_second_counter[fish_not_tracked] = np.nan
                track_data["DidASecondOBend"][trial, :] = obend_second_counter
                obend_dur[fish_not_tracked] = np.nan
                track_data["OBendDurations"][trial, :] = obend_dur
                obend_disp[fish_not_tracked] = np.nan
                track_data["DispPerOBend"][trial,:] = obend_disp
                obend_max_curve[fish_not_tracked] = np.nan
                track_data["MaxCurvatureOBendEvents"][trial,:] = obend_max_curve
                obend_dorient[fish_not_tracked] = np.nan
                track_data["DeltaOrientPerOBend"][trial,:] = obend_dorient
                obend_multibend[fish_not_tracked] = np.nan
                track_data["DidAMultiBendOBend"][trial,:] = obend_multibend
                obend_c1len[fish_not_tracked] = np.nan
                track_data["C1LengthOBendEvents"][trial,:] = obend_c1len
                obend_ang_vel[fish_not_tracked] = np.nan
                track_data["C1AngVelOBendEvents"][trial,:] = obend_ang_vel


            #gb = Glasbey(base_palette=[(0, 0, 0)])
            gb = Glasbey()
            p = gb.generate_palette(size=nTreat+2)
            col_vec = gb.convert_palette_to_rgb(p)
            col_vec = np.array(col_vec[1:], dtype=float)/255

            #col_vec = np.array(((0,0,0),(0,0,1), (1,0,0), (0,1,0), (0,1,1),(1,1,0)))
            #%
            if names[0] == '':
                names[0] = '0.1% DMSO'

            # use dummy stim times if we dont have all of the stim times recorded -- for example if we didnt manage to track all of the experiments. 
            if len(track_data["TiffTimeInds"]) < 420:
                with open(root_dir + 'dummy_data.pkl', 'rb') as f:
                            dummy_data = pickle.load(f)
                track_data["TiffTimeInds"] = dummy_data["TiffTimeInds"]

            stim_times = []
            for i in range(len(track_data["TiffTimeInds"])):
                stim_times.append((track_data["TiffTimeInds"][i] - track_data["TiffTimeInds"][0]).total_seconds()/60/60)
            stim_times = np.array(stim_times)

            if plot_burst:
                # probability
                HabTrackFunctions.plot_burst_data(track_data["OBendEvents"], 'probability of response', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                #% latency
                HabTrackFunctions.plot_burst_data(track_data["OBendLatencies"], 'latency of response', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                # displacement
                HabTrackFunctions.plot_burst_data(track_data["DispPerOBend"], 'displacement (px)', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                # duration
                HabTrackFunctions.plot_burst_data(track_data["OBendDurations"], 'duration (msec)', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                # curvature
                HabTrackFunctions.plot_burst_data(abs(track_data["MaxCurvatureOBendEvents"]), 'bend amplitude (rad)', rois, stim_times, names, stim_given, 15, col_vec,save_name = save_name)

                # multibend
                HabTrackFunctions.plot_burst_data(track_data["DidAMultiBendOBend"], 'proportion multibend', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                # second obend
                HabTrackFunctions.plot_burst_data(track_data["DidASecondOBend"], 'did second o-bend', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                # c1 length
                HabTrackFunctions.plot_burst_data(track_data["C1LengthOBendEvents"], 'c1 length', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                # ang vel
                HabTrackFunctions.plot_burst_data(abs(track_data["C1AngVelOBendEvents"]), 'ang velocity', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)



            #% now we will analyze the Multitracker .track file

            track_file = glob.glob("*"+str(plate)+".track")
            
            if len(track_file) > 1:
                raise ValueError('check track files from multitracker... more than one found')
            
            track_name = ' '.join(map(str, track_file))
            #%

            def load_multitracker_data(track_name, n_fish):
                print('loading ' + track_name + ' ...will take a few minutes')
                multitrack_data = pd.read_csv(track_name, delimiter='\t')
                frame_inds = multitrack_data.iloc[:, 0].to_numpy()
                t_stmp = multitrack_data.iloc[:, -1].to_numpy()
                ind_vec = np.arange(1, n_fish*3, 3)

                x_coors_multi = multitrack_data.iloc[:, ind_vec].to_numpy()
                y_coors_multi = multitrack_data.iloc[:, ind_vec+1].to_numpy()
                orient_multi = multitrack_data.iloc[:, ind_vec+2].to_numpy()

                print('\a')
                print('loaded : ' + track_name)

                return frame_inds, t_stmp, x_coors_multi, y_coors_multi, orient_multi

            frame_inds, t_stmp, x_coors_multi, y_coors_multi, orient_multi = load_multitracker_data(track_name, n_fish)

            #% load stim index data

            stim_name = track_name.replace("P_0.track", '0.tap')
            stim_name = stim_name.replace("P_1.track", '1.tap')
            print(stim_name)
            #%
            stim_inds = np.loadtxt(stim_name, delimiter='\t')

            # if the experiment go cut off for the stimuli, use the dummy data
            if len(stim_inds) < 420:
                
                stim_inds = np.loadtxt(root_dir + 'dummy_data_' + str(plate) + '.tap', delimiter='\t')
            #%

            n_frames = len(frame_inds)
            st = int(n_frames/2)
            end = st+10000
            # for i in range(n_fish):
            #     plt.plot(x_coors_multi[st:end, i],y_coors_multi[st:end, i] )
            # plt.show()

            #% determine the multitracker online tracking frame rate based on frame indexes

            base_frame_rate = int(frame_rate/ np.median(np.diff(frame_inds[1:1000])))
            

            #% analyze OMR data

            OMRStart = np.where(frame_inds>=stim_inds[OMRStimStart])[0][0]
            OMREnd = np.where(frame_inds>=stim_inds[OMRStimEnd])[0][0]
            # if len(OMRStart) == 0:
            #     warnings.warn('did not find OMR data properly...')
            #%

            #%


            # % Extract the orientation of the fish during the OMR phase, and
            # % we use asind(sind()) to extract only the left/right version
            # % of the fish's orentation and we
            # % use a fairly harsh 1s median filter. This is done to
            # % minimize noise, like full 180 flips. Since the OMR phase
            # % change is at 30s intervals, this should be OK.
            orient_omr = np.radians(orient_multi[OMRStart:OMREnd+1, :])
            from scipy.ndimage import median_filter
            
            orient_lr = np.rad2deg(np.arcsin(np.sin(orient_omr)))
            orient_lr = medfilt(orient_lr, kernel_size=(base_frame_rate+1,1))
            # plt.plot(np.nanmean(orient_lr, axis=1))
            # plt.show()
            #%
            # % now we loop through each phase of the transition, and create
            # % a single averaged trace of the orientation that will have a positive
            # % slope if the fish is performing OMR (ie reorientating right
            # % when the motion is to the right, left when the motion is to
            # % the left
            nfr_per_flip = base_frame_rate*30
            nfr_per_cyc = nfr_per_flip*2
            n_fr_OMR = orient_lr.shape[0]
            n_cyc_OMR = int(n_fr_OMR/(nfr_per_cyc))
            acc_LR = np.zeros((nfr_per_flip, n_fish)) # array to sum up the left/right angles
            n_obs = np.zeros((nfr_per_flip, n_fish)) # keep track of if the fish was actually tracked or not

            for i in range(n_cyc_OMR):
                trace_start = i*nfr_per_cyc
                trace_mid = trace_start+nfr_per_flip
                trace_end = trace_mid+nfr_per_flip
                acc_LR = np.nansum(np.stack((acc_LR, orient_lr[trace_start:trace_mid]), axis=2), axis=2)
                acc_LR = np.nansum(np.stack((acc_LR, orient_lr[trace_end:trace_mid:-1]), axis=2), axis=2)
                n_obs = np.nansum(np.stack((n_obs, ~np.isnan(orient_lr[trace_start:trace_mid])), axis=2), axis=2)
                n_obs = np.nansum(np.stack((n_obs, ~np.isnan(orient_lr[trace_end:trace_mid:-1])), axis=2), axis=2)

            acc_LR = acc_LR/n_obs


            #plt.plot(np.nanmean(acc_LR, axis=1))



            # remove fish that arent tracked in at least half of the stimlulus fiips
            not_omrtracked = np.mean(n_obs, axis=0) < (n_cyc_OMR/2)
            acc_LR[:, not_omrtracked] = np.nan

            # remove first and last observation, since flipping can create artifacts here. 
            acc_LR = acc_LR[1:-2]

            #%
            from scipy.stats import linregress

            omr_slopes = np.zeros((n_fish, 1))
            omr_slopes[:] = np.nan

            for i in range(n_fish):
                y = acc_LR[:,i]
                x = np.arange(acc_LR.shape[0])
                slope, intercept, r_value, p_value, std_err  = linregress(x, y)
                omr_slopes[i] = slope

            omr_slopes = omr_slopes*base_frame_rate # change units to degrees per second

            #% now analyze the swimming data

            #% extract speed trace
            sav_sz_multi = 9
            sav_ord_multi = 2
            # diff_x_multi = np.diff(savgol_filter(HabTrackFunctions.ffill_cols(x_coors_multi), sav_sz_multi, sav_ord_multi, axis=0), axis=0)
            # diff_y_multi = np.diff(savgol_filter(HabTrackFunctions.ffill_cols(y_coors_multi), sav_sz_multi, sav_ord_multi, axis=0), axis=0)
            # speed_multi = np.sqrt(np.square(diff_x_multi) + np.square(diff_y_multi))

            diff_x_multi = np.diff(x_coors_multi, axis=0)
            diff_y_multi = np.diff(y_coors_multi, axis=0)

            diff_x_multi[np.isnan(diff_x_multi)] = 0
            diff_y_multi[np.isnan(diff_y_multi)] = 0
            speed_multi = np.sqrt(np.square(diff_x_multi) + np.square(diff_y_multi))
            #remove any suriously large jumps that are caused by tracking errors and filter
            speed_multi[speed_multi > 15] = 0
            speed_multi = savgol_filter(speed_multi, sav_sz_multi, sav_ord_multi, axis=0)

            #% 
            bouts_multi = (speed_multi > SpeedThreshMulti).astype(int)
            bouts_starts = (np.diff(bouts_multi, axis=0) == 1).astype(int)
            bouts_ends = (np.diff(bouts_multi, axis=0) == -1).astype(int)
            disp_multi = np.copy(speed_multi)
            disp_multi[disp_multi < SpeedThreshMulti] = 0 


            indStart = np.where(frame_inds == stim_inds[0])[0][0]
            free_start = np.where(frame_inds == stim_inds[242])[0][0]
            free_end = np.where(frame_inds == stim_inds[269])[0][0]


            #%
            #%


            # loop through each fish and analyze its bouts and turns:
            #for fish in range(n_fish):
            #%

            # arrays to accumulate across fish
            disp_free = np.full(n_fish, np.nan)
            turn_bias_free = np.full(n_fish, np.nan)
            median_turn_free = np.full(n_fish, np.nan)
            disp_up_omr = np.full(n_fish, np.nan)
            disp_down_omr = np.full(n_fish, np.nan)
            turn_bias_omr = np.full(n_fish, np.nan)
            median_turn_omr = np.full(n_fish, np.nan)
            disp_blks = np.full((n_fish,2), np.nan)


            for fish in range(n_fish):
                bout_starts_fish = np.where(bouts_starts[:, fish] == 1)[0]
                bout_ends_fish = np.where(bouts_ends[:, fish] == 1)[0]

                if len(bout_starts_fish) > 0:
                    # make sure there are no problems at the start or the end, and all bouts have a beginning and and ending
                    if bout_ends_fish[0] < bout_starts_fish[0]:
                        bout_starts_fish = bout_starts_fish[1:]
                    if bout_starts_fish[-1] > bout_ends_fish[-1]:
                        bout_starts_fish = bout_starts_fish[:-1]
                    if bout_starts_fish[0] < 100:
                        bout_starts_fish = bout_starts_fish[1:]
                        bout_ends_fish = bout_ends_fish[1:]
                    if n_frames - bout_ends_fish[-1] < 100:
                        bout_starts_fish = bout_starts_fish[:-1]
                        bout_ends_fish = bout_ends_fish[:-1]

                    # remove bouts that are not at least 3 frames long, as these are likely noise threshold crossings. 

                    bout_len_fish = bout_ends_fish - bout_starts_fish
                    bout_ends_fish = bout_ends_fish[bout_len_fish >= 3]
                    bout_starts_fish = bout_starts_fish[bout_len_fish >= 3]

                    if not len(bout_starts_fish) == len(bout_ends_fish):
                        raise ValueError('bouts did not match up in multitracker data for fish ' + str(fish))

                    # get the difference in orientation by subtracing the orientation three frames before the bout from 3 frames after the bout

                    d_orient_fish = orient_multi[bout_ends_fish, fish] - orient_multi[bout_starts_fish, fish]

                    # assume the smalles angular difference is correct. large jumps are caused by 0-360 degree problems
                    d_orient_fish[d_orient_fish > 180] = 360 - d_orient_fish[d_orient_fish > 180]
                    d_orient_fish[d_orient_fish < -180] = 360 + d_orient_fish[d_orient_fish < -180]

                    # categorize as a swim vs a turn based on a threshold of 10, based on visual inspection of the histograms
                    # plt.hist(d_orient_fish, 70)
                    # plt.vlines([-10,10], 0, 1000)
                    # plt.show()

                    turn_swim = abs(d_orient_fish) > 10 # 1 = turn, 0 = swim
                    turns_fish = np.copy(d_orient_fish)
                    turns_fish[turn_swim == 0] = np.nan

                    # now analyze different epocs of the experiment. we will measure "displacement" as the average speed above threshold, multiplied by the number of frames per minute, give us the average displacement per minute in that period

                    # free swimming period
                    bout_st_inds = np.where((bout_starts_fish > free_start) & (bout_starts_fish < free_end))
                    disp_free[fish] = np.mean(disp_multi[free_start:free_end, fish])*base_frame_rate*60
                    #turn_bias_free[fish] = (np.sum(turn_swim[bout_st_inds]==1) - np.sum(turn_swim[bout_st_inds]==0))/(np.sum(turn_swim[bout_st_inds]==1) + np.sum(turn_swim[bout_st_inds]==0))
                    turn_bias_free[fish] = np.mean(turn_swim[bout_st_inds])
                    median_turn_free[fish] = np.nanmedian(abs(turns_fish[bout_st_inds]))

                    # print('disp free')
                    # print(disp_free[fish])
                    # print('turn bias free')
                    # print(turn_bias_free[fish])
                    # print('median turn free')
                    # print(median_turn_free[fish])

                    # OMR period

                    bout_st_inds = np.where((bout_starts_fish > OMRStart) & (bout_starts_fish < OMREnd))
                    turn_bias_omr[fish] = np.mean(turn_swim[bout_st_inds])
                    median_turn_omr[fish] = np.nanmedian(abs(turns_fish[bout_st_inds]))
                    # print('turn bias omr')
                    # print(turn_bias_omr[fish])
                    # print('median turn omr')
                    # print(median_turn_omr[fish])

                    # % for  the OMR period, we break this into two bins. The fish
                    # % are hyperactive during the first ~4 minutes after the OMR the
                    # % fish are hyperactive, likely due to the effective 1/8 light
                    # % levels from before. Then the rate of the fish dips below
                    # % baseline for the remaining time.
                    omr_up_inds = np.arange(OMRStart, OMRStart+base_frame_rate*60*4)
                    omr_down_inds = np.arange(OMRStart+base_frame_rate*60*4, OMREnd)
                    disp_up_omr[fish] = np.mean(disp_multi[omr_up_inds, fish])*base_frame_rate*60/disp_free[fish]
                    disp_down_omr[fish] = np.mean(disp_multi[omr_down_inds, fish])*base_frame_rate*60/disp_free[fish]
                    # print('disp up omr')
                    # print(disp_up_omr[fish])
                    # print('disp down omr')
                    # print(disp_down_omr[fish])

                    #% now analyze movement rate during the dark flash training blocks, but outside of the stimuli when the lights are on 

                    # dark flash training blocks, stim 0:239
                    inds_out_flash = np.array([], dtype=int)
                    for ind in stim_inds[0:240]:
                        flash_start = np.where(frame_inds == ind)[0][0]
                        out_start = flash_start + base_frame_rate*21
                        inds_out_flash = np.hstack((inds_out_flash, np.arange(out_start, out_start+base_frame_rate*39)))
                    disp_blks[fish, 0] = np.mean(disp_multi[inds_out_flash, fish])*base_frame_rate*60

                    # retest block, stim 360:419
                    inds_out_flash = np.array([], dtype=int)
                    for ind in stim_inds[360:419]:
                        flash_start = np.where(frame_inds == ind)[0][0]
                        out_start = flash_start + base_frame_rate*21
                        inds_out_flash = np.hstack((inds_out_flash, np.arange(out_start, out_start+base_frame_rate*39)))

                    disp_blks[fish, 1] = np.mean(disp_multi[inds_out_flash, fish])*base_frame_rate*60


            #%
            # get the SSMD per group

            def get_ssmds(dataset, fish_rois):
                # return the strinclty standardized mean difference between the treatment groups and the control. first entry in the roi list is the control group
                n_treat = len(fish_rois)-1
                SSMDs = np.zeros((n_treat))
                fish_data = np.copy(dataset)
                fish_data[~np.isfinite(fish_data)] = np.nan
                cont_data = fish_data[rois[0]]
                mean_cont = np.nanmean(cont_data)
                std_cont = np.nanstd(cont_data)
                for k, fish_ids in enumerate(rois[1:]):
                    mean_gr = np.nanmean(fish_data[fish_ids])
                    std_gr = np.nanstd(fish_data[fish_ids])
                    SSMDs[k] = (mean_gr-mean_cont)/(np.sqrt(np.square(std_cont) + np.square(std_gr)))
                
                return SSMDs

            ssmds = {}
            ssmds['disp_blkTrain'] = get_ssmds(disp_blks[:,0], rois)
            ssmds['disp_blkRet'] = get_ssmds(disp_blks[:,1], rois)
            ssmds['disp_free'] = get_ssmds(disp_free, rois)
            ssmds['turn_bias_free'] = get_ssmds(turn_bias_free, rois)
            ssmds['median_turn_free'] = get_ssmds(median_turn_free, rois)
            ssmds['disp_up_omr'] = get_ssmds(disp_up_omr, rois)
            ssmds['disp_down_omr'] = get_ssmds(disp_down_omr, rois)
            ssmds['turn_bias_omr'] = get_ssmds(turn_bias_omr, rois)
            ssmds['median_turn_omr'] = get_ssmds(median_turn_omr, rois)
            ssmds['omr_slopes'] = get_ssmds(omr_slopes, rois)

            # save the fish data for output

            track_data['disp_blks'] = disp_blks
            track_data['turn_bias_free'] = turn_bias_free
            track_data['median_turn_free'] = median_turn_free
            track_data['disp_up_omr'] = disp_up_omr
            track_data['disp_down_omr'] = disp_down_omr
            track_data['turn_bias_omr'] = turn_bias_omr
            track_data['median_turn_omr'] = median_turn_omr
            track_data['omr_slopes'] = omr_slopes
            track_data['acc_lr'] = acc_LR

            #% now do groupwise analysis of burst track data


            def get_ssmds_burst(stim_data, fish_rois):
                # calculate the striclty standardized mean difference, first averaging each fish's response per block, then using the average and std per group for the SSMD caclulation. Group 1 is treated as the control group
                stim_data[~np.isfinite(stim_data)] = np.nan
                fish_rois = rois
                n_treat = len(fish_rois)-1
                SSMDs = np.zeros((n_treat))

                # parse stimuli
                training_flash = np.where(stim_given == 1)[0]
                test_tap = np.where(stim_given == 2)[0]
                retest_flash = np.where(stim_given == 3)[0]

                # for dark flashes we will analyze the first n flashes for "naieve" response, then the mean of the remaining flashes for the habituation response
                n_init = 5
                
                # hard coded into blocks of 60 for training flashes
                block_inds = np.array((training_flash[:n_init],training_flash[n_init:], retest_flash, test_tap))



                # we will have the naieve, habitating, retest and and tap response for each group as the SSMD
                blk_results = np.full((n_treat, 4), np.nan)

                # get the controls
                cont_data = stim_data[:, fish_rois[0]]
                cont_means =  np.full(6, np.nan)
                cont_stds = np.full((6), np.nan)

                # summary stats for controls
                for i in range(4):
                    mean_fish = np.nanmean(cont_data[block_inds[i], :], axis=0)
                    cont_means[i] = np.nanmean(mean_fish)
                    cont_stds[i] = np.nanstd(mean_fish)

                # loop through treatment groups and blocks
                for treat in range(n_treat):
                    treat_data = stim_data[:, fish_rois[treat+1]]
                    for i in range(4):    
                        mean_fish = np.nanmean(treat_data[block_inds[i], :], axis=0)
                        treat_mean = np.nanmean(mean_fish)
                        treat_std = np.nanstd(mean_fish)
                        blk_results[treat, i] = (treat_mean - cont_means[i])/(np.sqrt(np.square(treat_std) + np.square(cont_stds[i])))
                
                return blk_results

            ssmds['OBendEvents'] = get_ssmds_burst(track_data['OBendEvents'], rois)
            ssmds['OBendLatencies'] = get_ssmds_burst(1000-track_data['OBendLatencies'], rois)  # For latencies subtract from max value (1000), because latencies increase during habituation
            ssmds['DidASecondOBend'] = get_ssmds_burst(track_data['DidASecondOBend'], rois)
            ssmds['DeltaOrientPerOBend'] = get_ssmds_burst(abs(track_data['DeltaOrientPerOBend']), rois)
            ssmds['DispPerOBend'] = get_ssmds_burst(track_data['DispPerOBend'], rois)
            ssmds['OBendDurations'] = get_ssmds_burst(track_data['OBendDurations'], rois)
            ssmds['MaxCurvatureOBendEvents'] = get_ssmds_burst(track_data['MaxCurvatureOBendEvents'], rois)
            ssmds['DidAMultiBendOBend'] = get_ssmds_burst(1-track_data['DidAMultiBendOBend'], rois) # subtract from 1 for proportion of simple o-bends
            ssmds['C1LengthOBendEvents'] = get_ssmds_burst(1000-track_data['C1LengthOBendEvents'], rois)  # subtract from max value (1000), because c1 length increases during habituation
            ssmds['C1AngVelOBendEvents'] = get_ssmds_burst(track_data['C1AngVelOBendEvents'], rois)



            #% convert the ssmds into the same format as the original fingerprints from the matlab-based analysis 

            fingerprint = np.vstack((
                # start with dark flash responses, Naieve, train and test blocks
                np.transpose(ssmds['OBendEvents'][:,:3]), 
                np.transpose(ssmds['DidASecondOBend'][:,:3]),
                np.transpose(ssmds['OBendLatencies'][:,:3]),
                np.transpose(ssmds['DispPerOBend'][:,:3]),
                np.transpose(ssmds['OBendDurations'][:,:3]),
                np.transpose(ssmds['MaxCurvatureOBendEvents'][:,:3]),
                np.transpose(ssmds['DeltaOrientPerOBend'][:,:3]),
                np.transpose(ssmds['DidAMultiBendOBend'][:,:3]),
                np.transpose(ssmds['C1LengthOBendEvents'][:,:3]),
                np.transpose(ssmds['C1AngVelOBendEvents'][:,:3]),
                # now tap blocks
                np.transpose(ssmds['OBendEvents'][:,3]), 
                np.transpose(ssmds['DidASecondOBend'][:,3]),
                np.transpose(ssmds['OBendLatencies'][:,3]),
                np.transpose(ssmds['DispPerOBend'][:,3]),
                np.transpose(ssmds['OBendDurations'][:,3]),
                np.transpose(ssmds['MaxCurvatureOBendEvents'][:,3]),
                np.transpose(ssmds['DeltaOrientPerOBend'][:,3]),
                ssmds['disp_blkTrain'],
                ssmds['disp_blkRet'],
                ssmds['disp_free'],
                ssmds['turn_bias_free'],
                ssmds['median_turn_free'],
                ssmds['disp_up_omr'],
                ssmds['disp_down_omr'],
                ssmds['turn_bias_omr'],
                ssmds['median_turn_omr'],
                ssmds['omr_slopes']
                ))

            fingerprint_order = ['Prob-Naieve', 'Prob-Train', 'Prob-Test',  
                        'TwoMvmt-Naieve', 'TwoMvmt-Train', 'TwoMvmt-Test',
                        'Lat-Naieve', 'Lat-Train', 'Lat-Test',
                        'Disp-Naieve', 'Disp-Train', 'Disp-Test',
                        'Dur-Naieve', 'Dur-Train', 'Dur-Test',
                        'Curve-Naieve', 'Curve-Train', 'Curve-Test',
                        'dOrient-Naieve', 'dOrient-Train', 'dOrient-Test',
                        'SimpleOBend-Naieve', 'SimpleOBend-Train', 'SimpleOBend-Test',
                        'C1Length-Naieve', 'C1Length-Train', 'C1Length-Test',
                        'C1AngVel-Naieve', 'C1AngVel-Train', 'C1AngVel-Test',
                        'Prob-Tap', 'TwoMvmt-Tap','Lat-Tap', 'Disp-Tap', 'Dur-Tap', 'Curve-Tap', 'dOrient-Tap', 
                        'SpntDisp-Train', 'SpntDisp-Test', 'SpntDisp-Free', 'TurnBias-Free', 'MedianTurnAng-Free',
                        'OMR-SpeedUp','OMR-SpeedDown','OMR-TurnBias', 'OMR-MedianTurnAng', 'OMR-Perf']


            plt.imshow(fingerprint, vmin=-2, vmax=2)

            # collect data and save

            ssmds['fingerprint'] = fingerprint
            ssmds['fingerprint_order'] = fingerprint_order
            ssmds['names'] = names


            save_name = Trials[0][:Trials[0].find('plate_')+7] + '_ssmddata_twoMeasures.pkl'
            with open(save_name,"wb") as f:
                pickle.dump(ssmds,f)

            #% save the burst data files

            save_name = Trials[0][:Trials[0].find('plate_')+7] + '_trackdata_twoMeasures.pkl'
            with open(save_name,"wb") as f:
                pickle.dump(track_data,f)
    except:
        print('failure in ' + exp_dir)


