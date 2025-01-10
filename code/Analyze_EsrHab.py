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
from itertools import combinations

from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import median_filter, uniform_filter1d
import scipy.stats as stats
import scikit_posthocs as sp
# Update system paths for local modules if needed
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
sys.path.append(os.path.realpath(os.path.join(current_dir, 'ExtraFunctions', 'glasbey-master')))
from glasbey import Glasbey
import HabTrackFunctions
import csv
import fnmatch
import importlib


# Set up color palette
gb = Glasbey()

# directory of datasets

root_dir = os.path.realpath(r'Z:\2025_EstrogenPaper\BigRigData')

# assume all subdirectories are two levels down

exp_dirs = glob.glob( os.path.realpath( root_dir + '\\*\\*'))
data_analysis_suffix = '_response_data.pkl'

#% Parameters for analysis
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

camera_rez =  180/2290 #mm/pixel

n_init = 1 # what is used as the 'naive' response to the first n_init stmuli
#%% Scan through raw data files and extract responses, etc. Does not need to be run if data has already been analyzed and saved. 
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
        plates = []

        for i in range(n_groups):
            roi_str = rows['ROIs'].iloc[i]
            if not roi_str=='[]' and not roi_str=='': # make sure it isnt empty
                names.append(rows['Group Name'].iloc[i])
                rois.append(HabTrackFunctions.convert_roi_str(roi_str))
                plates.append(plate)
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
            "Probability_Of_Response":np.zeros((n_trials, n_fish)),
            "Latency_(msec)":np.zeros((n_trials, n_fish)),
            "Proportion_of_Double_Responses":np.zeros((n_trials, n_fish)),
            "Reorientation_(deg)":np.zeros((n_trials, n_fish)),
            "Displacement_(mm)":np.zeros((n_trials, n_fish)),
            "Movement_Duration_(msec)":np.zeros((n_trials, n_fish)),
            "Bend_Amplitude_(deg)":np.zeros((n_trials, n_fish)),
            "Proportion_of_Compound_Responses":np.zeros((n_trials, n_fish)),
            "C1_Bend_Duration_(msec)":np.zeros((n_trials, n_fish)),
            "C1_Angular_Velocity_(deg/msec)":np.zeros((n_trials, n_fish)),
            "TiffFrameInds":[],
            "names":names,
            "plates":plates,
            "rois":rois,
            "spreadsheet":rows,
            "stim_given":stim_given }

        #%
        for trial in tqdm(range(n_trials)):

            trial_file = trials[trial]
            burst_frame = int(trial_file[trial_file.find('burst_frame_')+12:trial_file.find('_time_')])
            track_data['TiffFrameInds'].append(burst_frame)
            tail_coords, orientations, heading_dir, bend_amps = HabTrackFunctions.load_burst_pkl(trial_file)
            frame_rate = bend_amps.shape[1] # number of fps

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

                        # get the angular velocity of the C1 movement from initiation to peak in degrees per msec
                        obend_ang_vel[fish] = np.rad2deg(obend_peak_val/(1000/frame_rate*(obend_peak_ind - start_o)))

                        # use when the speed and curvature returns to near 0 as the end of the movement to find end of moevement

                        still_after = not_moving[not_moving > obend_peak_ind]
                        

                        # if we cant find the end, the movie cut of the end of the movement. can do this downstream analysis
                        if len(still_after) > 0:
                            end_o = still_after[0]

                            obend_dur[fish] = (end_o - start_o)*1000/frame_rate
                            obend_disp[fish] = np.sqrt(np.square(x_coors[start_o, fish] - x_coors[end_o, fish]) + np.square(y_coors[start_o, fish] - y_coors[end_o, fish])) * camera_rez
                            obend_dorient[fish] = np.rad2deg(HabTrackFunctions.subtract_angles(orientations.T[end_o, fish], orientations.T[start_o, fish]))
                            obend_max_curve[fish] = np.rad2deg(np.max(abs(curve[start_o:end_o, fish])))

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
            track_data["Probability_Of_Response"][trial, :] = obend_happened # 0 or 1, probability
            obend_start[fish_not_tracked] = np.nan
            track_data["Latency_(msec)"][trial, :] = obend_start # already in msec
            obend_second_counter[fish_not_tracked] = np.nan
            track_data["Proportion_of_Double_Responses"][trial, :] = obend_second_counter # 0 or 1, probability
            obend_dur[fish_not_tracked] = np.nan
            track_data["Movement_Duration_(msec)"][trial, :] = obend_dur # units are in msec
            obend_disp[fish_not_tracked] = np.nan
            track_data["Displacement_(mm)"][trial,:] = obend_disp # in units of mm
            obend_max_curve[fish_not_tracked] = np.nan
            track_data["Bend_Amplitude_(deg)"][trial,:] = obend_max_curve # in units of degrees
            obend_dorient[fish_not_tracked] = np.nan
            track_data["Reorientation_(deg)"][trial,:] = obend_dorient # in units of degrees
            obend_multibend[fish_not_tracked] = np.nan
            track_data["Proportion_of_Compound_Responses"][trial,:] = obend_multibend # 0 or 1, probability
            obend_c1len[fish_not_tracked] = np.nan
            track_data["C1_Bend_Duration_(msec)"][trial,:] = obend_c1len # units are in msec
            obend_ang_vel[fish_not_tracked] = np.nan
            track_data["C1_Angular_Velocity_(deg/msec)"][trial,:] = obend_ang_vel # units are degrees per msec

        stim_times = np.array(track_data['TiffFrameInds'])
        stim_times = stim_times - stim_times[0]
        stim_times = stim_times/(frame_rate*60*60)
        track_data['stim_times'] = stim_times
        
        os.chdir(graph_dir)

    
        save_name =trial_file[:trial_file.find('_burst_frame_')] + data_analysis_suffix
        with open(save_name,"wb") as f:
            pickle.dump(track_data,f)

        print('Saved data to: ' + save_name)

#% Combine data from all experiments into one file

analyzed_pkls = glob.glob( os.path.realpath( root_dir + '\\*\\*\\*response_data.pkl'))

track_data_combined = {}
n_loaded = 0
n_files = len(analyzed_pkls)
for k, track_name in enumerate(analyzed_pkls):
    exp_date = track_name.split(os.sep)[-2].split('_')[0]
    with open(track_name, "rb") as f:
        track_data = pickle.load(f)
        n_groups = len(track_data['rois'])
        
        if k == 0:
            track_data_combined = track_data.copy()
            track_data_combined['exp_date'] = []
            for gr in range(n_groups):
                track_data_combined['exp_date'].append(exp_date)



        else:
            comps = list(track_data_combined.keys())[:10]
            for comp in comps:
                comp_data = track_data[comp]
                n_stim, n_fish = comp_data.shape
                if n_stim == track_data_combined[comp].shape[0]:
                    track_data_combined[comp] = np.hstack((track_data_combined[comp], comp_data))
                else: # in some experiments we dont have the full stimulus run, will fill these up with NAN
                    comp_data_expand = np.full((track_data_combined[comp].shape[0], n_fish), np.nan)
                    comp_data_expand[:n_stim, :] = comp_data
                    track_data_combined[comp] = np.hstack((track_data_combined[comp], comp_data_expand))
                
            
            for gr in range(n_groups):
                track_data_combined['rois'].append(track_data['rois'][gr] + n_loaded)
                track_data_combined['names'].append(track_data['names'][gr])
                track_data_combined['plates'].append(track_data['plates'][gr])
                track_data_combined['exp_date'].append(exp_date)
              
        n_rois = track_data['Probability_Of_Response'].shape[1]
        n_loaded+=n_rois
        print(n_loaded)

output_file_IDs = os.path.join(root_dir, 'combined_data_IDs.csv')
with open(output_file_IDs, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index', 'Name_ExperimentDate_Plate', 'ROI'])
    for index, (roi, exp_date, plate, name) in enumerate(zip(track_data_combined['rois'],track_data_combined['exp_date'],track_data_combined['plates'], track_data_combined['names'])):
        writer.writerow([index,str(name) + '_' + str(exp_date) + '_plate' + str(plate), roi])

track_file_combined = os.path.join(root_dir, 'track_data_combined.pkl')
with open(track_file_combined,"wb") as f:
    pickle.dump(track_data_combined,f)

#%% laod the pre-analyzed datasets to analyze individual experiments

track_file_combined = os.path.join(root_dir, 'track_data_combined.pkl')
with open(track_file_combined,"rb") as f:
    track_data_combined = pickle.load(f)

    # Load combined_data_IDs.csv
combined_data_ids_path = os.path.join(root_dir, 'combined_data_IDs.csv')
combined_data_ids = pd.read_csv(combined_data_ids_path)
combined_names = combined_data_ids['Name_ExperimentDate_Plate'].values


stim_epochs = [
    np.arange(0,n_init),
    np.arange(n_init, n_stim_blocks),
    np.arange(n_stim_blocks, n_stim_blocks*2),
    np.arange(n_stim_blocks*2, n_stim_blocks*3),
    np.arange(n_stim_blocks*3, n_stim_blocks*4),
    np.arange(n_init, n_stim_blocks*4),
    np.arange(n_stim_blocks*4),
    np.where(track_data_combined['stim_given'] == 2)[0],

]

epoch_names = [
    'Naive Response (first ' + str(n_init) + ' stim.)', 
    'Block 1',
    'Block 2',
    'Block 3',
    'Block 4',
    'Trained Response',
    'All DF Responses',
    'Acoustic Responses',
]


#%% effect of estradiol and on habituation
importlib.reload(HabTrackFunctions)
exp_string = 'Estradiol'

group_categories = ['DMSO', 'Estradiol']  

group_names = np.array([
    ['DMSO_20220228_plate0', '10 µM beta estradiol_20220228_plate0'],
    ['DMSO_20220307_plate0','10 µM estradiol_20220307_plate0'],
    ['DMSO _20220307_plate1','10 µM estradiol_20220307_plate1'],
    ['DMSO_20220308_plate0','DMSO + 10 µM estradiol_20220308_plate0'],
    ['DMSO_20220308_plate1','10 µM estradiol_20220308_plate1'],
    ['DMSO_20220316_plate1','10 µM estradiol_20220316_plate1'],
    ['0.1% DMSO_20220405_plate0','10 µM estradiol_20220405_plate1'],
    ['0.1% DMSO_20220412_plate1','10 µM estradiol_20220412_plate1'],
    ['0.1% DMSO_20220413_plate0','10 µM estradiol_20220413_plate0'],
    ['0.1% DMSO_20220414_plate1','0.1% DMSO + 10 µM estradiol_20220414_plate1'],
    ['0.1% DMSO_20220504_plate0','10 µM estradiol_20220504_plate0'],
    ])

def get_group_indexes(search_names, names_list, print_finds=True):
    indexes = []
    matched_names = []
    for i, name in enumerate(search_names):
        for j, pattern in enumerate(names_list):
            if name in pattern:
                if pattern not in matched_names:
                    indexes.append(j)
                    matched_names.append(pattern)
                    if print_finds:
                        print(f"Found index: {j}, Name: {pattern}")
                else:
                    if print_finds:
                        print(f"Duplicate found and ignored: {pattern}")

    if not indexes:
        print("No matches found for the given search patterns.")

    return indexes, matched_names

def get_rois (group_names, print_finds=True):
    group_indexes = []
    found_names = []
    plot_rois = []
    for gr in range(group_names.shape[1]):
        if print_finds:
            print('Group = ' + group_categories[gr])
        index, names = get_group_indexes(group_names[:,gr], combined_names, print_finds)
        group_indexes.append(index)
        found_names.append(names)
        plot_rois.append(HabTrackFunctions.get_all_matching_ROIS(index, track_data_combined))
    return plot_rois

def plot_bursts_and_epochs(exp_string, group_categories,  group_names, plot_cumdiff = True):
    n_groups = group_names.shape[1]
    graph_folder = HabTrackFunctions.make_graph_folder(exp_string, root_dir)
    os.chdir(graph_folder)

    plot_rois = get_rois(group_names)          
    HabTrackFunctions.plot_burst_responses(track_data_combined, group_categories, plot_rois, gb, exp_string, smooth_window=15, plot_taps=True, plot_retest=False, stim_times=track_data_combined['stim_times'])

    p = gb.generate_palette(n_groups + 1)
    col_vec = gb.convert_palette_to_rgb(p)
    col_vec = list(np.array(col_vec[1:], dtype=float)/255)
    HabTrackFunctions.plot_means_epoch(track_data_combined, group_categories, plot_rois, stim_epochs, epoch_names, exp_string, gb, col_vec=col_vec, components_to_plot=np.arange(10))
    if plot_cumdiff:
        if n_groups > 2:
            for comp in list(combinations(np.arange(n_groups),2)):
                plot_rois_comp = get_rois(group_names[:, comp], print_finds=False) 
                group_categories_comp = []
                for gr in comp:
                    group_categories_comp.append(group_categories[gr])
                HabTrackFunctions.plot_cum_diff(track_data_combined, group_categories_comp, plot_rois_comp, exp_string, components_to_plot=[0,2,1,7,5,4,3,6], n_boots=2000, n_norm=n_init)
        else:
            HabTrackFunctions.plot_cum_diff(track_data_combined, group_categories, plot_rois, exp_string, components_to_plot=[0,2,1,7,5,4,3,6], n_boots=2000, n_norm=n_init)
plot_bursts_and_epochs(exp_string, group_categories,  group_names, plot_cumdiff=True)

#%% Esr1

exp_string = 'Esr1 Mutants'
group_categories = [r'DMSO, $\it{esr1^{+/?}}$', 'Estradiol, $\it{esr1^{+/?}}$', 'DMSO, $\it{esr1^{-/-}}$', 'Estradiol, $\it{esr1^{-/-}}$']  
group_names = np.array([
    ['DMSO esr1 +/?_20220706_plate0', 'Estradiol esr1 +/?_20220706_plate0', 'DMSO esr1 -/-_20220706_plate0', 'Estradiol esr1 -/-_20220706_plate0'],
    ['DMSO esr1 +/+;+/-_20220719_plate1', '10 µM estradiol esr1 +/+,+/-_20220719_plate1', 'DMSO esr1 -/-_20220719_plate1', '10 µM estradiol esr1 -/-_20220719_plate1']
])

plot_bursts_and_epochs(exp_string, group_categories, group_names, plot_cumdiff=True)
#%% Esr2a

exp_string = 'Esr2a Mutants'
# group_categories = ['DMSO, esr1+/?', 'Estradiol, esr1+/?', 'DMSO, esr1-/-', 'Estradiol, esr1 -/-']  
# group_names = np.array([
#     ['DMSO esr1 +/?_20220706_plate0', 'Estradiol esr1 +/?_20220706_plate0', 'DMSO esr1 -/-_20220706_plate0', 'Estradiol esr1 -/-_20220706_plate0'],
#     ['DMSO esr1 +/+;+/-_20220719_plate1', '10 µM estradiol esr1 +/+,+/-_20220719_plate1', 'DMSO esr1 -/-_20220719_plate1', '10 µM estradiol esr1 -/-_20220719_plate1']
# ])

# plot_bursts_and_epochs(exp_string, group_categories, group_names)
