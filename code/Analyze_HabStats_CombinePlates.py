#%%

import os, sys, glob, pickle, FishTrack
import numpy as np
import matplotlib.pyplot as plt
import seaborn

current_dir = os.path.dirname(__file__)

sys.path.append(current_dir)
sys.path.append(os.path.realpath(current_dir + r'/ExtraFunctions/glasbey-master/'))
from glasbey import Glasbey

#%%

# have the have the same ROI naming and ordering in the spreasheet!! 



track_names = [
    '/media/BigBoy/MultiTracker/20240305_120551/graphs/_plate_0_burst_trackdata_twoMeasures.pkl',
    '/media/BigBoy/MultiTracker/20240305_120551/graphs/_plate_1_burst_trackdata_twoMeasures.pkl'
]


n_loaded = 0
n_files = len(track_names)
for k, track_name in enumerate(track_names):
#track_name = track_names[0]
    with open(track_name, "rb") as f:
        track_data = pickle.load(f)
        
        if k == 0:
            track_data_combined = track_data.copy()

        else:
            comps = list(track_data_combined.keys())[:10]
            for comp in comps:
                track_data_combined[comp] = np.hstack((track_data_combined[comp], track_data[comp]))
        
            n_groups = len(track_data_combined['rois'])
            for gr in range(n_groups):
                track_data_combined['rois'][gr] = np.hstack((track_data_combined['rois'][gr],  track_data['rois'][gr] + n_loaded))
                
                if not track_data_combined['names'][gr] == track_data['names'][gr]:
                    raise ValueError('Group names do not match for group ' + track_data_combined['names'][gr])

        n_rois = track_data['OBendEvents'].shape[1]
        n_loaded+=n_rois
        print(n_loaded)
#%%
out_dir = os.path.join(os.path.split(track_name)[0], 'combined_graphs')

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
os.chdir(out_dir)


non_treat = np.arange(1,n_groups)

gb = Glasbey()
p = gb.generate_palette(size=n_groups+2)
col_vec = gb.convert_palette_to_rgb(p)
col_vec = np.array(col_vec[1:], dtype=float)/255

FishTrack.plot_burst_data_all(track_data_combined, non_treat, 0, col_vec, 'combined_' + str(n_files)+ '_plates', smooth_window=15, plot_taps=True, plot_retest=True, stim_times=track_data_combined['stim_times'])




#%%

treat_to_plot = np.arange(n_groups)
cont_ID = 0

for comp in comps:

    save_st = 'combined_CumMeanDifference_controlGroup=' + names[cont_ID] + 'component_' + comp
    FishTrack.plot_cum_diff_oneComp(track_data_combined, comp, treat_to_plot, cont_ID, col_vec, save_st, n_norm = 3, ylim=0.3)

#%%
names = track_data_combined['names']

for t in non_treat:
    FishTrack.plot_burst_data_all(track_data_combined, t, 0, col_vec, 'combined_' + str(n_files)+ '_plates_'+names[t], smooth_window=15, plot_taps=True, plot_retest=True, stim_times=track_data_combined['stim_times'])


#%% manually plot comparisons between two speicic groups:
k = 0
for name in names:

    print('group ' + str(k) + ' = ' + name)
    k = k+1

control_group = 4
treatment_group = 3

save_st = 'compare_two_groups_'+names[control_group] + '_VS_' + names[treatment_group] + '_'

FishTrack.plot_burst_data_all(track_data_combined, treatment_group, control_group, col_vec, save_st, smooth_window=15, plot_taps=True, plot_retest=True, stim_times=track_data_combined['stim_times'])
# %%


import seaborn as sns

obend_events = track_data_combined['OBendEvents']

stim_given = track_data_combined['stim_given']

rois = track_data_combined['rois']
names = track_data_combined['names']

obend_events_first4 = obend_events[stim_given==1, :]
obend_events_5 = obend_events[stim_given==3, :]

prob_resp_first4 = np.nanmean(obend_events_first4, axis=0)
prob_resp_5 = np.nanmean(obend_events_5, axis=0)

prob_resp_first4_groups = []
prob_resp_5_groups = []

for i, group in enumerate(rois):
    print('group = ' + names[i])
    prob_resp_first4_groups.append(prob_resp_first4[group])
    
    print('first 4 blocks')
    print(prob_resp_first4_groups[i])
    prob_resp_5_groups.append(prob_resp_5[group])

    print('block 5')
    print(prob_resp_5_groups[i])
    print('.......')
    print('.......')
    print('.......')
    print('.......')
    print('done')
#%%
ax = sns.violinplot(prob_resp_first4_groups, inner='stick')
ax.set_xticklabels(names)
plt.xticks(rotation=90)
plt.title('first 4 blocks, mean response')
plt.show()

ax = sns.violinplot(prob_resp_5_groups, inner='stick')
ax.set_xticklabels(names)
plt.xticks(rotation=90)
plt.title('block 5, mean response')
plt.show()


