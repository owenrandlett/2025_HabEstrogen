import pickle

import numpy as np
import matplotlib.pyplot as plt



def ffill_cols(a, startfillval=0):
    mask = np.isnan(a)
    tmp = a[0].copy()
    a[0][mask[0]] = startfillval
    mask[0] = False
    idx = np.where(~mask,np.arange(mask.shape[0])[:,None],0)
    out = np.take_along_axis(a,np.maximum.accumulate(idx,axis=0),axis=0)
    a[0] = tmp
    return out

def remove_brackets_invalid(string, invalid = '<>:"/\|?*() '):
    while '(' in string and ')' in string:
        start = string.index('(')
        end = string.index(')', start) + 1
        string = string[:start] + string[end:]

    for char in invalid:
        string = string.replace(char, '')
    return string

def load_burst_pkl(burst_file):
    with open(burst_file, 'rb') as f:
        burst_data = pickle.load(f)
    
    tail_coords = burst_data['tail_coords']
    orientations = burst_data['orientations']

    heading_dir = burst_data['heading_dir']
    bend_amps = burst_data['bend_amps']

    return tail_coords, orientations, heading_dir, bend_amps



def subtract_angles(lhs, rhs):
    import math
    """Return the signed difference between angles lhs and rhs

    Return ``(lhs - rhs)``, the value will be within ``[-math.pi, math.pi)``.
    Both ``lhs`` and ``rhs`` may either be zero-based (within
    ``[0, 2*math.pi]``), or ``-pi``-based (within ``[-math.pi, math.pi]``).
    """

    return math.fmod((lhs - rhs) + math.pi * 3, 2 * math.pi) - math.pi


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def convert_roi_str(roi_str):
    ### convert from a string that has a matlab stype indexing, using a mix of commans and colons, into a vector of indexes
    ### note that ROIs in the spreadsheet should be 1 indexed to match with old maltab formatting
    roi_str = roi_str.strip('[').strip(']').split(',')
    #print(roi_str)
    for k, roi_part in enumerate(roi_str):
        roi_part = roi_part.strip(' ')
        if roi_part.find(':') == -1: # if there is no colon, just a single number
            vec = int(roi_part)-1
        else:
            roi_subparts = roi_part.split(':')
            if len(roi_subparts) == 2:      # only 1 semicolon, count by ones                
                vec = np.arange(int(roi_subparts[0])-1, int(roi_subparts[1])) # subtract 1 from first index to make 0 indexed
            elif len(roi_subparts) == 3:    # 2 semicolons in matlab index style, jump by middle number
                vec = np.arange(int(roi_subparts[0])-1, int(roi_subparts[2]), int(roi_subparts[1]))
            else:
                raise ValueError('problem with ROI parsing for roi string' + str(roi_str))
        if k ==0:
            roi_vec = vec
        else:
            roi_vec = np.hstack((roi_vec, vec))
        
    return roi_vec



def plot_burst_responses(track_data, fish_names, fish_ids, gb, save_str, nStimInBlocks = 60, smooth_window=15, components_to_plot=range(10), plot_taps = True, plot_retest=True, stim_times = None, first_block = False):
    import warnings
    from scipy.signal import savgol_filter
    
    # Hardcoded font sizes
    legend_fontsize = 18
    axes_fontsize = 18
    ticks_fontsize = 14

    # Generate color palette
    n_gr = len(fish_ids)
    p = gb.generate_palette(n_gr + 1)
    col_vec = gb.convert_palette_to_rgb(p)
    col_vec = np.array(col_vec[1:], dtype=float)/255
    
    # If no stimulus times are given, calculate them from the TiffTimeInds
    if np.sum(stim_times == None) > 0:
        stim_times = []
        for i in range(len(track_data["TiffTimeInds"])):
            stim_times.append((track_data["TiffTimeInds"][i] - track_data["TiffTimeInds"][0]).total_seconds()/60/60)
        stim_times = np.array(stim_times)

    time_inds = stim_times
    stim_given = track_data['stim_given']

    dtype_to_plot = np.array(list(track_data.keys()))[components_to_plot]
    y_text = dtype_to_plot

    for d, dtype in enumerate(dtype_to_plot):

        data = abs(track_data[dtype])
        plt.figure(figsize=(10,7))
        plt.xlabel('time (hr)', fontsize=axes_fontsize)
        plt.ylabel(y_text[d].replace('_', ' '), fontsize=axes_fontsize)

        for i in range(n_gr): # plot the raw dark flash stimuli
            inds_stim = np.ix_((stim_given==1) | (stim_given==3))[0]
            inds_fish  = fish_ids[i]
            inds_both = np.ix_(inds_stim, inds_fish)

            plt.plot(time_inds[inds_stim], np.nanmean(data[inds_both], axis=1), '.', markersize=3, color= col_vec[i], label=fish_names[i]+' , n='+str(len(inds_fish)) )
           
        lgnd = plt.legend(fontsize=legend_fontsize, markerscale=3, loc="lower right")  


        for i in range(n_gr): # plot the smoothed data off of the frist 4 blocks, and retest block
            inds_fish  = fish_ids[i]
            #inds_stim = 
            for k in range(5):
                inds_block = np.ix_((stim_given==1) | (stim_given==3))[0][k*nStimInBlocks:k*nStimInBlocks+nStimInBlocks]
                inds_both_block =  np.ix_(inds_block, inds_fish)

                y = np.nanmean(data[inds_both_block], axis=1) 
                x = time_inds[inds_block]
                # remove NaNs
                x = x[~np.isnan(y)]
                y = y[~np.isnan(y)]
                try:
                    y = savgol_filter(y, smooth_window, 2)
                    plt.plot(x,y, '-', color= col_vec[i], linewidth=5, alpha=0.8)
                except:
                    warnings.warn('savgol did not converge')


        # plot taps
        if plot_taps:
            for i in range(n_gr):
                inds_fish  = fish_ids[i]
                inds_block = np.where(stim_given == 2)[0]
                n_blocks_taps = np.ceil(len(inds_block)/nStimInBlocks).astype(int)

                for tp_blk in range(n_blocks_taps):
                    inds_tp_block = np.zeros(stim_given.shape).astype(bool)
                    inds_tp_block[:] = False
                    
                    st_tap = tp_blk * nStimInBlocks
                    end_tap = min(nStimInBlocks+ tp_blk * nStimInBlocks, len(inds_block))
                    inds_tp_block[inds_block[st_tap:end_tap]] = True
                    inds_both_block =  np.ix_(inds_tp_block, inds_fish)
                    y1 = np.nanmean(data[inds_both_block], axis=1)
                    plt.plot(time_inds[inds_tp_block], y1, 'x', markersize=3, color= col_vec[i])

                    try:
                        y2 = savgol_filter(y1, smooth_window, 2)
                        plt.plot(time_inds[inds_tp_block], y2, '-', color= col_vec[i], linewidth=3, alpha=0.8)

                    except:
                            warnings.warn('savgol did not converge')


        
        if not plot_retest and not plot_taps:
            plt.xlim((-0.1,8.1))
        
        if plot_taps and not plot_retest:
            plt.xlim((-0.1, 9.55))

        if first_block: # plot only first block
            plt.xlim((-0.1, 1.1))
            
        simpleaxis(plt.gca())
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.savefig(remove_brackets_invalid(save_str +'_' +dtype+ '.svg'), bbox_inches='tight', transparent=True)
        plt.savefig(remove_brackets_invalid(save_str +'_' +dtype+ '.png'), bbox_inches='tight', transparent=True, dpi=100)

        plt.show()

def plot_cum_diff(data, fish_names, fish_ids, save_name, control_index = 0, components_to_plot=range(8), n_norm = 3, n_boots = 2000, ylim=0.2):
    ### calculate cumulative difference relative to controls, as in Randlett et al., Current Biology, 2019
    # n_norm will give the number of inital responses to normalize to

    from scipy import stats

    # Hardcoded font sizes
    legend_fontsize = 12
    axes_fontsize = 18
    ticks_fontsize = 14

    plt.fill_between(np.arange(240), np.ones(240)*-0.05, np.ones(240)*0.05, color=[0.5, 0.5, 0.5], alpha=0.4)
    plt.vlines(60, -1, 1, colors='k', linestyles='dashed')
    plt.vlines(120, -1, 1, colors='k', linestyles='dashed')
    plt.vlines(180, -1, 1, colors='k', linestyles='dashed')
    plt.hlines(0, 0, 240, colors='k', linestyles='dashed')
    stim_given = data['stim_given']
    
    # color palatte to match Randlett 2019
    col_vec = [[0,0,1],  [1,0,0], [0,1,0], [0,0,0], [1,0,1], [0.9, 0.9, 0], [0.20588235, 0.41764705, 0.06470587], [0.51764706, 0.41960784, 1]]
    
    dtype_to_plot = np.array(list(data.keys()))[components_to_plot]

    for col_id, data_type in enumerate(dtype_to_plot):
        if data_type == 'Latency_(msec)': # invert so that habituation changes match direction
            data_to_plot = 1000 - abs(data[data_type][stim_given==1, :])  
        else:
            data_to_plot = abs(data[data_type][stim_given==1, :]) # use absolute values to ignore tail bending/directional signs

        cont_ids = fish_ids[0] # 0'th index is the control
        cont_data = data_to_plot[:,cont_ids]
        n_cont = len(cont_ids)

        treat_ids = fish_ids[1] # 1st index is the treatment
        treat_data = data_to_plot[:,treat_ids]
        n_treat = len(treat_ids)

        cum_diff_dist = np.zeros((240, n_boots))

        for i in range(n_boots):
            mean_treat = np.nanmean(treat_data[:, np.random.randint(0, n_treat, n_treat)], axis=1)
            mean_cont = np.nanmean(cont_data[:, np.random.randint(0, n_cont, n_cont)], axis=1)
            nan_IDs = (np.isnan(mean_treat) | np.isnan(mean_cont))
            norm_vec = np.arange(1,241)
            for el, val in enumerate(norm_vec):
                if nan_IDs[el] == True:
                    norm_vec[el:] = norm_vec[el:]-1
            
            cum_diff_dist[:, i] = np.nancumsum(mean_cont/np.nanmean(mean_cont[:n_norm]) - mean_treat/np.nanmean(mean_treat[0:n_norm]))/norm_vec


        cum_diff_dist[~np.isfinite(cum_diff_dist)] = 0
        mu = np.nanmean(cum_diff_dist, axis=1)
        sigma = np.nanstd(cum_diff_dist, axis=1)
        CI = stats.norm.interval(0.95, loc=mu, scale=sigma/np.sqrt(n_treat))
        CI[0][np.isnan(CI[0])] = mu[np.isnan(CI[0])]
        CI[1][np.isnan(CI[1])] = mu[np.isnan(CI[1])]
        plt.plot(np.arange(240), mu, color=col_vec[col_id], label=data_type.replace('_', ' '), linewidth=2)
        plt.fill_between(np.arange(240), CI[0], CI[1], alpha=0.3, color=col_vec[col_id], label='_nolegend_', interpolate=True)
    
    plt.title(fish_names[0] + ', n = ' + str(n_treat) + ' vs.\n' + fish_names[1] + ', n = ' + str(n_cont), fontsize=axes_fontsize)
    
    plt.ylabel('Relative decrease\n in response \n(norm. cumul. mean)', fontsize=axes_fontsize)
    plt.xlabel('Stimuli', fontsize=axes_fontsize)
    legend = plt.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left', fontsize=legend_fontsize, markerscale=2)

    # Increase the linewidth in the legend
    for line in legend.get_lines():
        line.set_linewidth(4.0)

    plt.xticks((0, 60, 120, 180, 240), fontsize=ticks_fontsize)
    plt.ylim((-ylim, ylim))
    plt.xlim((0,240))

    plt.savefig(remove_brackets_invalid(save_name+'.png'), dpi=100, bbox_inches='tight')
    plt.savefig(remove_brackets_invalid(save_name+'.svg'), dpi=100, bbox_inches='tight')
    plt.show()