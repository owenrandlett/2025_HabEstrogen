
#%%
import os, glob, tifffile, natsort, pickle, FishTrack, cv2, napari, time
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

# enter the directory where the burst tiffs are
tiff_dir = '/mnt/md0/20241118_104854'

# stabilize taps? 
stabilize_taps = True

# tracking parameters
fish_len = 35
n_points = 7
tail_thresh = 3
blob_thresh = 14
search_degrees = np.pi/2.5
show_tracks = False # show results of tracking with Napari? make sure you are only tracking one image!

def save_tracks(burst_tracking, save_name):  
    with open(save_name,"wb") as f:
        pickle.dump(burst_tracking,f)

def stabilize_tseries(IM, dpix_thresh = 0.95):
    # dpix_thresh = threshold (dpix/pixel) to apply stabilization, set empirically at 0.95 based on tap vs dark fash videos

    n_im = IM.shape[0]
    nr, nc = IM[0,:, :].shape

    # image subset for stabilization routine
    yst = int(nr/2- nr/4)
    yend = int(nr/2 + nr/4)
    xst = int(nc/2- nc/4)
    xend = int(nc/2 +  nc/4)

    dpix = np.zeros(n_im) # running total of delta pixels, if use a threshold of dpix_thresh to trigger stabilization routine


    for fr in range(n_im):
        d_frame = cv2.subtract(IM[fr, yst:yend, xst:xend], IM[0,yst:yend, xst:xend])    
        dpix[fr] = np.sum(abs(d_frame))/((yend-yst)*(xend-xst))

    fr_to_stab = np.where(dpix > dpix_thresh)[0]
    
    if len (fr_to_stab > 0):
        for fr in tqdm(fr_to_stab, desc='stabilizing ' + tiff):

            shifts, error, phasediff = phase_cross_correlation(IM[0,yst:yend, xst:xend], IM[fr,yst:yend, xst:xend], upsample_factor=20, normalization=None)
            IM[fr,:, :] = shift(IM[fr,:, :], shifts)

    return IM


# get the ROI definitions from the online tracking file
online_tracking = tiff_dir + '/exp_data.pkl'
with open(online_tracking, "rb") as f:
        track_data = pickle.load(f)


im_rois = track_data['im_rois']
n_rois = np.max(im_rois)


# find and loop through all tiffs in the directory. Could instead manually enter path to specific tiffs here. 
tiffs = natsort.natsorted(glob.glob(tiff_dir+'/*.tiff'))

for tiff in tqdm(tiffs, desc='tracking files'):
    print('tracking image ' + tiff)
    # load image
    IM = tifffile.imread(tiff)

    # stabilize t_series if 'tap' stimulus and stabilizing taps flag is true:
    if tiff.find('stim_type_tp') > -1 and stabilize_taps:
        IM = stabilize_tseries(IM)


    # first frame is online calculated background. Take median across some frames in order to find fish that didnt move before the online background. Then take the max of either of those images 
    # eiether use the background or ignore it. 
    med_IM = np.median(IM[100::20,:,:], axis=0)
    bkg = np.max(np.stack((med_IM, IM[0,:,:])), axis=0)


    
    # remainin frames are the actual movie
    IM = IM[1:,:,:]
    n_frames = IM.shape[0]


    # saving dict
    burst_tracking = {}
    burst_tracking['tail_coords'] = np.zeros((2, n_rois, n_points, n_frames))
    burst_tracking['orientations'] = np.zeros((n_rois, n_frames))
    burst_tracking['heading_dir'] = np.zeros((n_rois, n_frames))
    burst_tracking['bend_amps'] = np.zeros((n_rois, n_frames))

    for frame in range(n_frames):

        binimage, stats, cents, conv = FishTrack.find_blobs(
            IM[frame,:,:],
            bkg,
            thresh=blob_thresh,
            fish_len=fish_len,
        )

        tail_coords_frame, orientations_frame, heading_dir_frame, bend_amps_frame, stats_frame = FishTrack.get_head_and_tail(binimage, stats, cents, conv, im_rois, fish_len=fish_len, n_points=n_points, tail_thresh=tail_thresh, search_degrees=search_degrees)
        tail_coords_frame = np.array(tail_coords_frame)

        burst_tracking['tail_coords'][:,:,:,frame] = tail_coords_frame
        burst_tracking['orientations'][:, frame] = orientations_frame
        burst_tracking['heading_dir'][:,frame] = heading_dir_frame
        burst_tracking['bend_amps'][:,frame] = bend_amps_frame

    save_tracks(burst_tracking, tiff.replace('.tiff', '_BurstTracks.pkl'))

    if show_tracks:
        # make a movie with tracking dots, show in Napari. Use for testing different trackign parameters on single Tiffs
        IM_tracked = np.zeros(IM.shape)
        print('making tracking image')
        for frame in range(n_frames):

            info_ch = np.zeros((IM.shape[1],IM.shape[2] ))
            for fish in range(n_rois):
                tail_x = burst_tracking['tail_coords'][0, fish,:,frame]-1
                tail_y = burst_tracking['tail_coords'][1, fish,:, frame]-1
                tail_x[np.isnan(tail_x)] = 0
                tail_y[np.isnan(tail_y)] = 0
                info_ch[tail_y.flatten().astype(int),tail_x.flatten().astype(int)] = 255


            info_ch = FishTrack.add_arrows_and_text(info_ch, burst_tracking['tail_coords'][:,:,0, frame], burst_tracking['orientations'][:,frame], brightness_text = 255, brightness_arrow = 255)
            #     disp_image[tail_bin > 0] ==  disp_image[tail_bin > 0] *3
            info_ch = cv2.morphologyEx(info_ch,cv2.MORPH_DILATE, np.ones((2,2)))
                        
            info_ch[im_rois==0] = 255
            info_ch = 255-info_ch
            base_im = IM[frame,:,:]
            info_ch[info_ch > 0] = base_im[info_ch > 0] 
            # disp_image = zoom(disp_image, display_zoom)
            IM_tracked[frame, :,:] = info_ch

        #%
        napari.view_image(IM_tracked)
        time.sleep(10)
#%%
