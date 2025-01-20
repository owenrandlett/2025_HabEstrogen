#%%
import FishTrack, cv2, sys, time, serial, sys, glob, os, datetime, tifffile, warnings, collections, copy, pickle, h5py
from natsort import natsorted
from pickletools import uint8
import PySimpleGUI as sg

import matplotlib.pyplot as plt


import numpy as np

from skimage.io import imsave
from scipy.ndimage import zoom
from multiprocessing import Process, Queue, SimpleQueue
from PIL import Image


# imaging/trackign requencies
frame_rate = 444  # in hZ. must match internal camera settings for timing to be correct

dipalyRate_im  = 1# in hz, max refresh rate for images
refresh_rate_params = 10 # in hz, max refresh rate for parameters window

display_zoom = 0.5
interp_method = cv2.INTER_LINEAR

baseline_hz = 40 # in hz, online tracking max rate, will slow down dynamically if buffer starts to fill
baseline_rate = int(frame_rate/baseline_hz)
n_burst_frames = frame_rate # number of frames of movie to record per stimulus



# tracking parameters
fish_len = 35
n_points = 7
tail_thresh = 3
blob_thresh = 8
search_degrees = np.pi/2.5
bkg_time = 10 # number of seconds of to use for intial background 
background_turnover = 3 # ~ factor for full background turnover, exponentail decay so not really actual time... bigger takes longer!
background_keep_prop = 1 - 1/(baseline_hz*background_turnover)

bkg_frames = bkg_time*baseline_hz
n_bkg_done = 0
new_bkg_frame = None # will be used to trigger a new background caclulation when switcing plates

burst_end = 0
stim_ind = 0
burst_time = n_burst_frames/frame_rate

# image parameters
width = 2336
height = 1728


total_frames = 0 # frame counter for camera being run
exp_frames = 0 # frame counter for experiment
bkg_im = np.zeros((height, width), dtype='uint8')
info_ch = np.zeros((height, width), dtype='uint8')
display_height = int(height*display_zoom)
display_width = int(width*display_zoom)

# burst/stimulus logic
exp_file ='/home/lab/fish_track/BigRig2/BigRig2_Experiment.csv' # firts column is triggers (in sec), second column is stim type, thrid colum is plate
exp_inst = np.genfromtxt(exp_file, delimiter=',', dtype=str)


stim_triggers = exp_inst[1:,0].astype(float)
stim_types = exp_inst[1:,1]
stim_plates = exp_inst[1:,2].astype(int)
stim_frames = (stim_triggers * frame_rate).astype(int)
mean_fr = frame_rate  # will be used for the online calculation of frame rate based on camera timestamps

#bools for updates during the experiment
calculate_background = False
done_background = False
running_experiment = False

# saving arrays
n_frames_grab = int((stim_triggers[-1] * frame_rate) + frame_rate*10)
tracking_data = {
    'tail_coords':[],
    'orientations':[],
    'heading_dir':[],
    'bend_amps':[],
    'stats_fish':[],
    'frame_index':[],
    'time_stamp':[],
    'plate':[]
}
current_plate = 0 # keep track of which plate we are on


# decue and array for burst recording
im_q = collections.deque(maxlen=n_burst_frames+1)

tail_bin = np.zeros((height, width), dtype='uint8')
im_write = np.zeros((n_burst_frames+1, height, width), dtype=np.uint8)

# set output folder
data_dir = r'/mnt/md0'
date_str = datetime.date.today().strftime("%Y%m%d")
time_str = datetime.datetime.now().strftime("%H%M%S")

out_dir = data_dir + '/' + date_str + '_' + time_str + '/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# set up file for writing online tracking data
f_h5 = h5py.File(os.path.join(out_dir, 'online_tracking.hdf5'), 'w')
first_data_write = True # used to trigger creation of datasets:

# default ROI settings
roi_start_coords = (35,15)
roi_end_coords = (width-35,height-20)
roi_spacing = 2
n_cols = 20
n_rows = 15
im_type = 'cam_im'

# IMPORT
print('Importing SiSo Wrapper')
try:
    import SiSoPyInterface as s
except ImportError:
    raise ImportError('SiSo module not loaded successfully')

print('Runtime Version', s.Fg_getSWVersion())


# DEFINITIONS

def save_tiff(image_queue):
    
    for i in range(n_burst_frames+1):
        im_write[i,:,:] = image_queue.popleft()
    

    os.chdir(out_dir)
    save_name = out_dir\
        + '_plate_' + str(stim_plates[stim_ind-1])\
        + '_burst_frame_' + str(burst_start)\
        + '_time_' + datetime.datetime.now().strftime("%H%M%S")\
        + '_stim_type_' + str(stim_types[stim_ind-1])\
        + '.tiff'
    tifffile.imwrite(save_name,im_write)


def dump_tracks(tracking_data, out_dir):
    # dump the tracking data to disk
    st = time.time()
    global first_data_write
    tail_coords = np.array(tracking_data['tail_coords'])
    n_frames, xy, n_fish, n_points = tail_coords.shape
    if first_data_write:
        dset_tail_coords = f_h5.create_dataset('tail_coords', data=tail_coords,chunks=True, maxshape = (None, 2, n_fish, n_points))
        dset_orientations = f_h5.create_dataset('orientations', data=np.array(tracking_data['orientations']),chunks=True, maxshape = (None, n_fish))
        dset_bend_amps = f_h5.create_dataset('bend_amps', data=np.array(tracking_data['bend_amps']),chunks=True, maxshape = (None, n_fish))
        dset_stats_fish = f_h5.create_dataset('stats_fish', data=np.array(tracking_data['stats_fish']),chunks=True, maxshape = (None, n_fish, 5))
        dset_frame_index = f_h5.create_dataset('frame_index', data=np.array(tracking_data['frame_index']),chunks=True, maxshape = (None,))
        dset_time_stamp = f_h5.create_dataset('time_stamp', data=np.array(tracking_data['time_stamp']),chunks=True, maxshape = (None,))
        dset_plate = f_h5.create_dataset('plate', data=np.array(tracking_data['plate']), chunks=True, maxshape = (None,)) 
        first_data_write = False
    else:
        dset_tail_coords = f_h5['tail_coords']
        n_frames_already = dset_tail_coords.shape[0]
        dset_tail_coords.resize(n_frames_already+n_frames, axis=0)
        dset_tail_coords[-n_frames:, :, :, :] = tail_coords

        dset_orientations = f_h5['orientations']
        dset_orientations.resize(n_frames_already+n_frames, axis=0)
        dset_orientations[-n_frames:, :] = np.array(tracking_data['orientations'])

        dset_bend_amps = f_h5['bend_amps']
        dset_bend_amps.resize(n_frames_already+n_frames, axis=0)
        dset_bend_amps[-n_frames:, :] = np.array(tracking_data['bend_amps'])

        dset_stats_fish = f_h5['stats_fish']
        dset_stats_fish.resize(n_frames_already+n_frames, axis=0)
        dset_stats_fish[-n_frames:, :, :] = np.array(tracking_data['stats_fish'])

        dset_frame_index = f_h5['frame_index']
        dset_frame_index.resize(n_frames_already+n_frames, axis=0)
        dset_frame_index[-n_frames:] = np.array(tracking_data['frame_index'])

        dset_time_stamp = f_h5['time_stamp']
        dset_time_stamp.resize(n_frames_already+n_frames, axis=0)
        dset_time_stamp[-n_frames:] = np.array(tracking_data['time_stamp'])

        dset_plate = f_h5['plate']
        dset_plate.resize(n_frames_already+n_frames, axis=0)
        dset_plate[-n_frames:] = np.array(tracking_data['plate'])


    for key in tracking_data.keys():
        tracking_data[key] = []
    print('dumped in ' + str(time.time() - st))

def write_pico(cmnd):
    ''' write to the serial port as one line, check for echo response back.
    '''
    global new_bkg_frame, n_bkg_done, done_background, current_plate, recording_burst, burst_grabbed, burst_start, burst_end


    cmnd_byte = (str(cmnd) + '\n').encode('utf-8')
    ser.write(cmnd_byte)
    return_str = ser.readline().decode("utf-8").strip()
    # print(return_str)
    # print('...')
    # print(cmnd)
    stim_string, stim_plate = cmnd.split(':')
    
    if stim_string == 'mc' or stim_string == 'st': # if we are moving the camera, trigger a new background in 22 seconds
        
        if running_experiment:
            new_bkg_frame = int(exp_frames + 25 * frame_rate)
        else: 
            new_bkg_frame = int(total_frames + 25 * frame_rate)
        
        current_plate = int(stim_plate)
        n_bkg_done = 0
        done_background = False
    elif running_experiment:
        recording_burst = True
        burst_grabbed = 0
        burst_start = np.copy(exp_frames)
        burst_end = burst_start + n_burst_frames
    if return_str == cmnd:
        print('pico com successful : ' + cmnd + ' , frame = ' + str(total_frames))
    else:
        warnings.warn('pico com failure : ' + cmnd + ', timing will be off...' + ' , frame = ' + str(total_frames))
    return return_str


def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

# returns count of available boards


def getNrOfBoards():
    nrOfBoards = 0
    (err, buffer, buflen) = s.Fg_getSystemInformation(
        None, s.INFO_NR_OF_BOARDS, s.PROP_ID_VALUE, 0)
    if (err == s.FG_OK):
        nrOfBoards = int(buffer)
    return nrOfBoards



# connect to pi pico running circuitpython

ports = serial_ports()
port_id = 0
print(' potential ports for pico = ')
print(ports)
print('selecting port ' + ports[port_id])

# it appears the the first port is the one we want. this may need to change at some point
ser = serial.Serial(

    ports[port_id],
    baudrate=115200,
    timeout=0.01)

time.sleep(0.5)
# move to plate 0 and turn lights on
write_pico('st:0')
#time.sleep(5)
#


# MAIN



# Board and applet selection
boardId = 0  # selectBoardDialog()




camPort = s.PORT_A

# number of buffers for acquisition
nbBuffers = int(frame_rate * 2)
samplePerPixel = 1
bytePerSample = 1
totalBufferSize = width * height * samplePerPixel * bytePerSample * nbBuffers

# Get Loaded Applet
boardType = s.Fg_getBoardType(boardId)
(err, applet) = s.Fg_findApplet(boardId)

# INIT FRAMEGRABBER

print('Initializing Board ..', end='')


fg = s.Fg_InitEx(applet, boardId, 0)

# error handling
err = s.Fg_getLastErrorNumber(fg)
mes = s.Fg_getErrorDescription(err)

if err < 0:
    print("Error", err, ":", mes)
    sys.exit()
else:
    print("ok")


#%
# allocating memory
memHandle = s.Fg_AllocMemEx(fg, totalBufferSize, nbBuffers)

# Set Applet Parameters
err = s.Fg_setParameterWithInt(fg, s.FG_WIDTH, width, camPort)
# err = s.Fg_setParameterWithInt(fg, s.FG_EXPOSURE, frameRate, camPort)
if (err < 0):
    print("Fg_setParameter(FG_WIDTH) failed: ",
          s.Fg_getLastErrorDescription(fg))
    s.Fg_FreeMemEx(fg, memHandle)
    s.Fg_FreeGrabber(fg)
    exit(err)

err = s.Fg_setParameterWithInt(fg, s.FG_HEIGHT, height, camPort)
if (err < 0):
    print("Fg_setParameter(FG_HEIGHT) failed: ",
          s.Fg_getLastErrorDescription(fg))
    s.Fg_FreeMemEx(fg, memHandle)
    s.Fg_FreeGrabber(fg)
    exit(err)

err = s.Fg_setParameterWithInt(
    fg, s.FG_BITALIGNMENT, s.FG_LEFT_ALIGNED, camPort)
if (err < 0):
    print("Fg_setParameter(FG_BITALIGNMENT) failed: ",
          s.Fg_getLastErrorDescription(fg))
    s.Fg_FreeMemEx(fg, memHandle)
    s.Fg_FreeGrabber(fg)
    exit(err)


# Read back settings
(err, oWidth) = s.Fg_getParameterWithInt(fg, s.FG_WIDTH, camPort)
if (err == 0):
    print('Width =', oWidth)
(err, oHeight) = s.Fg_getParameterWithInt(fg, s.FG_HEIGHT, camPort)
if (err == 0):
    print('Height =', oHeight)
(err, oString) = s.Fg_getParameterWithString(fg, s.FG_HAP_FILE, camPort)
if (err == 0):
    print('Hap File =', oString)
# %
# create a display window_params
# dispId0 = s.CreateDisplay(8 * bytePerSample * samplePerPixel, width, height)
# s.SetBufferWidth(dispId0, width, height)

updateRate_im  = int(frame_rate/dipalyRate_im )
updateRate_params = int(frame_rate/refresh_rate_params)
cam_im = sg.Image(filename='', key='image')
track_im = np.zeros((height, width), dtype='uint8')
#disp_info = np.zeros((int(height*display_zoom), int(width*display_zoom)))
info_im = sg.Image(key='image_disp')
# set font parameters for display image
font = cv2.FONT_HERSHEY_SIMPLEX
size = 1
stroke = 2

# get boundary of this text


# gtop = sg.Graph((200,200), (0,0),(200,200),background_color="white")

# make default ROI file
im_rois = FishTrack.make_rois(height, width, roi_start_coords, roi_end_coords,n_cols=20, n_rows=15, well_spacing=roi_spacing)
n_rois = np.max(im_rois)
# 
# run tracking functions once to ensure they are compiled            
binimage, stats, cents, conv = FishTrack.find_blobs(
    track_im,
    track_im,
    thresh=blob_thresh,
    fish_len=fish_len,
)
tail_coords, orientations, heading_dir, bend_amps, stats = FishTrack.get_head_and_tail(binimage, stats, cents, conv, im_rois, fish_len=fish_len, n_points=n_points, tail_thresh=tail_thresh, search_degrees=search_degrees)




# %






n_burst_recorded = 0
tstamps = []






stim_options = ('Move Camera', 'Dark Flash', 'Tap', 'OMR', 'Light ON', 'Lights OFF')
stim_codes = ('mc', 'df', 'tp', 'omr', 'lighton', 'lightoff')

layout_params = [
    [
        [
            sg.Multiline(default_text=str('Enter awesome science notes here'), key='notes', size=(150, 5)),
            sg.Listbox(values=(stim_options), size=(20, 5), key='stim'),
            sg.Listbox(values=('0', '1'), size=(5, 5), key='plate'),
            sg.Button('Send'), 
            sg.Text(text = 'info', key='info', size=(40,5)),
            [
            sg.Text('frame rate', size=(10,1)),
            sg.Text('track rate', size=(10,1)),
            sg.Text(r'% buff', size=(7,1)),
            sg.Text(r'lag (ms)', size=(10,1)),
            sg.Text('roi st. x',  size=(10,1)),
            sg.Text('roi st. y',  size=(10,1)),
            sg.Text('roi end x',  size=(10,1)),
            sg.Text('roi end y',  size=(10,1)),
            sg.Text('roi space', size = (10,1)),
            sg.Text('n_rows',  size=(10,1)),
            sg.Text('n_col', size = (10,1)),
            sg.Text('disp zoom', size = (10,1)),

        ], 
        [
            sg.Text(text = str(frame_rate), key = 'frame_rate', size=(10,1)),
            sg.Text(text = 'nan', key = 'track_rate', size=(10,1)),
            sg.Text(text = str(0), key = 'buff_perc', size=(7,1)),
            sg.Text(text = str(0), key = 'lag_sec', size=(10,1)),
            sg.InputText(default_text = str(roi_start_coords[0]), key = 'roi_start_x', size=(10,1)),
            sg.InputText(default_text = str(roi_start_coords[1]), key = 'roi_start_y', size=(10,1)),\
            sg.InputText(default_text = str(width - roi_end_coords[0]), key = 'roi_end_x', size=(10,1)),
            sg.InputText(default_text = str(height - roi_end_coords[1]), key = 'roi_end_y', size=(10,1)),
            sg.InputText(default_text = str(roi_spacing), key = 'roi_spacing', size=(10,1)),
            sg.InputText(default_text = str(n_rows), key = 'n_rows', size=(10,1)),
            sg.InputText(default_text = str(n_cols), key = 'n_cols', size=(10,1)),
            sg.InputText(default_text = str(display_zoom), key = 'disp_zoom', size=(10,1)),
            sg.Button('Compute ROIs', button_color='blue'),
            sg.Listbox(values=['cam_im', 'bkg', 'thresh'], default_values='cam_im', size=(10, 3), key='im_type'),
            sg.Checkbox('draw points:', default=False, key="skel_on"),
            sg.Button('Calculate bkg', button_color='orange' ),
            
        ],
          sg.Button('Start Experiment', button_color=('green'),enable_events=True),
          
          
        ],
        
            
    ],
    
]

layout_im = [[cam_im], [info_im]]
sg.ChangeLookAndFeel('Dark')

window_im = sg.Window(
    'BigRig -- image.py', 
    layout_im, 
    location=(0, 0), 
    return_keyboard_events=True, 
    use_default_focus=False)
window_im.Finalize()
# window_im.TKroot.focus_force()

window_params = sg.Window(
    'BigRig -- parameters.py', 
    layout_params, 
    location=(0, 0), 
    return_keyboard_events=True, 
    use_default_focus=False)
window_params.Finalize()
window_params.TKroot.focus_force()



color_lut =('black', 'blue','orange', 'yellow', 'red')

recording_burst = False

lag_msec = 0
done_one = False


# start acquisition
exp_start_computer = time.time()
err = s.Fg_AcquireEx(fg, camPort, s.GRAB_INFINITE, s.ACQ_STANDARD, memHandle)

if (err != 0):
    print('Fg_AcquireEx() failed:', s.Fg_getLastErrorDescription(fg))
    s.Fg_FreeMemEx(fg, memHandle)
    # s.CloseDisplay(dispId0)
    s.Fg_FreeGrabber(fg)
    exit(err)

while exp_frames < n_frames_grab:
    sec_elapsed_computer = time.time() - exp_start_computer 

    # process new frame frame?
    if s.Fg_getLastPicNumberEx(fg, camPort, memHandle) > 0:

        # get image from frame grabber
        buff_num = s.Fg_getImageEx(
            fg, s.SEL_NUMBER, total_frames+1, camPort,  5, memHandle)
        img = s.Fg_getImagePtrEx(fg,  total_frames+1, camPort, memHandle)
        im = s.getArrayFrom(img, width, height)

        (err, oTs) = s.Fg_getParameterEx(
            fg, s.FG_TIMESTAMP_LONG, camPort, memHandle,  total_frames+1)
        if total_frames == 0:
            first_time = np.copy(oTs)
            diffsec_comp_cam = exp_start_computer - first_time/1000000000
        else:
            current_delta = float(oTs - first_time)/1000000000
            tstamps.append([current_delta, total_frames, exp_frames])
            #tstamps[total_frames] = float(oTs - first_time)/1000000000
            lag_msec = int(1000*(sec_elapsed_computer - current_delta))
        
        # check camera buffer
        out_act_image = s.Fg_getStatusEx(
            fg, s.NUMBER_OF_ACT_IMAGE, total_frames+1, camPort, memHandle)

        # when this overflows we rely on the frame grabber buffer and will eventually drop frames. Must keep below
        buff_perc = 100 * (out_act_image - total_frames-1) / nbBuffers

       
       
        # time to send the next stimulus?
        
        if exp_frames in stim_frames:

            start_stim = time.time()
            stim_string = str(stim_types[stim_ind]).strip(' ')
            stim_plate = str(stim_plates[stim_ind]).strip(' ')
            write_pico(stim_string + ':' + stim_plate)
            end_cmd = time.time()
            print('...msec = ' + str(1000*(end_cmd-start_stim)))
            
            stim_ind +=1
        
       
       
        # snag images for writing burst?
        
        if recording_burst: 

            # first add the background frame as the first image
            if burst_grabbed == 0:
                im_q.append(np.copy(bkg_im))


            if burst_grabbed < n_burst_frames:
                im_q.append(np.copy(im))
                #im_q.append(im)
                #im_q.put(im)
                burst_grabbed += 1
            else:
                #p = Process(target=save_tiff, args=(copy.deepcopy(im_cue),))
                p = Process(target=save_tiff, args=(im_q,))
                p.start()
                n_burst_recorded += 1

                recording_burst = False

        # need to make a new background?
        if running_experiment and exp_frames == new_bkg_frame:
            calculate_background = True
        if not running_experiment and total_frames == new_bkg_frame:
            calculate_background = True
        
        # create background
        if calculate_background:
            if n_bkg_done == 0:
                done_background = False
                bkg_array = np.zeros((bkg_frames, height, width), dtype='uint8')
            if n_bkg_done < bkg_frames and total_frames % round(frame_rate/baseline_hz) == 0:
                bkg_array[n_bkg_done, :, :] = np.copy(im)
                n_bkg_done += 1
            if n_bkg_done >= bkg_frames:
                bkg_im = np.mean(bkg_array, axis = 0).astype(float)
                done_background = True
                calculate_background = False
                n_bkg_done = 0
                print('done bkg')
        
        # track frame and update background, only do this is the background is already calculated, and the frame buffer is empty, and we arent about to give a burst stimulus
        if (
            total_frames % baseline_rate == 0 
            and done_background
            and buff_perc < 5
            and stim_ind < len(stim_frames) 
            and frame_rate < (stim_frames[stim_ind] - exp_frames )
            ):
            
            binimage, stats, cents, conv = FishTrack.find_blobs(
                im,
                bkg_im,
                thresh=blob_thresh,
                fish_len=fish_len,
            )
            tail_coords, orientations, heading_dir, bend_amps, stats = FishTrack.get_head_and_tail(binimage, stats, cents, conv, im_rois, fish_len=fish_len, n_points=n_points, tail_thresh=tail_thresh, search_degrees=search_degrees)
            tail_coords = np.array(tail_coords)

            
            include_image = FishTrack.make_bkg_mask(stats, height,width)
             
            bkg_im[include_image] = bkg_im[include_image]*background_keep_prop + im[include_image]*(1-background_keep_prop)

            # record the tracking arrays:
            if running_experiment:
                tracking_data['tail_coords'].append(tail_coords)
                tracking_data['orientations'].append(orientations)
                tracking_data['bend_amps'].append(bend_amps)
                tracking_data['stats_fish'].append(stats)
                tracking_data['frame_index'].append(exp_frames)
                tracking_data['time_stamp'].append(current_delta)
                tracking_data['plate'].append(current_plate)
            
            # dump the tracking data if we have reached 10000 frames of tracking
            if len(tracking_data['frame_index']) > 1000: 
                dump_tracks(tracking_data, out_dir)
        
        # update the iamge display

        if  (total_frames % updateRate_im  == 0 
            and total_frames > 1000
            and stim_ind < len(stim_frames) 
            and frame_rate < (stim_frames[stim_ind] - exp_frames )
            # and buff_perc == 0 % 
            ):

            info_ch[:] = 0

            if im_type ==['cam_im']:
                base_im = im
            elif im_type == ['bkg']:
                base_im = bkg_im
            elif im_type == ['thresh']:
                base_im = binimage
                

            if skel_on and done_background and len(stats) > 0:
            

                for fish in range(n_rois):
                    tail_x = tail_coords[0, fish,:]-1
                    tail_y = tail_coords[1, fish,:]-1
                    tail_x[np.isnan(tail_x)] = 0
                    tail_y[np.isnan(tail_y)] = 0
                    info_ch[tail_y.flatten().astype(int),tail_x.flatten().astype(int)] = 255

                
                info_ch = FishTrack.add_arrows_and_text(info_ch, tail_coords[:,:,0], orientations, brightness_text = 0, brightness_arrow = 255)
            #     disp_image[tail_bin > 0] ==  disp_image[tail_bin > 0] *3
                info_ch = cv2.morphologyEx(info_ch,cv2.MORPH_DILATE, np.ones((4,4)))
                            
            info_ch[im_rois==0] = 255
            info_ch = 255-info_ch
            info_ch[info_ch > 0] = base_im[info_ch > 0] 
            # disp_image = zoom(disp_image, display_zoom)
            # disp_info = zoom(binimage, display_zoom)
            #disp_image = Image.fromarray(disp_image, 'RGB')
            disp_image = np.stack((base_im,info_ch, base_im), axis=-1).astype('uint8')
            disp_image = cv2.resize(disp_image, (display_width, display_height), interpolation=interp_method)
            

            # for i, text in enumerate(text_lines):
            #     cv2.putText(disp_info, text, (10, (offset*i) +
            #                 offset), font, size, 255, stroke)

            #
            # cam_im.Update(data=cv2.imencode('.png', disp_image)[1].tobytes())  # Update image in window_params
            cam_im.Update(data=cv2.imencode('.png', disp_image)[1].tobytes())  # Update image in window_params

            event_im, values_im = window_im.Read(timeout=1, timeout_key='timeout')
            
            # update values

            if event_im is None:
                break
        
        # update the parameters display
        if  (total_frames % updateRate_params  == 0 
            and total_frames > 1000
            and stim_ind < len(stim_frames) 
            and frame_rate < (stim_frames[stim_ind] - exp_frames )
            ):
            event, values = window_params.Read(timeout=1, timeout_key='timeout')
            
            # update values

            if event is None:
                break
            elif event == 'Send':
                if len(values['stim']) == 1 and len(values['plate']) == 1:
                    write_pico(stim_codes[stim_options.index(values['stim'][0])] + ':' + values['plate'][0])
            elif event == 'Compute ROIs':
                roi_start_coords = (
                    int(values['roi_start_x']),
                    int(values['roi_start_y'])
                    )
                roi_end_coords = (
                    width - int(values['roi_end_x']),
                    height - int(values['roi_end_y'])
                    )
                n_cols = int(values['n_cols'])
                n_rows = int(values['n_rows'])
                roi_spacing = int(values['roi_spacing'])
                
                im_rois = FishTrack.make_rois(height, width, roi_start_coords, roi_end_coords,n_cols=n_cols, n_rows=n_rows, well_spacing=roi_spacing)
                n_rois = np.max(im_rois)
            elif event == 'Calculate bkg':
                calculate_background = True
            elif event == 'Start Experiment':

                running_experiment=True
                exp_frames = 0
                window_params.TKroot.configure(background='green')

                if not done_background:
                    calculate_background = True
                
                # save experiment data
                exp_data = {}
                exp_data['exp_defs'] = exp_inst
                exp_data['im_rois'] = im_rois
                exp_data['notes'] = notes
                save_name = out_dir + 'exp_data.pkl'
                with open(save_name,"wb") as f:
                    pickle.dump(exp_data,f)
                window_params['Start Experiment'].update(disabled=True)
                
                
            im_type = values['im_type']

                
            window_params['buff_perc'].update(
                value = str(int(np.round(buff_perc, decimals=0))), 
                background_color = color_lut[np.min((4,np.round(abs(buff_perc)/100 * 4).astype(int)))]
                )
            
            window_params['lag_sec'].update(
                value = str(lag_msec)
                )
            

            if total_frames > 102:
                tstamp_hist = np.array(tstamps[total_frames-100:total_frames-2])[:,0]
                mean_fr = np.nanmean(1/np.diff(tstamp_hist))
                window_params['frame_rate'].update(value = str(np.round(mean_fr, decimals=3)))
                if np.round(mean_fr).astype(int) == frame_rate:
                    window_params['frame_rate'].update(background_color = 'black')
                else: 
                    window_params['frame_rate'].update(background_color = 'red')
            
            if len(tracking_data['time_stamp']) > 11: 
                online_tracking_rate = 1/np.nanmean(np.diff(tracking_data['time_stamp'][-10:]))
                window_params['track_rate'].update(value=str(online_tracking_rate))

            # update info window_params
            disp_string = (
                'currently imaging plate: ' + str(current_plate) +
                ', \nframes acquired: ' + str(total_frames) +
                ', \nexperiment frames acquired: ' + str(exp_frames)
            )
            if calculate_background:
                disp_string = disp_string + '\ncalculating new background...'
            
            window_params['info'].update(value=disp_string)
            notes = values['notes']
            skel_on = values['skel_on']


            # update the display zoom
            display_zoom = float(values['disp_zoom'])
            if np.isfinite(display_zoom) and display_zoom>0 and display_zoom <=1:
                display_height = int(height*display_zoom)
                display_width = int(width*display_zoom)
                    
            

        #increment frame counter
        total_frames += 1
        if running_experiment:
            exp_frames +=1

print("Acquisition stopped")
window_params.close()
window_im.close()


dump_tracks(tracking_data, out_dir)

write_pico('mc:0')
# % Clean up

s.Fg_stopAcquire(fg, camPort)
s.Fg_FreeMemEx(fg, memHandle)
s.Fg_FreeGrabber(fg)
f_h5.close()
if 'p' in globals():
    p.join()

print("Exited.")

#%%
#def dump_tracks(tracking_data)


#%%


dump_tracks(tracking_data, out_dir)


#%%



