import time
import adafruit_dotstar as dotstar
import board
import countio
import digitalio
import pwmio
import usb_cdc
import supervisor
import microcontroller
from ulab import numpy as np

stream = usb_cdc.data # stream data over usb com to host computer
stream.timeout = 0.001

# set up stepper motor control for camera
freq = 5000
duty_cycle = int(2**15)
motor_step =pwmio.PWMOut(board.GP28, frequency=freq,  variable_frequency=True)
motor_step.duty_cycle = 0
motor_step.frequency = freq
motor_dir = digitalio.DigitalInOut(board.GP27)
motor_dir.direction = digitalio.Direction.OUTPUT
motor_dir.value = False

tapper_left = pwmio.PWMOut(board.GP14, frequency=500000)
tapper_left.duty_cycle = 0
tapper_right = pwmio.PWMOut(board.GP16, frequency=500000)
tapper_right.duty_cycle = 0

tappers = (tapper_left, tapper_right)

motor_switch_left = digitalio.DigitalInOut(board.GP26)
motor_switch_left.direction = digitalio.Direction.INPUT
motor_switch_right = digitalio.DigitalInOut(board.GP17)
motor_switch_right.direction = digitalio.Direction.INPUT

motor_enable = digitalio.DigitalInOut(board.GP21)
motor_enable.direction = digitalio.Direction.OUTPUT
motor_enable.value = True

start_exp = time.monotonic()

print('rebooted...')

num_pixels = 117
brightness = 0.5
pixels_0 = dotstar.DotStar(board.GP2, board.GP3, num_pixels, brightness = brightness, auto_write=False)
pixels_1 = dotstar.DotStar(board.GP10, board.GP11, num_pixels, brightness = brightness, auto_write=False)
num_pixels_top = 32
top_px = np.arange(num_pixels_top) + 2

ud_space = 58
bottom_px = top_px + ud_space

top_bottom_vals = np.zeros(num_pixels_top)

led_strips = (pixels_0, pixels_1)
pixels_0.fill((255,255,255))
pixels_0.show()
pixels_1.fill((255,255,255))
pixels_1.show()



def darkflash(plate, off_time = 1, ramp_time = 19):
    # turn off pixels
    LEDs = led_strips[plate]
    LEDs.fill((0,0,0))
    LEDs.show()

    # wait for time off
    time.sleep(off_time)

    # ramp back on
    if ramp_time > 0:
        for val in range(1, 256, 1):
            loop_st = time.time()
            LEDs.fill((val,val,val))
            LEDs.show()
            del_time = ramp_time/254 - (time.time() - loop_st)
            del_time = max((0,del_time))
            time.sleep(del_time)
    else:
        LEDs.fill((255,255,255))
        LEDs.show()


def OMR_LR(plate, spacing=8, time_omr=55 * 60, time_cyc=30, ramp_delay=0.05):

    start_OMR = time.time()  # log time when function called

    LEDs = led_strips[plate]  # LED strip to deal with

    spaced_inds = np.arange(0, num_pixels_top, spacing)
    skip_val = 4  # increments for brightening/dimming
    max_val = 252  # max value, need to adjust along with skip value to make sure ranges match 0 - ~ 255
    offset_vals = np.arange(spacing)
    inc_dec_offset = 1

    elapsed_sec = time.time() - start_OMR  # timer for switching

    # dim the lights
    LEDs.brightness = 1
    init_bright = int(255 * brightness)
    dim_value = 0
    for val in range(init_bright, dim_value - 1, -1):
        LEDs.fill((val, val, val))
        for i in spaced_inds:
            LEDs[top_px[i]] = (255, 255, 255)
            LEDs[bottom_px[i]] = (255, 255, 255)
        LEDs.show()
        time.sleep(ramp_delay)
    # print('ramp time is')
    # print(time.monotonic() - start_OMR)

    # OMR cycles, beginning with leftward motion, flipping every 'time_cyc' seconds
    while True:
        for off_ind in range(len(offset_vals)):
            offset = offset_vals[off_ind]

            for val in range(0, max_val + 1, skip_val):
                for i in spaced_inds:
                    LEDs[top_px[(i + offset) % num_pixels_top]] = (
                        max_val - val,
                        max_val - val,
                        max_val - val,
                    )
                    LEDs[top_px[(i + offset + inc_dec_offset) % num_pixels_top]] = (
                        val,
                        val,
                        val,
                    )

                    LEDs[bottom_px[-(i + offset) % num_pixels_top]] = (
                        max_val - val,
                        max_val - val,
                        max_val - val,
                    )
                    LEDs[bottom_px[-(i + offset + inc_dec_offset) % num_pixels_top]] = (
                        val,
                        val,
                        val,
                    )

                LEDs.show()
                elapsed_sec = time.time() - start_OMR
                # if elapsed_sec > time_omr:
                #     break

        # if we roll over the direction swith time, change direction

        if int(np.floor(elapsed_sec / time_cyc)) % 2 != 0:
            offset_vals = -np.arange(spacing)
            inc_dec_offset = -1
        else:
            offset_vals = np.arange(spacing)
            inc_dec_offset = 1

        if elapsed_sec >= time_omr:
            break
    # ramp lights back on
    for val in range(dim_value, init_bright):
        LEDs.fill((val, val, val))
        for i in spaced_inds:
            LEDs[top_px[(i + offset) % num_pixels_top]] = (255, 255, 255)
            LEDs[bottom_px[-(i + offset) % num_pixels_top]] = (255, 255, 255)
        LEDs.show()
        time.sleep(ramp_delay)

    # change light settings back to normal
    LEDs.brightness = brightness
    LEDs.fill((255, 255, 255))
    LEDs.show()


def move_camera(plate):

    if plate > 1:  # only using plate 0 and 1 for now.
        plate = 1

    # plate 0 = move left, 1 = move right
    if plate == 0:
        sensor = motor_switch_left
    else:
        sensor = motor_switch_right

    dir = 1-plate
    max_freq =23000
    min_freq = 500
    n_steps = 500
    glide_sec = 8
    step_size = int((max_freq - min_freq)/n_steps)

    motor_step.duty_cycle = duty_cycle
    motor_dir.value = dir
    motor_enable.value = False

    for freq_ramp in range(min_freq,max_freq, step_size):
            motor_step.frequency = freq_ramp
            time.sleep(0.01)
            if sensor.value:
                break

    # stay at same speed for
    for i in range(int(glide_sec*100)):
        time.sleep(0.01)
        if sensor.value:
                break

    for freq_ramp in range(max_freq,min_freq, -int(step_size/2)):
            motor_step.frequency = freq_ramp
            time.sleep(0.01)
            if sensor.value:
                break

    # move toward plate until a sensor is tripped, slow down continuously in case we get stuck at high frequencies
    while not sensor.value:
        time.sleep(0.0001)



    # delay and check again that we are there and its not a flick on the sensor
    motor_step.duty_cycle = 0
    time.sleep(0.1)

    motor_step.duty_cycle = duty_cycle
    while not sensor.value:
        time.sleep(0.00001)


    # back up until switch off
    motor_dir.value = plate

    motor_step.frequency = min_freq
    motor_step.duty_cycle = duty_cycle
    while sensor.value:
        time.sleep(0.00001)

    # move forward again slowly until switch on
    motor_dir.value = dir
    while not sensor.value:
        time.sleep(0.00001)

    # move away again until switch off
    motor_dir.value = plate
    while sensor.value:
        time.sleep(0.00001)

    # turn off motor, double check sensor
    motor_step.duty_cycle = 0
    time.sleep(0.1)

    motor_step.duty_cycle = duty_cycle
    while sensor.value:
        time.sleep(0.00001)

    motor_step.duty_cycle = 0

    motor_enable.value = True

def tap_it(plate, power = int((2**16)*0.99)):
    tapper = tappers[plate]
    tapper.duty_cycle = power
    time.sleep(0.03)
    tapper.duty_cycle = 0



def startup(plate):
    # turn lights on and move to plate 0
    led_strips[0].fill((255,255,255))
    led_strips[0].show()

    led_strips[1].fill((255,255,255))
    led_strips[1].show()

    move_camera(0)


def lights_on(plate):
    if plate == 0 or plate == 1:
        LEDs = led_strips[plate]
        LEDs.fill((255,255,255))
        LEDs.show()
    elif plate == 2: # do it to both plates
        LEDs = led_strips[0]
        LEDs.fill((255,255,255))
        LEDs.show()
        LEDs = led_strips[1]
        LEDs.fill((255,255,255))
        LEDs.show()

def lights_off(plate):
    if plate == 0 or plate == 1:
        LEDs = led_strips[plate]
        LEDs.fill((0,0,0))
        LEDs.show()
    elif plate == 2: # do it to both plates
        LEDs = led_strips[0]
        LEDs.fill((0,0,0))
        LEDs.show()
        LEDs = led_strips[1]
        LEDs.fill((0,0,0))
        LEDs.show()


def give_stim(stim, plate):

    if stim == "mc":
        move_camera(plate)

    elif stim == "tp":
        tap_it(plate)

    elif stim == "df":
        darkflash(plate)

    elif stim == "omr":
        OMR_LR(plate)

    elif stim == "st":
        startup(plate)
    
    elif stim == "lighton":
        lights_on(plate)

    elif stim == "lightoff":
        lights_off(plate)

k = 0
startup(0)
while True:

    if stream.in_waiting > 0:
        written = False
        value = stream.readline().decode('utf-8').strip()
        stream.write((value).encode("utf-8"))
        stim_parts = value.split(':')
        give_stim(stim_parts[0], int(stim_parts[1]))
