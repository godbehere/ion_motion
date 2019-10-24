
from math import pi
import math
import numpy as np
import matplotlib.pyplot as plt

avagadros = 6.0221409*10**23
default_f = 0.8
default_r = .004
electron_q = -1.602*10**-19

def find_v(q, mz = 500, f = default_f, r = default_r):
    freq = 2 * pi * f * 1000000
    electron = electron_q
    m = calculate_mass(mz)
    v = (q*m*(freq**2)*(r**2))/(4*electron)
    return v

def find_q(v, mz = 500, f = default_f, r = default_r):
    freq = 2 * pi * f * 1000000
    electron = electron_q
    m = calculate_mass(mz)
    q = (4*electron*v)/(m*(freq**2)*(r**2))
    return q

def find_u(a, mz = 500, f = default_f, r = default_r):
    freq = 2 * pi * f * 1000000
    electron = electron_q
    m = calculate_mass(mz)
    u = (a*m*(freq**2)*(r**2))/(8*electron)
    return u

def find_a(u, mz = 500, f = default_f, r = default_r):
    freq = 2 * pi * f * 1000000
    electron = electron_q
    m = calculate_mass(mz)
    a = (8*electron*u)/(m*(freq**2)*(r**2))
    return a

def calculate_mass(mz, z=1):
    mass = (mz*z)/(avagadros*1000)
    return mass

def calculate_motion(init_x,init_y,a,q,mz,step,duration,flag_unstable=False,magnetic_field=False,B=1.4,theta=pi/2,*optional_arguments):
    unstable_motion = False         #flag for unstable ion motion
    x_motion = np.array([init_x])   #an array to contain all x positions. first index is set to initial x position
    y_motion = np.array([init_y])   #an array to contain all y positions. first index is set to initial y position
    m = calculate_mass(mz)          #finds the mass in Kilograms (assuming charge number is 1)
    v = find_v(q,mz)                #finds required RF voltage based on q value and M/Z
    u = find_u(a,mz)                #finds required DC voltage based on a value and M/Z
    current_v_x = 0                 #sets the initial velocity in the x direction to 0
    current_v_y = 0                 #sets the initial velocity in the y direction to 0
                                    #steps --> an array that contains all of the values to be used as time in future calculations
                                    #steps cont. - this array is generated based on duration and step size input to this function
    steps = np.array([x for x in np.arange(step,duration,step)])
    print('RF Voltage(V) is: {}v'.format(v))
    print('DC Voltage(U) is: {}v'.format(u))
    #print(steps)

    #this for loop is used to calculate the motion of ions
    for time in steps:
            
            #---------------------------------#
            #--------Calculate X Steps--------#
            #---------------------------------#
        current_x_index = len(x_motion)-1           #sets the current index so that current ion position can be referenced
        current_x = x_motion[current_x_index]       #sets the current position using current index
        fx = fx_trap(mz,time,current_x,v,u)         #uses trapping force function to calculate force on ion due to RF/DC quadrupole 
        
        if magnetic_field == True:
            f_mag_x = magnetic_force(electron_q,current_v_x,B,theta)
            fx += f_mag_x
        
        acc_x = fx/m                                #calculates current acceleration using force and mass
        current_v_x += acc_x*step                   #calculates change in velocity and adds it to the current velocity
        current_x += current_v_x*step               #calculates change in position and adds it to the current position
        x_motion = np.append(x_motion, current_x)   #adds the new position to the position array
        #print(time)
        #acc = -(a-2*q*cos(2*time))*current_x
        #print("Acceleration @ {} = {}".format(time, acc_x))
        #print("Velocity @ {} = {}".format(time, current_v_x))
        #print("Position @ {} = {}".format(time, current_x))
            
            #----------------------------------#
            #--------Calculate Y Steps---------#
            #----------------------------------#
        current_y_index = len(y_motion)-1           #sets the current index so that current ion positioncan be referenced
        current_y = y_motion[current_y_index]       #sets the current position using current index
        fy = fy_trap(mz,time,current_y,v,u)         #uses trapping force function to calculate force on ion due to RF/DC quadrupole 
        
        if magnetic_field == True:
            f_mag_y = magnetic_force(electron_q,current_v_y,B,theta)
            fy += f_mag_y
        
        acc_y = fy/m                                #calculates current acceleration using force and mass
        current_v_y += acc_y*step                   #calculates change in velocity and adds it to the current velocity
        current_y += current_v_y*step               #calculates change in position and adds it to the current position
        y_motion = np.append(y_motion, current_y)   #adds the new position to the position array
        #print(time)
        #acc = -(a-2*q*cos(2*time))*current_x
        #print("Acceleration @ {} = {}".format(time, acc))
        #print("Velocity @ {} = {}".format(time, current_v))
        #print("Position @ {} = {}".format(time, current_x))
        
        if flag_unstable:
            if current_x > default_r * 1000 or current_y > default_r * 1000:        #check if ion position is outside r naught
                steps = np.array([x for x in np.arange(step,time+(2*step),step)])   #regenerates steps array for plotting purposes
                print("ION MOTION IS NOT STABLE")                                   #tells user that motion is not stable
                unstable_motion = True                                              #sets unstable motion flag to True
                break                                                               #exits loop

    #print(steps)

    fig, (ax1, ax2, comb) = plt.subplots(3)         #creates plot for displaying x_pos*time graph; y_pos*time graph; and x_pos&y_pos*time graph
    fig2, ax3 = plt.subplots(1)                     #creates plot for displaying x*y graph
    if unstable_motion == False:    
        steps = np.append(steps, duration)          #adds additional element to steps array to correct for mismatch in length between positions arrays and steps array (this is due to starting the positions arrays with an initial position value)
    ax1.plot(steps,x_motion,'r-')                   #creates first axis with a red line
    ax2.plot(steps,y_motion)                        #creates second axis with default line (blue)
    comb.plot(steps,x_motion,'r-',steps,y_motion)   #creates third axis (which combines the first two)
    ax3.plot(x_motion,y_motion)                     #creats axis on second plot
    ax3.set(xlim=(-4,4),ylim=(-4,4))                #sets size for second plot
    fig.suptitle('Position vs Time')
    fig2.suptitle('X-Y Plot')
    #ax3.set_aspect('equal','box')
    plt.show()                                      #shows all plots

def fx_trap(mz,t,x,v,u):
    freq = 2 * pi * default_f * 1000000
    r = default_r
    #m = calculate_mass(mz)
    electron = electron_q
    x_pos = x
    f = -(2*electron*(v*math.cos(freq*t) + u)*x_pos)/r**2
    #print('The frequency is: {}'.format(freq))
    #print('Quad radius is: {}m'.format(r))
    #print('Mass of molecule is: {}kg'.format(m))
    #print('Electron charge is: {}C'.format(electron))
    #print('X position is: {}m'.format(x_pos))
    #print('Time is: {}s'.format(t))
    #print('The force in the X direction is: {} Newtons'.format(f))
    return f

def fy_trap(mz,t,y,v,u):
    freq = 2 * pi * default_f * 1000000
    r = default_r
    #m = calculate_mass(mz)
    electron = electron_q
    y_pos = y
    f = (2*electron*(v*math.cos(freq*t) + u)*y_pos)/r**2
    #print('Quad radius is: {}m'.format(r))
    #print('Mass of molecule is: {}kg'.format(m))
    #electron = -2.527*10**-29
    #print('Electron charge is: {}C'.format(electron))
    #print('X position is: {}m'.format(x_pos))
    #print('Time is: {}s'.format(t))
    #print('The force in the Y direction is: {} Newtons'.format(f))
    return f

def magnetic_force(charge,v,B=1.4,theta=pi/2):
    f = charge * v * B * math.sin(theta)
    return f

#print(find_q(261))
#print(find_a(31))
calculate_motion(1,-1,.3,.706,500,.0000001,.00005,True,False)
#print(calculate_mass(500))
#print(magnetic_force(20e-9,10,5e-5))