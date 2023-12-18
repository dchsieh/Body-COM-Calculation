# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:17:39 2023

@author: dingc
"""

import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


#%%Function

def coordinate_transformation(data):
    data = np.array(data)
    x = -data[1]
    y = -data[0]
    z = data[2]
    data = np.array([x,y,z])
    return data

def get_segment_com(data1, data2):
    x = (data1[:,0]+data2[:,0])/2
    y = (data1[:,1]+data2[:,1])/2
    z = (data1[:,2]+data2[:,2])/2
    com = np.array([x,y,z])
    return com.T

def get_segment_com_fit(time, Data1, Data2):
    order = 17
    Time, data1, data2 = list(), list(), list()
    for i in range(len(time)):
        if not(np.isnan(Data1[i,0]) or np.isnan(Data1[i,1]) or np.isnan(Data1[i,2]) or np.isnan(Data2[i,0]) or np.isnan(Data2[i,1]) or np.isnan(Data2[i,2])):
            Time.append(time[i])
            data1.append(Data1[i])
            data2.append(Data2[i])
    data1 = np.array(data1)
    data2 = np.array(data2)
    x = (data1[:,0]+data2[:,0])/2
    y = (data1[:,1]+data2[:,1])/2
    z = (data1[:,2]+data2[:,2])/2
    x = np.polyval(np.polyfit(Time,x,order), time)
    y = np.polyval(np.polyfit(Time,y,order), time)
    z = np.polyval(np.polyfit(Time,z,order), time)
    com = np.array([x,y,z])
    return com.T

def plot_com(Time, data1, data2, frame, name = '', plot = False):
    com = get_segment_com(data1, data2)
    if plot:
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        ax.set_title(name)
        # ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],0], label='x')
        ax.plot(Time, com[:,1], label='y')
        # ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],2], label='z')
        ax.legend()
        ax.set_xlim(frame[0],frame[1]-50)
        ax.set_ylim(-100, 100)
    return com

def plot_com_fit(Time, data1, data2, frame, name = '', plot = False):
    com = get_segment_com_fit(Time,data1, data2)
    com, center = data_correction(Time, com, plot = False)
    if plot:
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        ax.set_title(name+'_fit')
        # ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],0], label='x')
        ax.plot(Time, com[:,1], label='y')
        # ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],2], label='z')
        ax.legend()
        ax.set_xlim(frame[0],frame[1])
        ax.set_ylim(-100, 100)
    return com, center

def data_correction(Time, data, plot = False):
    center = [np.polyval(np.polyfit(Time, data[:,0], 1),Time),np.polyval(np.polyfit(Time, data[:,1], 1),Time),np.polyval(np.polyfit(Time, data[:,2], 1),Time)]
    center = np.array(center).T
    data = data- center
    if plot:
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        ax.set_title('Corection')
        # ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],0], label='x')
        ax.plot(Time, center[:,1], label='y')
        ax.plot(Time, data[:,1], label='y')
        # ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],2], label='z')
        ax.legend()
    return np.array(data), center

def find_peak_value(data):
    y_peak = list()
    for i in range(len(data)):
        max_value = max(data[i])
        min_value = min(data[i])
        y_peak.append((max_value-min_value)/2)
    return y_peak

def plot_multiple(y_com_without_hand,y_com_with_hand):
    fig, axs = plt.subplots(2, 1, layout='constrained')
    for i in range(len(y_com_without_hand)):
        axs[0].plot(Time_without_hand[i],y_com_without_hand[i],label=str(number_without_hand[i])+', peak: '+str(round(y_peak_without_hand[i],1)))
    for i in range(len(y_com_with_hand)):
        axs[1].plot(Time_with_hand[i],y_com_with_hand[i],label=str(number_with_hand[i])+', peak: '+str(round(y_peak_with_hand[i],1)))
    for i in range(2):
        axs[i].set_ylim(-50, 50)
        axs[i].set_ylabel('Position(mm)')
        axs[i].legend()
        axs[i].grid()
    axs[0].set_title('Subject: '+str(subject_without_hand)+', Without hand, Peak avg.'+str(round(np.average(y_peak_without_hand),1)))
    axs[1].set_title('Subject: '+str(subject_without_hand)+', With hand, Peak avg.'+str(round(np.average(y_peak_with_hand),1)))
    axs[0].set_xlim(0, max(Size_without_hand))
    axs[1].set_xlim(0, max(Size_with_hand))
    axs[1].set_xlabel('Time(s)')
    fig.suptitle('Y_COM_Comparison')
    return

def plot_xyz(Time, body_com, title):
    fig, axs = plt.subplots(3, 1, layout='constrained')
    axs[0].plot(Time, body_com[:][0], label='x')
    axs[0].set_title('X')
    axs[1].plot(Time, body_com[:][1], label='y')
    axs[1].set_title('Y')
    axs[1].set_ylabel('Position(mm)')
    axs[1].set_ylim(-50, 50)
    axs[2].plot(Time, body_com[:][2], label='z')
    axs[2].set_title('Z')
    axs[2].set_xlabel('Time(s)')
    fig.suptitle(title)



def update_plot(num, data, lines):
    lines.set_data(data[frame[0]:num, 0], data[frame[0]:num, 1])
    lines.set_3d_properties(data[frame[0]:num, 2])
    return lines,

def init():
    lines.set_data([], [])
    lines.set_3d_properties([])
    return lines,
#%% Load data
# df = pd.read_excel('./data/subject001_without_hand02.xlsx')
# data = df.values.tolist()
# column_name = df.columns
# for i in range(len(column_name)):
#     temp_data = list()
#     if i == 0 or i%3==1:
#         exec(column_name[i]+'=[]')
#         for j in range(len(data)):
#             if i == 0:
#                 exec(column_name[i]+'.append(data[j][i])')
#             else:
#                 temp_data.append(coordinate_transformation(data[j][i:i+3]))
#         if i != 0:
#             exec(column_name[i]+'=np.array(temp_data)')
                
    
#%%Input data

mr_head = 8.21/100
mr_trunk = 42.88/100
mr_upper_arm = 3.25/100
mr_lower_arm = (1.36+0.54)/100 #Included wrist
mr_thigh = 13.5/100
mr_shank = 4.63/100
mr_foot = 1.175/100

start_frame = 1
end_frame = 425
# frame = [start_frame, end_frame]
#%%Subject 1

subject_without_hand = 1
number_without_hand = [1,2,3,4,5]
frame_without_hand = {1:[0,600],
                      2:[0,600],
                      3: [0,600],
                      4:[0,600],
                      5:[0,600]}
subject_with_hand = 1
number_with_hand = [1]
frame_with_hand = {1:[0,600],
                    2:[0,600],
                    3:[0,600],
                    4:[0,600],
                    5:[0,600]}
#%%Subject 2

# subject_without_hand = 2
# number_without_hand = [2,3,4,6,7]
# frame_without_hand = {2:[0,600],
#                       3:[50,600],
#                       4:[0,600],
#                       6:[0,600],
#                       7:[0,600]}
# subject_with_hand = 2
# number_with_hand = [3,4]
# frame_with_hand = {1:[0,600],
#                    2:[0,600],
#                    3:[0,600],
#                    4:[0,600],
#                    5:[0,600]}

#%%Without hand
y_com_without_hand = list()
z_com_without_hand = list()
Time_without_hand = list()
Size_without_hand = list()
for n in range(len(number_without_hand)):
    datapath = './data/subject00'+str(subject_without_hand)+'_without_hand0'+str(number_without_hand[n])+'.xlsx'
    df = pd.read_excel(datapath)
    data = df.values.tolist()
    column_name = df.columns
    for i in range(len(column_name)):
        temp_data = list()
        if i == 0 or i%3==1:
            exec(column_name[i]+'=[]')
            for j in range(len(data)):
                if i == 0:
                    exec(column_name[i]+'.append(data[j][i])')
                else:
                    temp_data.append(coordinate_transformation(data[j][i:i+3]))
            if frame_without_hand[number_without_hand[n]][1] > len(Time):  
                frame_without_hand[number_without_hand[n]][1] = len(Time)     
            temp_data = temp_data[frame_without_hand[number_without_hand[n]][0]:frame_without_hand[number_without_hand[n]][1]]
            if i != 0:
                exec(column_name[i]+'=np.array(temp_data)')
    Time = Time[frame_without_hand[number_without_hand[n]][0]:frame_without_hand[number_without_hand[n]][1]]
    Time = np.array(Time)-Time[0]
    head_com = plot_com(Time, HR, HL, frame_without_hand, 'Head')
    tb_com = plot_com(Time, TBF, TBB, frame_without_hand, 'Trunk')
    rua_com = plot_com(Time, RUAB, RUAF, frame_without_hand, 'Right Upper Arm')
    lua_com = plot_com(Time, LUAB, LUAF, frame_without_hand, 'Left Upper Arm')
    rfa_com = plot_com(Time, RFAB, RFAF, frame_without_hand, 'Right Fore Arm')
    lfa_com = plot_com(Time, LFAB, LFAF, frame_without_hand, 'Right Fore Arm')
    rt_com = plot_com(Time, RTB, RTF, frame_without_hand, 'Right thigh')
    lt_com = plot_com(Time, LTB, LTF, frame_without_hand, 'Left thigh')
    rs_com = plot_com(Time, RSB, RSF, frame_without_hand, 'Right shank')
    ls_com = plot_com(Time, LSB, LSF, frame_without_hand, 'Left shank')
    rfo_com = plot_com(Time, RFI, RFO, frame_without_hand, 'Right foot')
    lfo_com = plot_com(Time, LFI, LFO, frame_without_hand, 'Left foot')

    body_com = list()
    
    for i in range(3): 
        com_position = head_com[:,i]*mr_head + tb_com[:,i]*mr_trunk + rua_com[:,i]*mr_upper_arm + lua_com[:,i]*mr_upper_arm 
        + rfa_com[:,i]*mr_lower_arm + lfa_com[:,i]*mr_lower_arm + rt_com[:,i]*mr_thigh + lt_com[:,i]*mr_thigh 
        + rs_com[:,i]*mr_shank + ls_com[:,i]*mr_shank + rfo_com[:,i]*mr_foot + lfo_com[:,i]*mr_foot    
        body_com.append(np.array(com_position))
    body_com = np.array(body_com).T
    body_com = plot_com(Time, body_com, body_com, frame_without_hand[number_without_hand[n]], 'COM'+str(number_without_hand[n]))
    # body_com_fit, center = plot_com_fit(Time, body_com, body_com, frame_without_hand[n], 'COM'+str(n+1))
    body_com_correction, center = data_correction(Time, body_com)
    plot_xyz(Time, [body_com[:,0], body_com_correction[:,1], body_com[:,2]], 'COM_without_hand'+str(number_without_hand[n]))


    
    # fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    # ax.set_title('COM'+str(n+1))
    # ax.plot(Time, body_com[:,1], label='y_original')
    # ax.plot(Time, body_com_correction[:,1], label='y_correction')
    # ax.plot(Time, center[:,1], label='centerline')
    # ax.legend()
    y_com_without_hand.append(body_com_correction[:,1])
    z_com_without_hand.append(body_com[:,2])
    Time_without_hand.append(Time)
    Size_without_hand.append(Time[-1])
#%% With hand
y_com_with_hand = list()
z_com_with_hand = list()
Time_with_hand = list()
Size_with_hand = list()
for n in range(len(number_with_hand)):
    datapath = './data/subject00'+str(subject_with_hand)+'_with_hand0'+str(number_with_hand[n])+'.xlsx'
    df = pd.read_excel(datapath)
    data = df.values.tolist()
    column_name = df.columns
    for i in range(len(column_name)):
        temp_data = list()
        if i == 0 or i%3==1:
            exec(column_name[i]+'=[]')
            for j in range(len(data)):
                if i == 0:
                    exec(column_name[i]+'.append(data[j][i])')
                else:
                    temp_data.append(coordinate_transformation(data[j][i:i+3]))
            if frame_without_hand[number_with_hand[n]][1] > len(Time):    
                frame_without_hand[number_with_hand[n]][1] = len(Time)     
            temp_data = temp_data[frame_with_hand[number_with_hand[n]][0]:frame_with_hand[number_with_hand[n]][1]]
            if i != 0:
                exec(column_name[i]+'=np.array(temp_data)')
    Time = Time[frame_with_hand[number_with_hand[n]][0]:frame_with_hand[number_with_hand[n]][1]]  
    Time = np.array(Time)-Time[0]
    head_com = plot_com(Time, HR, HL, frame_with_hand, 'Head')
    # head_com_fit = plot_com_fit(Time, HR, HL, frame_with_hand, 'Head')
    tb_com = plot_com(Time, TBF, TBB, frame_with_hand, 'Trunk')
    rua_com = plot_com(Time, RUAB, RUAF, frame_with_hand, 'Right Upper Arm')
    lua_com = plot_com(Time, LUAB, LUAF, frame_with_hand, 'Left Upper Arm')
    rfa_com = plot_com(Time, RFAB, RFAF, frame_with_hand, 'Right Fore Arm')
    lfa_com = plot_com(Time, LFAB, LFAF, frame_with_hand, 'Right Fore Arm')
    rt_com = plot_com(Time, RTB, RTF, frame_with_hand, 'Right thigh')
    lt_com = plot_com(Time, LTB, LTF, frame_with_hand, 'Left thigh')
    rs_com = plot_com(Time, RSB, RSF, frame_with_hand, 'Right shank')
    ls_com = plot_com(Time, LSB, LSF, frame_with_hand, 'Left shank')
    rfo_com = plot_com(Time, RFI, RFO, frame_with_hand, 'Right foot')
    lfo_com = plot_com(Time, LFI, LFO, frame_with_hand, 'Left foot')
    # lfo_com_fit = plot_com_fit(Time, LFI, LFO, frame_with_hand, 'Left foot')

    body_com = list()
    
    for i in range(3): 
        com_position = head_com[:,i]*mr_head + tb_com[:,i]*mr_trunk + rua_com[:,i]*mr_upper_arm + lua_com[:,i]*mr_upper_arm 
        + rfa_com[:,i]*mr_lower_arm + lfa_com[:,i]*mr_lower_arm + rt_com[:,i]*mr_thigh + lt_com[:,i]*mr_thigh 
        + rs_com[:,i]*mr_shank + ls_com[:,i]*mr_shank + rfo_com[:,i]*mr_foot + lfo_com[:,i]*mr_foot    
        body_com.append(np.array(com_position))
    body_com = np.array(body_com).T
    body_com = plot_com(Time, body_com, body_com, frame_with_hand[number_with_hand[n]], 'COM'+str(number_with_hand[n]))
    # body_com_fit, center = plot_com_fit(Time, body_com, body_com, frame_with_hand[n], 'COM'+str(n+1))
    body_com_correction, center = data_correction(Time, body_com)
    
    plot_xyz(Time, [body_com[:,0], body_com_correction[:,1], body_com[:,2]], 'COM_with_hand'+str(number_with_hand[n]))
    
    # fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    # ax.set_title('COM'+str(n+1))
    # ax.plot(Time, body_com[:,1], label='y_original')
    # ax.plot(Time, body_com_correction[:,1], label='y_correction')
    # ax.plot(Time, center[:,1], label='centerline')
    # ax.legend()
    
    y_com_with_hand.append(body_com_correction[:,1])
    z_com_with_hand.append(body_com[:,2])
    Time_with_hand.append(Time)
    Size_with_hand.append(Time[-1])

#%% Comparison
y_peak_without_hand = find_peak_value(y_com_without_hand)
y_peak_with_hand = find_peak_value(y_com_with_hand)
plot_multiple(y_com_without_hand, y_com_with_hand)

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.set_title('Y-Z')
for i in range(len(Time_without_hand)):
    ax.plot(y_com_without_hand[i], z_com_without_hand[i], label=i+1)
for i in range(len(Time_with_hand)):
    ax.plot(np.array(y_com_with_hand[i]).T, np.array(z_com_with_hand[i]).T, dashes=[6, 2], label = 'withhand')
ax.legend()

# fig = plt.figure()
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(body_com[frame[0]:frame[1],0], body_com[frame[0]:frame[1],1], body_com[frame[0]:frame[1],2], label='body')
# ax.plot(head_com[frame[0]:frame[1],0], head_com[frame[0]:frame[1],1], head_com[frame[0]:frame[1],2], label='head')
# ax.plot(tb_com[frame[0]:frame[1],0], tb_com[frame[0]:frame[1],1], tb_com[frame[0]:frame[1],2], label='trunk')
# ax.plot(rt_com[frame[0]:frame[1],0], rt_com[frame[0]:frame[1],1], rt_com[frame[0]:frame[1],2], label='right thigh')
# ax.plot(rs_com[frame[0]:frame[1],0], rs_com[frame[0]:frame[1],1], rs_com[frame[0]:frame[1],2], label='right shank')
# ax.legend()

# Create a 3D animation
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# lines, = ax.plot([], [], [], label='body')
# ax.set_xlim(-1000, 0)
# ax.set_ylim(-100, 100)
# ax.set_zlim(500, 1500)
# ax.legend()

# animation_data = body_com  # Change this to the data you want to animate
# ani = animation.FuncAnimation(fig, update_plot, frames=range(start_frame, end_frame), fargs=(animation_data, lines),
#                               init_func=init, blit=True)

# animation_file_name = 'animation.gif'  # Change the filename and extension as needed
# ani.save(animation_file_name, writer='imagemagick')  # You may need to install imagemagick for GIFs

# still_frame_file_name = 'still_frame.png'  # Change the filename and extension as needed
# plt.savefig(still_frame_file_name)