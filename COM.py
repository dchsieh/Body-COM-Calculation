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
def PolyFit(xi, yi, n):
    # Create the Vandermonde matrix
    A = np.vander(xi, n+1)

    # Solve the least squares problem using the matrix method
    p, residuals, _, _ = np.linalg.lstsq(A, yi, rcond=None)

    return p

def PolyVal(p, xi):
    # TODO
    number = len(xi)
    n = len(p)-1
    A = np.ones((number,n+1))
    A[:,0] = np.ones((1,len(xi)))
    for i in range(1,n+1):
        for j in range(len(A)):
            A[j,i] = math.pow(xi[j],i)
    # print(A)
    p = p[::-1]
    yi  = A@p
    return yi

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
    x = np.polyval(np.polyfit(Time,x,order), time[frame[0]:frame[1]])
    y = np.polyval(np.polyfit(Time,y,order), time[frame[0]:frame[1]])
    z = np.polyval(np.polyfit(Time,z,order), time[frame[0]:frame[1]])
    com = np.array([x,y,z])
    return com.T

def plot_com(Time, data1, data2, frame, name = '', plot = False):
    com = get_segment_com(data1, data2)
    if plot:
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        ax.set_title(name)
        # ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],0], label='x')
        ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],1], label='y')
        # ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],2], label='z')
        ax.legend()
        ax.set_xlim(frame[0],frame[1]-50)
        ax.set_ylim(-100, 100)
    return com

def plot_com_fit(Time, data1, data2, frame, name = '', plot = False):
    com = get_segment_com_fit(Time,data1, data2)
    com = data_correction(Time, com)
    if plot:
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        ax.set_title(name+'_fit')
        # ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],0], label='x')
        ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],1], label='y')
        # ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],2], label='z')
        ax.legend()
        ax.set_xlim(frame[0],frame[1]-50)
        ax.set_ylim(-100, 100)
    return com

def data_correction(Time, data):
    center = [np.polyval(np.polyfit(Time[frame[0]:frame[1]], data[:,0], 1),Time[frame[0]:frame[1]]),np.polyval(np.polyfit(Time[frame[0]:frame[1]], data[:,1], 1),Time[frame[0]:frame[1]]),np.polyval(np.polyfit(Time[frame[0]:frame[1]], data[:,2], 1),Time[frame[0]:frame[1]])]
    center = np.array(center).T
    data = data- center
    fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    ax.set_title('Corection')
    # ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],0], label='x')
    ax.plot(Time[frame[0]:frame[1]], center[:,1], label='y')
    ax.plot(Time[frame[0]:frame[1]], data[:,1], label='y')
    # ax.plot(Time[frame[0]:frame[1]], com[frame[0]:frame[1],2], label='z')
    ax.legend()
    return np.array(data)

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

start_frame = 0
end_frame = 425
# frame = [start_frame, end_frame]
frame = [[0,450],
         [0,450],
         [0,450],
         [0,450],
         [0,450]]
#%%
for n in range(1,2):
    datapath = './data/subject001_without_hand0'+str(n)+'.xlsx'
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
            if i != 0:
                exec(column_name[i]+'=np.array(temp_data)')
                    
    Time = Time[frame[i][0]:frame[i][1]]   
    head_com = plot_com(Time, HR, HL, frame, 'Head')
    # head_com_fit = plot_com_fit(Time, HR, HL, frame, 'Head')
    tb_com = plot_com(Time, TBF, TBB, frame, 'Trunk')
    rua_com = plot_com(Time, RUAB, RUAF, frame, 'Right Upper Arm')
    lua_com = plot_com(Time, LUAB, LUAF, frame, 'Left Upper Arm')
    rfa_com = plot_com(Time, RFAB, RFAF, frame, 'Right Fore Arm')
    lfa_com = plot_com(Time, LFAB, LFAF, frame, 'Right Fore Arm')
    rt_com = plot_com(Time, RTB, RTF, frame, 'Right thigh')
    lt_com = plot_com(Time, LTB, LTF, frame, 'Left thigh')
    rs_com = plot_com(Time, RSB, RSF, frame, 'Right shank')
    ls_com = plot_com(Time, LSB, LSF, frame, 'Left shank')
    rfo_com = plot_com(Time, RFI, RFO, frame, 'Right foot')
    lfo_com = plot_com(Time, LFI, LFO, frame, 'Left foot')
    # lfo_com_fit = plot_com_fit(Time, LFI, LFO, frame, 'Left foot')

    body_com = list()
    
    for i in range(3): 
        com_position = head_com[:,i]*mr_head + tb_com[:,i]*mr_trunk + rua_com[:,i]*mr_upper_arm + lua_com[:,i]*mr_upper_arm 
        + rfa_com[:,i]*mr_lower_arm + lfa_com[:,i]*mr_lower_arm + rt_com[:,i]*mr_thigh + lt_com[:,i]*mr_thigh 
        + rs_com[:,i]*mr_shank + ls_com[:,i]*mr_shank + rfo_com[:,i]*mr_foot + lfo_com[:,i]*mr_foot    
        body_com.append(np.array(com_position))
    body_com = np.array(body_com).T
    body_com = plot_com(Time, body_com, body_com, frame, 'COM'+str(n), True)
    body_com_fit = plot_com_fit(Time, body_com, body_com, frame, 'COM'+str(n), True)


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