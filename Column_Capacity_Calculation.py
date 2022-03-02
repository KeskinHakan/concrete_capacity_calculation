# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 22:37:06 2022

@author: hakan
"""
import numpy as np
import streamlit as st
from bokeh.plotting import figure
from pandas import read_excel
from numpy import arange
import pandas as pd
import pydeck as pdk
import math
from PIL import Image
import openseespy.postprocessing.Get_Rendering as opsplt
import openseespy.opensees as ops
import openseespy.postprocessing.ops_vis as opsv
import matplotlib.pyplot as plt

def MomentCurvature(secTag, colDepth1, colDepth2, colWidth1, colWidth2, axialLoad, concreteStrain, steelStrain, momentCurvature, maxK, numIncr=100):
    
    # Define two nodes at (0,0)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, 0.0)

    # Fix all degrees of freedom except axial and bending
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 0, 1, 0)
    
    # Define element
    #                             tag ndI ndJ  secTag
    ops.element('zeroLengthSection',  1,   1,   2,  secTag)
    

    
    # Create recorder  
    ops.recorder('Element', '-file', str(concreteStrain), '-precision', int(5), '-time', '-dT', float(0.1) , '-ele', 1, 'section', 'fiber', str(colDepth1), str(colWidth1), '1', 'stressStrain')
    ops.recorder('Element', '-file', str(steelStrain), '-precision', int(5), '-time', '-dT', float(0.1) , '-ele', 1, 'section', 'fiber', str(colDepth2), str(colWidth2), '3', 'stressStrain')
    ops.recorder('Node', '-file', str(momentCurvature), '-precision', int(3), '-time' , '-node', "2", '-dof', "3", 'disp')

    # Define constant axial load
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(2, int(axialLoad), 0.0, 0.0)

    # Define analysis parameters
    ops.integrator('LoadControl', 0.0)
    ops.system('SparseGeneral', '-piv')
    ops.test('NormUnbalance', 1e-6, 10)
    ops.numberer('Plain')
    ops.constraints('Plain')
    ops.algorithm('Newton')
    ops.analysis('Static')

    # Do one analysis for constant axial load
    ops.analyze(1)

    # Define reference moment
    ops.timeSeries('Linear', 2)
    ops.pattern('Plain',2, 2)
    ops.load(2, 0.0, 0.0, 1.0)

    # Compute curvature increment
    dK = maxK / numIncr

    # Use displacement control at node 2 for section analysis
    ops.integrator('DisplacementControl', 2,3,dK,1,dK,dK)

    # Do the section analysis
    ops.analyze(numIncr)


ops.wipe()
print("Start MomentCurvature.py example")
        
b = st.sidebar.number_input("Depth (mm): ", value=800, step=100)
h = st.sidebar.number_input("Width (mm): ",value=500, step=100)
degree = st.sidebar.number_input("Degree: ",value=0, step=100)
cover = st.sidebar.number_input("Cover (mm): ",value=30, step=100)
d = h - cover
concrete_strength = st.sidebar.number_input("fc (MPa): ",value=25, step=100)
young_modulus_concrete = gamma_steel = st.sidebar.number_input("Ec (MPa): ",value=25000, step=100)
fctk = 0.35*math.sqrt(concrete_strength)

steel_strength = st.sidebar.number_input("fy (MPa): ",value=420, step=100)
young_modulus_steel = gamma_steel = st.sidebar.number_input("Es (MPa): ",value=200000, step=100)

gamma_concrete = st.sidebar.number_input("γconcrete (MPa): ",value=1, step=100)
gamma_steel = st.sidebar.number_input("γsteel (MPa): ",value=1, step=100)

fcd = concrete_strength/gamma_concrete
fyd = steel_strength/gamma_steel
fctd = fctk/gamma_concrete

diameter = st.sidebar.number_input("Diameter of Rebar: ",value=20, step=100)
diameter_area = int(diameter*diameter*math.pi/4)
n = st.sidebar.number_input("Total Rebar Layer: ",value=4, step=100)
nx = st.sidebar.number_input("Number of Rebar Layer: ", value=5, step=100)
ny = st.sidebar.number_input("Number of Rebar Layer: ", value=4, step=100)

x_dir_list = []
k = 1
x_reb = (b-2*cover)/(nx-1)
increase = (b-2*cover)/(nx-1)
while k <= nx:
    if k == 1:
        x_ = cover
    elif k == nx:
        x_ = b - cover
    else:
        x_ = cover + increase
        increase = increase + x_reb
    x_dir_list.append(int(x_))
    k = k + 1
    

ecu = 0.003
esy = fyd/young_modulus_steel
k1 = 0.82
index_0 = 15
index_90 = 15
k =min(b,h)/index_0
sensibility_0 = min(b,h)/index_0
sensibility_90 = min(b,h)/index_90
cb = ecu*(h-cover)/(esy+ecu)

if cb != 0:
    if cb*k1 > h:
        k1cb_ = h
    else:
        k1cb_ = cb*k1
total_step = int(math.ceil((b-cover)/sensibility_0)+2)
k1cb_list = []
c_list = []
deneme_mb_list = []
c_index = 1
total_layer_index = 1
Force_List = []
Moment_List = []
while c_index <= total_step:
    if c_index == 1:
        c = b*1.3
    elif c_index == 2:
        c = b
    elif c_index == total_step:
        c = cover
    else:
        c = b - sensibility_0   
        sensibility_0 = sensibility_0+k
    if c*k1 > b:
        k1cb = b 
    else:
        k1cb = c*k1 
    k1cb_list.append(k1cb)
    Fc = 0.85*fcd*h*k1cb/1000
    Mb_ = Fc*(b/2-k1cb/2)/1000
    
    rebar_number_list = []
    rebar_distance_list = []
    diameter_list = []
    area_list = []
    Le1_list = []
    ec1_list = []
    sigma_s_list = []
    Fs1_list = []
    Mb_list = []
    distance_dif = (h-2*cover)/(n-1)
    total_layer_index = 1
    while total_layer_index <= n:
        if total_layer_index == 1:
            rebar_number = nx 
            rebar_distance = cover
            area = rebar_number*diameter_area
           
        elif total_layer_index > 1 and total_layer_index < n:
            rebar_number = 2
            rebar_distance = cover + distance_dif
            area = rebar_number*diameter_area
            distance_dif = distance_dif + distance_dif
            
        else:
            rebar_number = nx
            rebar_distance = h-cover
            area = rebar_number*diameter_area
            
        Le1 = c - rebar_distance    
        if (Le1/c)*ecu > 0:
            if (Le1/c)*ecu > esy:
                es1 = esy
            else:
                es1 = (Le1/c)*ecu
        elif (Le1/c)*ecu <= 0:
            if (Le1/c)*ecu < esy*-1:
                es1 = esy*-1
            else:
                es1 = (Le1/c)*ecu
                
        if es1*young_modulus_steel > 0:
            if es1*young_modulus_steel > fyd:
                sigma = fyd
            else:
                sigma = es1*young_modulus_steel
        elif es1*young_modulus_steel <= 0:
            if es1*young_modulus_steel <= fyd*-1:
                sigma = fyd*-1
            else:
                sigma = es1*young_modulus_steel
                
        Fs1 = (area*sigma/1000)
        Mb = Fs1*(b/2-rebar_distance)/1000
                
        rebar_number_list.append(rebar_number)
        rebar_distance_list.append(rebar_distance)
        diameter_list.append(diameter)
        area_list.append(area)
        Le1_list.append(int(Le1))
        ec1_list.append(es1)
        sigma_s_list.append(sigma)
        Fs1_list.append(Fs1)
        Mb_list.append(Mb)
        if c_index == 1:
            deneme_mb_list.append(Mb)

 
        #print(Fs1_list)
        total_layer_index = total_layer_index +1
    total_Fc = int(math.ceil(sum(Fs1_list) + Fc))
    total_Mb = int(math.ceil(sum(Mb_list) + Mb_))
    total_area = int(math.ceil(sum(area_list)))
    Moment_List.append(total_Mb)
    Force_List.append(total_Fc)
    c_list.append(c)
    c_index = c_index +1 

Nc = (b*h*0.85*fcd+fyd*total_area)/1000
Nt = -1*fyd*total_area/1000
Moment_List.insert(0, 0)
Moment_List.insert(total_step+1, 0)
Force_List.insert(0, Nc)
Force_List.insert(total_step+1, Nt)

# #-----
# concreteStrain = "Col_ConcreteStrain.mdb"
# steelStrain = "Col_SteelStrain.mdb"
# momentCurvature = "Col_MomentCurvature.mdb"

# secTag = 1
# # Define materials for nonlinear columns
# # ------------------------------------------
# # CONCRETE                  tag   f'c        ec0   ecu E
# # Core concrete (confined)
# sqrttool = math.sqrt(fcd)
# Ec = 5000*sqrttool
# ops.uniaxialMaterial('Concrete04',1, int(fcd), -0.002,  -0.02,  young_modulus_concrete, 0.0, 0.0, 0.1)

# # Cover concrete (unconfined)
# ops.uniaxialMaterial('Concrete04',2, int(fcd),  -0.002,  -0.004,  young_modulus_concrete, 0.0, 0.0, 0,1)

# # STEEL
# # Reinforcing steel 
# Ey = 200000.0    # Young's modulus
# by = 0.01
# R0 = 15.0
# cR1 = 0.925
# cR2 = 0.15

# #                        tag  fy E0    b
# ops.uniaxialMaterial('Steel02', 3, float(fyd), Ey, by, R0, cR1, cR2)


# #-----

y1 = int(b/2.0)
z1 = int(h/2.0)

# colWidth1 = (b/2 - cover)      
# colWidth2 = (b/2 - cover)*-1
# colDepth1 = (h/2 - cover)
# colDepth2 = (h/2 - cover)*-1

length_le1 = len(x_dir_list)          
top = ['layer','straight', 3, int(nx), diameter_area, y1-cover-diameter, cover-z1+diameter, y1-cover-diameter, z1-cover-diameter]
bottom = ['layer','straight', 3, int(nx), diameter_area, cover-y1+diameter, cover-z1+diameter, cover-y1+diameter, z1-cover-diameter]

mid_index = 1
lenlen = x_dir_list[mid_index]

fib_sec_2 = [['section', 'Fiber', 1],
['patch', 'rect',2,1,1 ,-y1, z1-cover, y1, z1],
['patch', 'rect',2,1,1 ,-y1, -z1, y1, cover-z1],
['patch', 'rect',2,1,1 ,-y1, cover-z1, cover-y1, z1-cover],
['patch', 'rect',2,1,1 , y1-cover, cover-z1, y1, z1-cover],
['patch', 'rect',1,1,1 ,cover-y1, cover-z1, y1-cover, z1-cover],
top,
bottom]

# #-----
# ops.section('Fiber', 1)

# # Create the concrete core fibers
# ops.patch('rect',1,50,1 ,cover-y1, cover-z1, y1-cover, z1-cover)


# # Create the concrete cover fibers (top, bottom, left, right)
# ops.patch('rect',2,10,1 ,-y1, z1-cover, y1, z1)
# ops.patch('rect',2,10,1 ,-y1, -z1, y1, cover-z1)
# ops.patch('rect',2,2,1 ,-y1, cover-z1, cover-y1, z1-cover)
# ops.patch('rect',2,2,1 , y1-cover, cover-z1, y1, z1-cover)
# ops.layer('straight', 3, int(nx), diameter_area, y1-cover-diameter, cover-z1+diameter, y1-cover-diameter, z1-cover-diameter)
# ops.layer('straight', 3, 2, diameter_area, y1-x_dir_list[mid_index], cover-z1+diameter, y1-x_dir_list[mid_index], z1-cover-diameter)
# ops.layer('straight', 3, int(nx), diameter_area, cover-y1+diameter, cover-z1+diameter, cover-y1+diameter, z1-cover-diameter)

# #-----

while mid_index <= length_le1-2:

    o = ['layer','straight', 3, int(2), diameter_area, y1-x_dir_list[mid_index], cover-z1+diameter, y1-x_dir_list[mid_index], z1-cover-diameter]  
    fib_sec_2.append(o)
    mid_index = mid_index + 1
    
matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
m = opsv.plot_fiber_section(fib_sec_2, matcolor=matcolor)

plt.axis('equal')    
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(m)

df_m_n = pd.DataFrame(list(zip(Force_List, Moment_List)), columns =['Axial Force', 'Moment'], dtype = float)
Moment_List_180 = [x * -1 for x in Moment_List]
df_m_n_180 = pd.DataFrame(list(zip(Force_List, Moment_List_180)), columns =['Axial Force', 'Moment'], dtype = float)

# #-----  
# # Estimate yield curvature
# # (Assuming no axial load and only top and bottom steel)
# # d -- from cover to rebar
# d = b-cover
# # steel yield strain
# epsy = float(fyd)/Ey
# Ky = epsy/(0.7*d)

# # Print estimate to standard output
# print("Estimated yield curvature: ", Ky)
   
   
# # Target ductility for analysis
# mu = 30

# # Number of analysis increments
# numIncr = 500

# # Call the section analysis procedure
# MomentCurvature(secTag, colDepth1, colDepth2, colWidth1, colWidth2, -10, concreteStrain, steelStrain, momentCurvature, Ky*mu, numIncr)

# results = open('results.out','a+')

# #-----      
plt.grid()
plt.xlabel("Moment [kNm]")
plt.ylabel("Axial Force [kN]")
plt.plot(Moment_List, Force_List, '-o')
plt.plot(Moment_List_180, Force_List, '-o')
plt.legend(["M-N 0 Degree", "M-N 180 Degree"])
plt.show()    
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

def convert_design_df(df_m_n):
   return df_m_n.to_csv().encode('utf-8')


csv_design = convert_design_df(df_m_n)

st.download_button(
   "M-N Interaction - Press to Download",
   csv_design,
   "m_n_interaction.csv",
   "text/csv",
   key='download-csv'
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)



