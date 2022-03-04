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
import openseespy.postprocessing.ops_vis as opsv
import matplotlib.pyplot as plt
from openseespy.opensees import *


st.title("M-N Interaction & Moment-Curvature Calculator")

# Section Properties Defining

st.sidebar.header("Section Properties")
        
b = st.sidebar.number_input("Depth (mm): ", value=500, step=50)
h = st.sidebar.number_input("Width (mm): ",value=500, step=50)
degree = st.sidebar.selectbox("Degree Type: ", {0, 90})
cover = st.sidebar.number_input("Cover (mm): ",value=30, step=5)

st.sidebar.header("Material Properties Properties")
concrete_strength = st.sidebar.number_input("fc (MPa): ",value=25, step=100)
young_modulus_concrete = gamma_steel = st.sidebar.number_input("Ec (MPa): ",value=25000, step=100)
fctk = 0.35*math.sqrt(concrete_strength)

steel_strength = st.sidebar.number_input("fy (MPa): ",value=420, step=100)
young_modulus_steel = gamma_steel = st.sidebar.number_input("Es (MPa): ",value=200000, step=100)

gamma_concrete = st.sidebar.number_input("γconcrete (MPa): ",value=1, step=100)
gamma_steel = st.sidebar.number_input("γsteel (MPa): ",value=1, step=100)

col1, col2, col3 = st.columns(3)
with col1:
    diameter = st.number_input("Diameter of Rebar (mm): ",value=20, step=1)
with col2:
    nx = st.number_input("Number of Rebar Layer: ", value=5, step=1)
with col3:
    ny = st.number_input("Number of Rebar Layer: ", value=4, step=100)
    
diameter_area = int(diameter*diameter*math.pi/4)



# Material Properties

fcd = concrete_strength/gamma_concrete
fyd = steel_strength/gamma_steel
fctd = fctk/gamma_concrete
# 

x_dir_list = []

if degree == 0:
    d = h - cover
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
elif degree == 90:
    d = h - cover
    k = 1
    x_reb = (h-2*cover)/(ny-1)
    increase = (h-2*cover)/(ny-1)
    while k <= ny:
        if k == 1:
            x_ = cover
        elif k == ny:
            x_ = h - cover
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

if degree == 0:
    cb = ecu*(h-cover)/(esy+ecu)
    
    if cb != 0:
        if cb*k1 > h:
            k1cb_ = h
        else:
            k1cb_ = cb*k1
    total_step = int(math.ceil((h-cover)/sensibility_0)+2)
    k1cb_list = []
    c_list = []
    deneme_mb_list = []
    c_index = 1
    total_layer_index = 1
    Force_List = []
    Moment_List = []
    while c_index <= total_step:
        if c_index == 1:
            c = h*1.3
        elif c_index == 2:
            c = h
        elif c_index == total_step:
            c = cover
        else:
            c = h - sensibility_0   
            sensibility_0 = sensibility_0+k
        if c*k1 > h:
            k1cb = h 
        else:
            k1cb = c*k1 
        k1cb_list.append(k1cb)

        
        rebar_number_list = []
        rebar_distance_list = []
        diameter_list = []
        area_list = []
        Le1_list = []
        ec1_list = []
        sigma_s_list = []
        Fs1_list = []
        Mb_list = []
        distance_dif_cons = (b-2*cover)/(nx-1)
        distance_dif = (b-2*cover)/(nx-1)
        total_layer_index = 1
        while total_layer_index <= nx:
            if total_layer_index == 1:
                rebar_number = ny 
                rebar_distance = cover
                area = rebar_number*diameter_area
               
            elif total_layer_index > 1 and total_layer_index < nx:
                rebar_number = 2
                rebar_distance = cover + distance_dif
                area = rebar_number*diameter_area
                distance_dif = distance_dif + distance_dif_cons
                
            else:
                rebar_number = ny
                rebar_distance = b-cover
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
            Fc = 0.85*fcd*b*k1cb/1000
                  
            Fs1 = (area*sigma/1000) 
            Mb_ = Fc*(h/2-k1cb/2)/1000
            Mb = Fs1*(h/2-rebar_distance)/1000  
            
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
        
elif degree == 90:
    cb = ecu*(b-cover)/(esy+ecu)
    
    if cb != 0:
        if cb*k1 > b:
            k1cb_ = b
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

        
        rebar_number_list = []
        rebar_distance_list = []
        diameter_list = []
        area_list = []
        Le1_list = []
        ec1_list = []
        sigma_s_list = []
        Fs1_list = []
        Mb_list = []
        distance_dif_cons = (h-2*cover)/(ny-1)
        distance_dif = (h-2*cover)/(ny-1)
        total_layer_index = 1
        while total_layer_index <= ny:
            if total_layer_index == 1:
                rebar_number = nx 
                rebar_distance = cover
                area = rebar_number*diameter_area
               
            elif total_layer_index > 1 and total_layer_index < ny:
                rebar_number = 2
                rebar_distance = cover + distance_dif
                area = rebar_number*diameter_area
                distance_dif = distance_dif + distance_dif_cons
                
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
            Fc = 0.85*fcd*h*k1cb/1000
                  
            Fs1 = (area*sigma/1000) 
            Mb_ = Fc*(b/2-k1cb/2)/1000
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

if degree == 0:
    y1 = int(b/2.0)
    z1 = int(h/2.0)

    length_le1 = len(x_dir_list)          
    top = ['layer','straight', 3, int(ny), diameter_area, y1-cover-diameter, cover-z1+diameter, y1-cover-diameter, z1-cover-diameter]
    bottom = ['layer','straight', 3, int(ny), diameter_area, cover-y1+diameter, cover-z1+diameter, cover-y1+diameter, z1-cover-diameter]
    
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
elif degree == 90:
    y1 = int(h/2.0)
    z1 = int(b/2.0)
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

while mid_index <= length_le1-2:

    o = ['layer','straight', 3, int(2), diameter_area, y1-x_dir_list[mid_index], cover-z1+diameter, y1-x_dir_list[mid_index], z1-cover-diameter]  
    fib_sec_2.append(o)
    mid_index = mid_index + 1
    
matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
m = opsv.plot_fiber_section(fib_sec_2, matcolor=matcolor)

plt.axis('equal')    
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(m)

col1, col2, col3 = st.columns(3)
with col2:
    axialForce = st.number_input("Axial Load (kN): ", value=300, step=10) 
    
axialLoad = -1000*axialForce

df_m_n = pd.DataFrame(list(zip(Force_List, Moment_List)), columns =['Axial Force', 'Moment'], dtype = float)
Moment_List_180 = [x * -1 for x in Moment_List]
df_m_n_180 = pd.DataFrame(list(zip(Force_List, Moment_List_180)), columns =['Axial Force', 'Moment'], dtype = float)


# #-----      
plt.grid()
plt.xlabel("Moment [kNm]")
plt.ylabel("Axial Force [kN]")
plt.plot(Moment_List, Force_List, '-o')
plt.plot(Moment_List_180, Force_List, '-o')

if degree ==0:
    plt.legend(["M-N 0 Degree", "M-N 180 Degree"])
elif degree == 90:
    plt.legend(["M-N 90 Degree", "M-N 270 Degree"])
plt.show()    
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

def convert_design_df(df_m_n):
   return df_m_n.to_csv().encode('utf-8')


csv_design = convert_design_df(df_m_n)

if degree == 0:
    document_code = "M-N Interaction (0 Degree) - Press to Download"
    csv_file = "m_n_interaction_0.csv"
elif degree == 90:
    document_code = "M-N Interaction (90 Degree) - Press to Download"
    csv_file = "m_n_interaction_90.csv"
st.download_button(
   document_code,
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


def MomentCurvature(name, concrete_output,steel_output,moment_curvature_output,secTag, b1, b2, h1, h2, axialLoad, maxK, numIncr):
        
        # Define two nodes at (0,0)
        node(1, 0.0, 0.0)
        node(2, 0.0, 0.0)
    
        # Fix all degrees of freedom except axial and bending
        fix(1, 1, 1, 1)
        fix(2, 0, 1, 0)
        
        # Define element
        #                             tag ndI ndJ  secTag
        element('zeroLengthSection',  1,   1,   2,  secTag)
        
             
        # Create recorder  
        recorder('Element', '-file', concrete_output, '-precision', int(5), '-time', '-dT', float(0.1) ,'-ele', 1, 'section', 'fiber', str(b1), str(h1), '1', 'stressStrain')
        recorder('Element', '-file', steel_output, '-precision', int(5), '-time', '-dT', float(0.1) ,'-ele', 1, 'section', 'fiber', str(b2), str(h2), '3', 'stressStrain')
        recorder('Node', '-file', moment_curvature_output, '-time','-dT', float(0.1) , '-node', 2, '-dof', 3, 'disp')
    
        # Define constant axial load
        timeSeries('Constant', 1)
        pattern('Plain', 1, 1)
        load(2, int(axialLoad), 0.0, 0.0)
    
        # Define analysis parameters
        integrator('LoadControl', 0.0)
        system('SparseGeneral', '-piv')
        test('NormUnbalance', 1e-6, 10)
        numberer('Plain')
        constraints('Plain')
        algorithm('Newton')
        analysis('Static')
    
        # Do one analysis for constant axial load
        analyze(1)
    
        # Define reference moment
        timeSeries('Linear', 2)
        pattern('Plain',2, 2)
        load(2, 0.0, 0.0, 1.0)
    
        # Compute curvature increment
        dK = maxK / numIncr

        # Use displacement control at node 2 for section analysis
        integrator('DisplacementControl', 2,3,dK,1,dK,dK)

        # Do the section analysis
        analyze(numIncr)

# def ConfinedConcrete(name, secTag, b1, b2, h1, h2, axialLoad, maxK, numIncr=100):    
#     bo = width - cover
#     ho = depth - cover
    
    
wipe()
print("Start MomentCurvature.py example")

concrete_output = "concrete_strain.txt"
steel_output = "steel_strain.txt"
moment_curvature_output = "moment_curvature.txt"   
name = "name"

# ------------------------------ Moment Curvature ------------------------------
# Define model builder
# --------------------
model('basic','-ndm',2,'-ndf',3)

secTag = 1
# Define materials for nonlinear columns
# ------------------------------------------
# CONCRETE                  tag   f'c        ec0   ecu E
# Core concrete (confined)

ecc = 0.002

uniaxialMaterial('Concrete04',1, int(-fcd), float(-0.0150),  -0.02,  int(young_modulus_concrete), 0.0, 0.0, 0.1)

# Cover concrete (unconfined)
uniaxialMaterial('Concrete04',2, -fcd,  -0.002,  -0.004,  young_modulus_concrete, 0.0, 0.0, 0,1)

# uniaxialMaterial('Concrete04',1, float(fcc),  ecc,  -0.02,  Ec)

# # Cover concrete (unconfined)
# uniaxialMaterial('Concrete04',2, float(fc0),  -0.002,  -0.004,  Ec)

# STEEL
# Reinforcing steel 

by = 0.01
R0 = 15.0
cR1 = 0.925
cR2 = 0.15

#                        tag  fy E0    b
uniaxialMaterial('Steel01', 3, int(fyd), young_modulus_steel, by)

# Define cross-section for nonlinear columns
# ------------------------------------------

if degree == 0:
    width = h
    depth = b
    cover = cover
    number_of_layer = length_le1
    n_top = ny
    n_bot = ny
    n_int = 2
    
    
    b1 = width/2 - cover
    b2 = (width/2 - cover)*-1
    h1 = depth/2 - cover
    h2 = (depth/2 - cover)*-1
    
    # some variables derived from the parameters
    y1 = depth/2.0
    z1 = width/2.0
    total_y = depth - 2*cover
    total_y_layer = total_y/(number_of_layer-1)
    total_y_layer_step = total_y/(number_of_layer-1)
    
    section('Fiber', 1)
    
    # Create the concrete core fibers
    patch('rect',1,50,1 ,cover-y1, cover-z1, y1-cover, z1-cover)
    
    
    # Create the concrete cover fibers (top, bottom, left, right)
    patch('rect',2,50,1 ,-y1, z1-cover, y1, z1)
    patch('rect',2,50,1 ,-y1, -z1, y1, cover-z1)
    patch('rect',2,2,1 ,-y1, cover-z1, cover-y1, z1-cover)
    patch('rect',2,2,1 , y1-cover, cover-z1, y1, z1-cover)
    
    top = ['layer','straight', 3, n_top, diameter_area, y1-cover-diameter, cover-z1+diameter, y1-cover-diameter, z1-cover-diameter]
    bottom = ['layer','straight', 3, n_bot, diameter_area, cover-y1+diameter, cover-z1+diameter, cover-y1+diameter, z1-cover-diameter]
    
    fib_sec_1 = [['section', 'Fiber', 1],
    ['patch', 'rect',2,50,1 ,-y1, z1-cover, y1, z1],
    ['patch', 'rect',2,50,1 ,-y1, -z1, y1, cover-z1],
    ['patch', 'rect',2,2,1 ,-y1, cover-z1, cover-y1, z1-cover],
    ['patch', 'rect',2,2,1 , y1-cover, cover-z1, y1, z1-cover],
    ['patch', 'rect',1,50,1 ,cover-y1, cover-z1, y1-cover, z1-cover],
    top,
    bottom]
         
    
    # plt.savefig('fibsec_rc.png')
    
    # # Create the reinforcing fibers (left, middle, right)
    
    layer('straight', 3, n_top, diameter_area, y1-cover, cover-z1, y1-cover, z1-cover)
    layer('straight', 3, n_bot, diameter_area, cover-y1, cover-z1, cover-y1, z1-cover)
    
    total_int_layer = number_of_layer-2
    int_layer = 1
    while int_layer <= total_int_layer:
    
        layer('straight', 3, int(n_int), int(diameter_area), int(y1-cover-total_y_layer), int(cover-z1+diameter), int(y1-cover-total_y_layer), int(z1-cover-diameter))
        int_layer_def = ['layer','straight', 3, int(n_int), int(diameter_area), int(y1-cover-total_y_layer), int(cover-z1+diameter), int(y1-cover-total_y_layer), int(z1-cover-diameter)]
        fib_sec_1.append(int_layer_def)
        total_y_layer = total_y_layer + total_y_layer_step
        int_layer = int_layer +1
        
    # d -- from cover to rebar
    d = depth-cover

elif degree == 90:
    width = b
    depth = h
    cover = cover
    number_of_layer = length_le1
    n_top = nx
    n_bot = nx
    n_int = 2
    
    
    b1 = width/2 - cover
    b2 = (width/2 - cover)*-1
    h1 = depth/2 - cover
    h2 = (depth/2 - cover)*-1
    
    # some variables derived from the parameters
    y1 = depth/2.0
    z1 = width/2.0
    total_y = depth - 2*cover
    total_y_layer = total_y/(number_of_layer-1)
    total_y_layer_step = total_y/(number_of_layer-1)
    
    section('Fiber', 1)
    
    # Create the concrete core fibers
    patch('rect',1,50,1 ,cover-y1, cover-z1, y1-cover, z1-cover)
    
    
    # Create the concrete cover fibers (top, bottom, left, right)
    patch('rect',2,50,1 ,-y1, z1-cover, y1, z1)
    patch('rect',2,50,1 ,-y1, -z1, y1, cover-z1)
    patch('rect',2,2,1 ,-y1, cover-z1, cover-y1, z1-cover)
    patch('rect',2,2,1 , y1-cover, cover-z1, y1, z1-cover)
    
    top = ['layer','straight', 3, n_top, diameter_area, y1-cover-diameter, cover-z1+diameter, y1-cover-diameter, z1-cover-diameter]
    bottom = ['layer','straight', 3, n_bot, diameter_area, cover-y1+diameter, cover-z1+diameter, cover-y1+diameter, z1-cover-diameter]
    
    fib_sec_1 = [['section', 'Fiber', 1],
    ['patch', 'rect',2,50,1 ,-y1, z1-cover, y1, z1],
    ['patch', 'rect',2,50,1 ,-y1, -z1, y1, cover-z1],
    ['patch', 'rect',2,2,1 ,-y1, cover-z1, cover-y1, z1-cover],
    ['patch', 'rect',2,2,1 , y1-cover, cover-z1, y1, z1-cover],
    ['patch', 'rect',1,50,1 ,cover-y1, cover-z1, y1-cover, z1-cover],
    top,
    bottom]
         
    
    # plt.savefig('fibsec_rc.png')
    
    # # Create the reinforcing fibers (left, middle, right)
    
    layer('straight', 3, n_top, diameter_area, y1-cover, cover-z1, y1-cover, z1-cover)
    layer('straight', 3, n_bot, diameter_area, cover-y1, cover-z1, cover-y1, z1-cover)
    
    total_int_layer = number_of_layer-2
    int_layer = 1
    while int_layer <= total_int_layer:
    
        layer('straight', 3, int(n_int), int(diameter_area), int(y1-cover-total_y_layer), int(cover-z1+diameter), int(y1-cover-total_y_layer), int(z1-cover-diameter))
        int_layer_def = ['layer','straight', 3, int(n_int), int(diameter_area), int(y1-cover-total_y_layer), int(cover-z1+diameter), int(y1-cover-total_y_layer), int(z1-cover-diameter)]
        fib_sec_1.append(int_layer_def)
        total_y_layer = total_y_layer + total_y_layer_step
        int_layer = int_layer +1
        
    # d -- from cover to rebar
    d = width-cover
    
matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
m_c = opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
plt.axis('equal')    
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(m_c)       

# Estimate yield curvature
# (Assuming no axial load and only top and bottom steel)
# d -- from cover to rebar

# steel yield strain
epsy = fyd/young_modulus_steel
Ky = epsy/(0.7*d)

# Print estimate to standard output
print("Estimated yield curvature: ", Ky)
   
   
# Target ductility for analysis
mu = 30

# Number of analysis increments
numIncr = 10000

# Call the section analysis procedure

results = open('results.out','a+')

# u = nodeDisp(2,3)
# if abs(u-0.00190476190476190541)<1e-12:
#     results.write('PASSED : MomentCurvature.py\n');
#     print("Passed!")
# else:
#     results.write('FAILED : MomentCurvature.py\n');
#     print("Failed!")

results.close()

print("==========================")

MomentCurvature(name, concrete_output,steel_output,moment_curvature_output, secTag, b1, b2, h1, h2, axialLoad, Ky*mu, numIncr)

# Reading of Moment Curvature Results
with open(moment_curvature_output) as f:
    coords = f.read().split()

# Splitting data as Moment & Curvature
moment = coords[0::2]
curvature = coords[1::2]

moment = moment[:-1]
curvature = curvature[:-1]

df = pd.DataFrame(list(zip(curvature, moment)), columns =['Curvature', 'Moment'], dtype = float)
df['Curvature'] = 1000*df['Curvature']
df['Moment'] = df['Moment']/1000000
df.plot(kind='line',x='Curvature',y='Moment',color='red')
plt.grid()
plt.xlabel("Curvature [1/m]")
plt.ylabel("Moment [kNm]")
if degree ==0:
    plt.legend(["M-N 0 Degree"])
elif degree == 90:
    plt.legend(["M-N 90 Degree"])
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

def convert_design_m_c(df):
   return df.to_csv().encode('utf-8')

csv_design = convert_design_m_c(df)

if degree == 0:
    document_code_mc = "Moment - Curvature (0 Degree) - Press to Download"
    csv_file_mc = "m_n_interaction_0.csv"
elif degree == 90:
    document_code_mc = "Moment - Curvature (90 Degree) - Press to Download"
    csv_file_mc = "m_n_interaction_90.csv"
    
st.download_button(
   document_code_mc,
   csv_design,
   csv_file_mc,
   "text/csv",
   key='download-csv'
)
