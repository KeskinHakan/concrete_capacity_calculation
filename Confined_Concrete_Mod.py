# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:24:21 2022

@author: hakan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:56:55 2022

@author: hakan
"""

import streamlit as st
import pandas as pd
from math import sqrt
from PIL import Image
import math

# Unconfined and Confined Concrete Model (Mander Model)

def concrete_func(fcd,fyd, b, h, cover, diameter, total_rebar, dia_trans, nx, ny, n_leg_x, n_leg_y, s, young_modulus_concrete, coefficient, ecu_maximum):
    d = h-cover # clear height
    b_clear = b-cover # clear height
    A_long = 3.14*diameter**2/4 # pi için math kütüphanesi çağıralacak. Longitudinal rebar area
    A_tra = 3.14*dia_trans**2/4 
    As = A_long*total_rebar
    bo = d-cover-dia_trans
    ho = b_clear-cover-dia_trans
    A = ho*bo/1000000
    A_tra_x = A_tra*n_leg_x
    A_tra_y = A_tra*n_leg_y
    ratio_x = (n_leg_x*bo*A_tra)/(ho*bo*s)
    ratio_y = (n_leg_y*ho*A_tra)/(ho*bo*s)
    total_ratio = ratio_x + ratio_y
    
    img_1 = Image.open("ColumnSection.png")
    col1, col2, col3,col4,col5 = st.columns([1,1,1,1,1])
    col2.image(img_1, width = 500, caption = "Typical Column Section")
    
    if ny == 2:
        col1, col2, col3 = st.columns(3)
        with col1:
            ax1 = st.number_input("ax1: ", value=360.0, step=1.0)
        total_ax = ax1**2
    elif ny == 3:
        col1, col2, col3 = st.columns(3)
        with col1:
            ax1 = st.number_input("ax1: ", value=360.0, step=1.0)
            ax2 = st.number_input("ax2: ", value=180.0, step=1.0)
        total_ax = ax1**2+ax2**2
    elif ny == 4:
        col1, col2, col3 = st.columns(3)
        with col1:
            ax1 = st.number_input("ax1: ", value=143.0, step=1.0)
            ax2 = st.number_input("ax2: ", value=143.0, step=1.0)
        with col2:
            ax3 = st.number_input("ax3: ", value=143.0, step=1.0)
        total_ax = ax1**2+ax2**2+ax3**2
    elif ny == 5:
        col1, col2, col3 = st.columns(3)
        with col1:
            ax1 = st.number_input("ax1: ", value=360.0, step=1.0)
            ax2 = st.number_input("ax2: ", value=180.0, step=1.0)
        with col2:
            ax3 = st.number_input("ax3: ", value=180.0, step=1.0)
            ax4 = st.number_input("ax4: ", value=180.0, step=1.0)
        total_ax = ax1**2+ax2**2+ax3**2+ax4**2
    elif ny == 6:
        col1, col2, col3 = st.columns(3)
        with col1:
            ax1 = st.number_input("ax1: ", value=360.0, step=1.0)
            ax2 = st.number_input("ax2: ", value=180.0, step=1.0)
        with col2:
            ax3 = st.number_input("ax3: ", value=180.0, step=1.0)
            ax4 = st.number_input("ax4: ", value=180.0, step=1.0)
        with col3:
            ax5 = st.number_input("ax5: ", value=180.0, step=1.0)
        total_ax = ax1**2+ax2**2+ax3**2+ax4**2+ax5**2
    elif ny == 7:
        col1, col2, col3 = st.columns(3)
        with col1:
            ax1 = st.number_input("ax1: ", value=360.0, step=1.0)
            ax2 = st.number_input("ax2: ", value=180.0, step=1.0)
        with col2:
            ax3 = st.number_input("ax3: ", value=180.0, step=1.0)
            ax4 = st.number_input("ax4: ", value=180.0, step=1.0)
        with col3:
            ax5 = st.number_input("ax5: ", value=180.0, step=1.0)
            ax6 = st.number_input("ax6: ", value=180.0, step=1.0)
        total_ax = ax1**2+ax2**2+ax3**2+ax4**2+ax5**2+ax6**2
    if nx == 2:
        col1, col2, col3 = st.columns(3)
        with col1:
            ay1 = st.number_input("ay1: ", value=360.0, step=1.0)
        total_ay = ay1**2
    elif nx == 3:
        col1, col2, col3 = st.columns(3)
        with col1:
            ay1 = st.number_input("ay1: ", value=360.0, step=1.0)
            ay2 = st.number_input("ay2: ", value=180.0, step=1.0)
        total_ay = ay1**2+ay2**2
    elif nx == 4:
        col1, col2, col3 = st.columns(3)
        with col1:
            ay1 = st.number_input("ay1: ", value=143.0, step=1.0)
            ay2 = st.number_input("ay2: ", value=143.0, step=1.0)
        with col2:
            ay3 = st.number_input("ay3: ", value=143.0, step=1.0)
        total_ay = ay1**2+ay2**2+ay3**2
    elif nx == 5:
        col1, col2, col3 = st.columns(3)
        with col1:
            ay1 = st.number_input("ay1: ", value=360.0, step=1.0)
            ay2 = st.number_input("ay2: ", value=180.0, step=1.0)
        with col2:
            ay3 = st.number_input("ay3: ", value=180.0, step=1.0)
            ay4 = st.number_input("ay4: ", value=180.0, step=1.0)
        total_ay = ay1**2+ay2**2+ay3**2+ay4**2
    elif nx == 6:
        col1, col2, col3 = st.columns(3)
        with col1:
            ay1 = st.number_input("ay1: ", value=360.0, step=1.0)
            ay2 = st.number_input("ay2: ", value=180.0, step=1.0)
        with col2:
            ay3 = st.number_input("ay3: ", value=180.0, step=1.0)
            ay4 = st.number_input("ay4: ", value=180.0, step=1.0)
        with col3:
            ay5 = st.number_input("ay5: ", value=180.0, step=1.0)
        total_ay = ay1**2+ay2**2+ay3**2+ay4**2+ay5**2
    elif nx == 7:
        col1, col2, col3 = st.columns(3)
        with col1:
           ay1 = st.number_input("ay1: ", value=360.0, step=1.0)
           ay2 = st.number_input("ay2: ", value=180.0, step=1.0)
        with col2:
           ay3 = st.number_input("ay3: ", value=180.0, step=1.0)
           ay4 = st.number_input("ay4: ", value=180.0, step=1.0)
        with col3:
           ay5 = st.number_input("ay5: ", value=180.0, step=1.0)
           ay6 = st.number_input("ay6: ", value=180.0, step=1.0)
        total_ay = ay1**2+ay2**2+ay3**2+ay4**2+ay5**2+ay6**2
    
    total_a = 2*total_ay+ 2*total_ax
    
    ke = (1-total_a/(6*bo*ho))*(1-s/(2*bo))*(1-s/(2*ho))*(1-As/(bo*ho))**-1

    if coefficient == "Nominal":
        fyh = fyd
    elif coefficient == "Expected":
        fyh = fyd*1.2
    fex = ke*ratio_x*fyh
    fey = ke*ratio_y*fyh
    f1 = (fex+fey)/2
    if coefficient == "Nominal":
        fco = fcd
    elif coefficient == "Expected":
        fco = fcd*1.3
    lambda_c = 2.254*sqrt(1+7.94*f1/fco)-2*(f1/fco)-1.254
    
    fcc = float(format(fco*lambda_c, ".2f"))
    fsp = 0
    eco = 0.002
    ecu = 0.0035
    esp = 0.005
    ecc = eco*(1+5*(lambda_c-1))
    
    # Ec = 5000*sqrt(fco)
    Esec = fcc/ecc
    Esec_unc = fco/eco
    r = young_modulus_concrete/(young_modulus_concrete-Esec)
    r_unc = young_modulus_concrete/(young_modulus_concrete-Esec_unc)
    f_cu = fco*(ecu/eco)*r_unc/(r_unc-1+(ecu/eco)**r_unc)
    
    e = 0
    fc_conf= []
    fc_unconf= []
    ec_conf = []
    ec_unconf = []
    x_conf = []
    x_unconf = []
    while e < 0.02:
            x = e/ecc
            fc = (fcc*x*r)/(r-1+x**r)
            fc_conf.append(format(fc, ".2f"))
            ec_conf.append(format(e, ".5f"))
            x_conf.append(format(x, ".3f"))
            e = e + 0.0001
            
    e = 0
    while e <= esp:
            x = e/eco
            if e <= ecu:
                fc = fco*x*r_unc/(r_unc-1+x**r_unc)
            elif e > ecu and e<=esp:
                fc = f_cu+(e-ecu)*((fsp-f_cu)/(esp-ecu))
            fc_unconf.append(format(fc, ".2f"))
            ec_unconf.append(format(e, ".5f"))
            x_unconf.append(format(x, ".3f"))
            e = e + 0.0001
            
    df_conf = pd.DataFrame(list(zip(ec_conf, fc_conf)), columns =['Strain', 'Stress'], dtype = float)

    #Line Chart
    as_list = df_conf["Strain"].tolist()
    
    df_conf.index = as_list
    
    df_fc = df_conf['Stress']
    st.line_chart(df_fc)
    
    df_unconf = pd.DataFrame(list(zip(ec_unconf, fc_unconf)), columns =['Strain', 'Stress'], dtype = float)
    
    #Line Chart
    as_list = df_unconf["Strain"].tolist()
    
    df_unconf.index = as_list
    
    df_fc = df_unconf['Stress']
    st.line_chart(df_fc)
    
    
    fcc_max = df_conf.loc[df_conf['Stress']==fcc]
    eco_max = fcc_max['Strain'].iloc[0]
    ecu_max = df_conf.loc[df_conf['Strain']==ecu_maximum]
    fcu_max = ecu_max['Stress'].iloc[0]
    
    
    def convert_confined_df(df_conf):
        return df_conf.to_csv().encode('utf-8')
    def convert_unconfined_df(df_unconf):
        return df_unconf.to_csv().encode('utf-8')
    
    
    csv_conf = convert_confined_df(df_conf)
    csv_unconf = convert_unconfined_df(df_unconf)
    
    st.download_button(
        "Confined Concrete - Press to Download",
        csv_conf,
        "confined.csv",
        "text/csv",
        key='download-csv'
    )
    
    st.download_button(
        "Unconfined Concrete - Press to Download",
        csv_unconf,
        "unconfined.csv",
        "text/csv",
        key='download-csv'
    )
    
    return fcc, eco, esp, ecu_max, fcu_max, df_conf, eco_max

# coefficient = st.sidebar.selectbox("Material Coefficient: ", {"Nominal", "Expected"})


# b = st.sidebar.number_input("Depth (mm): ", value=500, step=50)
# h = st.sidebar.number_input("Width (mm): ",value=500, step=50)
# degree = st.sidebar.selectbox("Degree Type: ", {0, 90})
# cover = st.sidebar.number_input("Cover (mm): ",value=30, step=5)

# st.sidebar.header("Material Properties Properties")
# concrete_strength = st.sidebar.number_input("fc (MPa): ",value=25, step=100)
# young_modulus_concrete = gamma_steel = st.sidebar.number_input("Ec (MPa): ",value=25000, step=100)
# fctk = 0.35*math.sqrt(concrete_strength)

# steel_strength = st.sidebar.number_input("fy (MPa): ",value=420, step=100)
# young_modulus_steel = gamma_steel = st.sidebar.number_input("Es (MPa): ",value=200000, step=100)

# gamma_concrete = st.sidebar.number_input("γconcrete (MPa): ",value=1, step=100)
# gamma_steel = st.sidebar.number_input("γsteel (MPa): ",value=1, step=100)

# col1, col2, col3 = st.columns(3)
# with col1:
#     diameter = st.number_input("Diameter of Rebar (mm): ",value=20, step=1)
# with col2:
#     nx = st.number_input("Number of Rebar Layer: ", value=4, step=1)
# with col3:
#     ny = st.number_input("Number of Rebar Layer: ", value=4, step=100)
    
# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     dia_trans = st.number_input("Diameter of Transverse Rebar - Φ: ",value=8.0, step=1.0)
# with col2:
#     s = st.number_input("Spacing of Transverse Rebar - s: ",value=150.0, step=10.0)
# with col3:
#     n_leg_x = st.number_input("Number of Transverse Rebar - X Dir - n_leg_x: ",value=3.0, step=1.0)
# with col4:
#     n_leg_y = st.number_input("Number of Transverse Rebar - Y Dir - n_leg_y: ",value=3.0, step=1.0)

    
# total_rebar = nx*2 + (ny-2)*2
# diameter_area = int(diameter*diameter*math.pi/4)


# # Material Properties

# fcd = concrete_strength/gamma_concrete
# fyd = steel_strength/gamma_steel
# fctd = fctk/gamma_concrete   

# ec_conf, fc_conf, fc_unconf, ec_unconf = soilType_func(fcd,fyd, b, h, cover, diameter, total_rebar, dia_trans, nx, ny, n_leg_x, n_leg_y, s, young_modulus_concrete)

# e = 0.0036
# fc1 = f_cu+(e-ecu)*((fsp-f_cu)/(esp-ecu))



