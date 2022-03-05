# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 13:39:41 2022

@author: hakan
"""

from openseespy.opensees import *

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
