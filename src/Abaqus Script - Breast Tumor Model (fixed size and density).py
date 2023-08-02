# -*- coding: mbcs -*-
import sys
import os
import random
import math
import xlsxwriter
import numpy as np


from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *

# Set the flag to True if odb is not needed
remove_odb = True
failure_flag = 0
trial_index = 0


##### Initiate file creation #####
features = xlsxwriter.Workbook('C:/Users/fabulenc/Documents/2-2023/features_600_fixedDensityAndSize.xlsx')
labels = xlsxwriter.Workbook('C:/Users/fabulenc/Documents/2-2023/labels_600_fixedDensityAndSize.xlsx')

f_worksheet = features.add_worksheet()
l_worksheet = labels.add_worksheet()

# List to store failed simulation info
failed_mesh = []
failed_earlystop = []

num_sims = 600


##### Breast Parameters (FIXED) #####
breast_radius = 0.040 # in m
E_breast = 0.02*(10**6)    # Pascal
v_breast = 0.4999985
p_breast = 1000


##### Mesh calculations #####
K = E_breast/(3*(1-2*v_breast))
c = np.sqrt(K/p_breast)
f = 1*1000000 # Frequency in Hz

lambda_ = c/f

mesh_low = lambda_/10
mesh_high = lambda_/15


##### Transducer Partition Calculation (Edge) for RF-P #####
# Probe Parameters
probe_radius = 0.012 # in m
s = probe_radius/2

theta = s/breast_radius # in radians
 
x_coord_probe = 0.040*math.tan(theta)  # in m
y_coord_probe = 0.040*math.cos(theta)  # in m


##### Specify Range to Vary Tumor Parameters #####

E_tumor_list = [0]*num_sims
tumorSize_list = [0]*num_sims
tumorDensity_list = [0]*num_sims

random.seed(42)


for i in range(0,num_sims):
       # 0.1-0.5 MPa
    tumorSize_list[i] = 0.015  #0.01    # 10-20 mm
    tumorDensity_list[i] = 1030 #random.uniform(1040, 1080)  # kg/m3

E_tumor_list = np.linspace(100000, 500000, num_sims)

##### Specify Time Parameters #####
total_time = 7e-05
time_increment = 2e-09   #user-fixed time increment
sampling_number = total_time / time_increment


c_fixed = 1480   #1500 m/s in literature value for breast tissue
current_index = 1

while current_index <= num_sims:

    p_tumor = tumorDensity_list[current_index-1]
    E_tumor = E_tumor_list[current_index-1]
    v_tumor = ((-E_tumor/(3*p_tumor*c_fixed*c_fixed))+1)/2


    tumorSize = tumorSize_list[current_index-1]
    tumorSize_radius = tumorSize/2


    

    ##### Creates Breast Part #####
    mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=1.0)
    mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
        viewStyle=AXISYM)
    mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(point1=(0.0, 
        -0.5), point2=(0.0, 0.5))
    mdb.models['Model-1'].sketches['__profile__'].FixedConstraint(entity=
        mdb.models['Model-1'].sketches['__profile__'].geometry[2])
    mdb.models['Model-1'].sketches['__profile__'].CircleByCenterPerimeter(center=(
        0.0, 0.0), point1=(0.04, 0.0))
    mdb.models['Model-1'].sketches['__profile__'].Line(point1=(0.0, 0.0), point2=(
        0.0, 0.04))
    mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
        False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[4])
    mdb.models['Model-1'].sketches['__profile__'].Line(point1=(0.0, 0.0), point2=(
        0.04, 0.0))
    mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
        addUndoState=False, entity=
        mdb.models['Model-1'].sketches['__profile__'].geometry[5])
    mdb.models['Model-1'].sketches['__profile__'].autoTrimCurve(curve1=
        mdb.models['Model-1'].sketches['__profile__'].geometry[3], point1=(
        -0.0369940288364887, 0.0143605135381222))
    mdb.models['Model-1'].sketches['__profile__'].autoTrimCurve(curve1=
        mdb.models['Model-1'].sketches['__profile__'].geometry[7], point1=(
        0.0335051640868187, -0.0218111537396908))
    mdb.models['Model-1'].Part(dimensionality=AXISYMMETRIC, name='Part-1', type=
        DEFORMABLE_BODY)
    mdb.models['Model-1'].parts['Part-1'].BaseShell(sketch=
        mdb.models['Model-1'].sketches['__profile__'])
    del mdb.models['Model-1'].sketches['__profile__']
    mdb.models['Model-1'].Material(name='Breast')


    ##### Specify Material Parameters #####
    mdb.models['Model-1'].materials['Breast'].Density(table=((p_breast, ), ))
    mdb.models['Model-1'].materials['Breast'].Elastic(table=((E_breast, v_breast), 
        ))
    mdb.models['Model-1'].Material(name='Tumor')
    mdb.models['Model-1'].materials['Tumor'].Density(table=((p_tumor, ), ))
    mdb.models['Model-1'].materials['Tumor'].Elastic(table=((E_tumor, v_tumor), 
        ))


    mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.002, name='__profile__', 
    sheetSize=0.113, transform=
    mdb.models['Model-1'].parts['Part-1'].MakeSketchTransform(
    sketchPlane=mdb.models['Model-1'].parts['Part-1'].faces[0], 
    sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.016977, 0.016977, 
    0.0)))
    mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
        decimalPlaces=3)
    mdb.models['Model-1'].parts['Part-1'].projectReferencesOntoSketch(filter=
        COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
    del mdb.models['Model-1'].sketches['__profile__']
    mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.002, name='__profile__', 
        sheetSize=0.113, transform=
        mdb.models['Model-1'].parts['Part-1'].MakeSketchTransform(
        sketchPlane=mdb.models['Model-1'].parts['Part-1'].faces[0], 
        sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.016977, 0.016977, 
        0.0)))
    mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
        decimalPlaces=3)

    ##### Sketch Tumor by Specifying Radius #####

    mdb.models['Model-1'].parts['Part-1'].projectReferencesOntoSketch(filter=
        COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
    mdb.models['Model-1'].sketches['__profile__'].CircleByCenterPerimeter(center=(
        -0.016977, 0.003023), point1=(-0.017, (0.003023+tumorSize_radius)))


    mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
        addUndoState=False, entity1=
        mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
        mdb.models['Model-1'].sketches['__profile__'].geometry[4])
    mdb.models['Model-1'].sketches['__profile__'].EqualDistanceConstraint(
        addUndoState=False, entity1=
        mdb.models['Model-1'].sketches['__profile__'].vertices[2], entity2=
        mdb.models['Model-1'].sketches['__profile__'].vertices[0], midpoint=
        mdb.models['Model-1'].sketches['__profile__'].vertices[3])
    mdb.models['Model-1'].sketches['__profile__'].autoTrimCurve(curve1=
        mdb.models['Model-1'].sketches['__profile__'].geometry[6], point1=(
        -0.0214893044401407, 0.00461527430820465))
    mdb.models['Model-1'].parts['Part-1'].PartitionFaceBySketch(faces=
        mdb.models['Model-1'].parts['Part-1'].faces.getSequenceFromMask(('[#1 ]', 
        ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
    del mdb.models['Model-1'].sketches['__profile__']
    mdb.models['Model-1'].HomogeneousSolidSection(material='Breast', name=
        'Section-1-Breast', thickness=None)
    mdb.models['Model-1'].HomogeneousSolidSection(material='Tumor', name=
        'Section-2-Tumor', thickness=None)
    mdb.models['Model-1'].parts['Part-1'].SectionAssignment(offset=0.0, 
        offsetField='', offsetType=MIDDLE_SURFACE, region=Region(
        faces=mdb.models['Model-1'].parts['Part-1'].faces.getSequenceFromMask(
        mask=('[#1 ]', ), )), sectionName='Section-1-Breast', thicknessAssignment=
        FROM_SECTION)
    mdb.models['Model-1'].parts['Part-1'].SectionAssignment(offset=0.0, 
        offsetField='', offsetType=MIDDLE_SURFACE, region=Region(
        faces=mdb.models['Model-1'].parts['Part-1'].faces.getSequenceFromMask(
        mask=('[#2 ]', ), )), sectionName='Section-2-Tumor', thicknessAssignment=
        FROM_SECTION)


    ##### Assembly #####
    mdb.models['Model-1'].rootAssembly.DatumCsysByThreePoints(coordSysType=
        CYLINDRICAL, origin=(0.0, 0.0, 0.0), point1=(1.0, 0.0, 0.0), point2=(0.0, 
        0.0, -1.0))

    mdb.models['Model-1'].rootAssembly.Instance(dependent=OFF, name='Part-1-1', 
    part=mdb.models['Model-1'].parts['Part-1'])
    mdb.models['Model-1'].rootAssembly.ReferencePoint(point=(x_coord_probe, 
    y_coord_probe, 0.0))


    mdb.models['Model-1'].rootAssembly.PartitionEdgeByParam(edges=
    mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].edges.getSequenceFromMask(
    ('[#8 ]', ), ), parameter=0.903419530020852)
    mdb.models['Model-1'].rootAssembly.Set(edges=
        mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].edges.getSequenceFromMask(
        ('[#10 ]', ), ), name='PROBE')
    mdb.models['Model-1'].rootAssembly.Surface(name='PROBE-SURF', side1Edges=
        mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].edges.getSequenceFromMask(
        ('[#10 ]', ), ))
    mdb.models['Model-1'].rootAssembly.PartitionEdgeByPoint(edge=
        mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].edges[4], point=
        mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].InterestingPoint(
        mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].edges[4], MIDDLE))
    mdb.models['Model-1'].rootAssembly.Set(edges=
        mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].edges.getSequenceFromMask(
        ('[#20 ]', ), ), name='SENSOR')

    ##### Step #####

    mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-1', 
    previous='Initial', timeIncrementationMethod=FIXED_USER_DEFINED_INC, 
    timePeriod=total_time, userDefinedInc=time_increment)

    ##### History Output Request (U2) #####

    mdb.models['Model-1'].HistoryOutputRequest(createStepName='Step-1', frequency=1
    , name='H-Output-2', rebar=EXCLUDE, region=
    mdb.models['Model-1'].rootAssembly.sets['SENSOR'], sectionPoints=DEFAULT, 
    variables=('U2', ))


    ##### Displacement Boundary Condition #####
    mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Step-1', 
    distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
    'BC-1', region=Region(
    edges=mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].edges.getSequenceFromMask(
    mask=('[#4 ]', ), )), u1=0.0, u2=0.0, ur3=0.0)


    mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Step-1', 
    distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
    'BC-2', region=Region(
    edges=mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].edges.getSequenceFromMask(
    mask=('[#c2 ]', ), )), u1=0.0, u2=UNSET, ur3=UNSET)

    mdb.models['Model-1'].TabularAmplitude(data=((3e-08, 0.00193832), (6e-08, 
    0.007331565), (9e-08, 0.014955369), (1.2e-07, 0.02290189), (1.5e-07, 
    0.028768258), (1.8e-07, 0.029899718), (2.1e-07, 0.02366855), (2.4e-07, 
    0.007766768), (2.7e-07, -0.019510884), (3e-07, -0.059016994), (3.3e-07, 
    -0.110556088), (3.6e-07, -0.172761899), (3.9e-07, -0.243060632), (4.2e-07, 
    -0.31772778), (4.5e-07, -0.392039522), (4.8e-07, -0.460513061), (5.1e-07, 
    -0.517223685), (5.4e-07, -0.556180509), (5.7e-07, -0.571737999), (6e-07, 
    -0.559016994), (6.3e-07, -0.514307163), (6.6e-07, -0.435422866), (6.9e-07, 
    -0.321986312), (7.2e-07, -0.175615544), (7.5e-07, -1.84e-16), (7.8e-07, 
    0.199147085), (8.1e-07, 0.414262793), (8.4e-07, 0.636230724), (8.7e-07, 
    0.854787048), (9e-07, 1.059016994), (9.3e-07, 1.237916106), (9.6e-07, 
    1.380985813), (9.9e-07, 1.478829772), (1.02e-06, 1.523716342), (1.05e-06, 
    1.510073511), (1.08e-06, 1.43488558), (1.11e-06, 1.297965853), (1.14e-06, 
    1.102086081), (1.17e-06, 0.85295126), (1.2e-06, 0.559016994), (1.23e-06, 
    0.231155583), (1.26e-06, -0.117814271), (1.29e-06, -0.473711225), (
    1.32e-06, -0.821658865), (1.35e-06, -1.146802247), (1.38e-06, 
    -1.435035365), (1.41e-06, -1.673700482), (1.44e-06, -1.852221406), (
    1.47e-06, -1.962636182), (1.5e-06, -2.0), (1.53e-06, -1.962636182), (
    1.56e-06, -1.852221406), (1.59e-06, -1.673700482), (1.62e-06, 
    -1.435035365), (1.65e-06, -1.146802247), (1.68e-06, -0.821658865), (
    1.71e-06, -0.473711225), (1.74e-06, -0.117814271), (1.77e-06, 0.231155583), 
    (1.8e-06, 0.559016994), (1.83e-06, 0.85295126), (1.86e-06, 1.102086081), (
    1.89e-06, 1.297965853), (1.92e-06, 1.43488558), (1.95e-06, 1.510073511), (
    1.98e-06, 1.523716342), (2.01e-06, 1.478829772), (2.04e-06, 1.380985813), (
    2.07e-06, 1.237916106), (2.1e-06, 1.059016994), (2.13e-06, 0.854787048), (
    2.16e-06, 0.636230724), (2.19e-06, 0.414262793), (2.22e-06, 0.199147085), (
    2.25e-06, 5.51e-16), (2.28e-06, -0.175615544), (2.31e-06, -0.321986312), (
    2.34e-06, -0.435422866), (2.37e-06, -0.514307163), (2.4e-06, -0.559016994), 
    (2.43e-06, -0.571737999), (2.46e-06, -0.556180509), (2.49e-06, 
    -0.517223685), (2.52e-06, -0.460513061), (2.55e-06, -0.392039522), (
    2.58e-06, -0.31772778), (2.61e-06, -0.243060632), (2.64e-06, -0.172761899), 
    (2.67e-06, -0.110556088), (2.7e-06, -0.059016994), (2.73e-06, 
    -0.019510884), (2.76e-06, 0.007766768), (2.79e-06, 0.02366855), (2.82e-06, 
    0.029899718), (2.85e-06, 0.028768258), (2.88e-06, 0.02290189), (2.91e-06, 
    0.014955369), (2.94e-06, 0.007331565), (2.97e-06, 0.00193832), (3e-06, 
    0.0)), name='Amp-1', smooth=SOLVER_DEFAULT, timeSpan=STEP)

    mdb.models['Model-1'].Pressure(amplitude='Amp-1', createStepName='Step-1', 
        distributionType=UNIFORM, field='', magnitude=1000000.0, name='OneMHz', 
        region=mdb.models['Model-1'].rootAssembly.surfaces['PROBE-SURF'])
    mdb.models['Model-1'].rootAssembly.seedEdgeBySize(constraint=FINER, 
        deviationFactor=0.1, edges=
        mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].edges.getSequenceFromMask(
        ('[#ff ]', ), ), size=mesh_low)
    mdb.models['Model-1'].rootAssembly.setMeshControls(regions=
        mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].faces.getSequenceFromMask(
        ('[#1 ]', ), ), technique=STRUCTURED)
    mdb.models['Model-1'].rootAssembly.PartitionFaceByAuto(face=
        mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].faces[1])
    mdb.models['Model-1'].rootAssembly.PartitionFaceByAuto(face=
        mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].faces[1])
    mdb.models['Model-1'].rootAssembly.setMeshControls(elemShape=QUAD, regions=
        mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].faces.getSequenceFromMask(
        ('[#f ]', ), ), technique=STRUCTURED)
    mdb.models['Model-1'].rootAssembly.generateMesh(regions=(
        mdb.models['Model-1'].rootAssembly.instances['Part-1-1'], ))


    mdb.Job(activateLoadBalancing=False, atTime=None, contactPrint=OFF, 
        description='', echoPrint=OFF, explicitPrecision=DOUBLE, historyPrint=OFF, 
        memory=90, memoryUnits=PERCENTAGE, model='Model-1', modelPrint=OFF, 
        multiprocessingMode=DEFAULT, name='Breast_Simulation_'+str(current_index), nodalOutputPrecision=FULL, 
        numCpus=1, numDomains=1, parallelizationMethodExplicit=DOMAIN, queue=None, 
        resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=
        0, waitMinutes=0)


    mdb.jobs['Breast_Simulation_'+str(current_index)].submit(consistencyChecking=OFF)
    mdb.jobs['Breast_Simulation_'+str(current_index)].waitForCompletion()


    ##### Open ODB file after the job is finished #####
    Odb_path = 'Breast_Simulation_'+str(current_index)+'.odb'
    odb = openOdb(Odb_path)
    
    step = 'Step-1'

    # Open history steps data
    step1 = odb.steps[step]

    # Use a loop to get the  data of all the sensor nodes
    set_sensor = odb.rootAssembly.nodeSets['SENSOR'].nodes[0]

    try:
        region = step1.historyRegions['Node PART-1-1.' + str(set_sensor[0].label)]

    except KeyError:
        odb.close()
        failure_flag = 1
		
        failed_mesh.append(current_index)
        pass

    else:
        time, U2_total = zip(
                *step1.historyRegions['Node PART-1-1.' + str(set_sensor[0].label)].historyOutputs['U2'].data)
        

        for i in range(1, len(set_sensor)):
            time, U2 = zip(
                *step1.historyRegions['Node PART-1-1.' + str(set_sensor[i].label)].historyOutputs['U2'].data)
            U2_total = [a + b for a, b in zip(U2_total, U2)]

        U2_ave = [a / len(set_sensor) for a in U2_total]

        nn = len(U2_ave)

        for i in range(nn):
            f_worksheet.write_number(i, current_index - 1, U2_ave[i])

        l_worksheet.write_number(0, current_index - 1, E_tumor)
        l_worksheet.write_number(1, current_index - 1, tumorSize)
        l_worksheet.write_number(2, current_index - 1, p_tumor)
        l_worksheet.write_number(3, current_index - 1, v_tumor)
    
        odb.close()
        failure_flag = 0


    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.abq')
    except WindowsError:
        pass
    
    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.inp')
    except WindowsError:
        pass
    
    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.ipm')
    except WindowsError:
        pass
    
    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.pac')
    except WindowsError:
        pass
    
    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.prt')
    except WindowsError:
        pass

    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.res')
    except WindowsError:
        pass
    
    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.sel')
    except WindowsError:
        pass

    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.stt')
    except WindowsError:
        pass
    
    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.sta')
    except WindowsError:
        pass
        
    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.mdl')
    except WindowsError:
        pass
        
    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.log')
    except WindowsError:
        pass
        
    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.msg')
    except WindowsError:
        pass
        
    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.dat')
    except WindowsError:
        pass
        
    try:
        os.remove('Breast_Simulation_'+str(current_index)+'.com')
    except WindowsError:
        pass

    if remove_odb:
        try:
            os.remove('Breast_Simulation_' + str(current_index) + '.odb')
            
        except WindowsError:
            pass

    current_index = current_index + 1

features.close()
labels.close()

# Write error message
len_fail_mesh = len(failed_mesh)

with open('Failed_simulation.txt', 'w') as f:
	for i in range(0, len_fail_mesh):
		f.write('%s, ' % failed_mesh[i])
	f.write('\nOut of '+str(num_sims)+' simulations, '+str(len_fail_mesh)+' have failed due to mesh generation.')