#!/usr/bin/env python
# coding: utf-8

info = '''
    Requirements: Python 3.x
                  Numpy package
                  Matplotlib package
                  MDAnalysis package (MUST HAVE!!!)

    This code is developed by Cong Phuong Cao, as member of VNU Keylab.

    It was based from paper:

    Khelashvili, G., Kollmitzer, B., Heftberger, P., Pabst, G., & Harries, D. (2013). 
    Calculating the bending modulus for multicomponent lipid membranes in different 
    thermodynamic phases. Journal of chemical theory and computation, 9(9), 3866-3871.

    https://pubs.acs.org/doi/10.1021/ct400492e

    If you have any problem, please contact my email:
    
        >>>     congphuongcao@gmail.com     <<<
        
    First update: 27/03/2021
    Last update:  17/04/2021
'''

print(info)


import sys
sys.path.append('/home/cao/anaconda3/envs/hus/lib/python3.8/site-packages')


import datetime


import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt 
from matplotlib import gridspec as grs

from scipy.optimize import curve_fit


import MDAnalysis as mda
from MDAnalysis import transformations as trans


import warnings


import ast # for reading dictionary from file


import configparser


# configparser updates values from config file
# https://docs.python.org/3/library/configparser.html
# https://www.programcreek.com/python/example/72204/configparser.Error

parser = configparser.ConfigParser()
parser.read('Kc.cfg')


# debug flag
try:
    debug_mode = parser.getboolean('Program config', 'debug_mode')
except configparser.Error:
    debug_mode = False


# BE CAREFUL OF THESE LINES!!!
np.seterr(divide = 'ignore')

warnings.filterwarnings('ignore',category=UserWarning)


# read the data from tpr and xtc
# https://www.mdanalysis.org/pages/basic_example/
# https://www.mdanalysis.org/MDAnalysisTutorial/basics.html
# https://www.mdanalysis.org/MDAnalysisTutorial/atomgroups.html

try:
    input_tpr = parser.get('User config', 'input_tpr')
except configparser.Error:
    print('Error: Input .tpr file missing!')
    sys.exit(1)
    
try:
    input_xtc = parser.get('User config', 'input_xtc')
except configparser.Error:
    print('Error: Input .xtc file missing!')
    sys.exit(1)

lipsys = mda.Universe(input_tpr, input_xtc)
print(lipsys)


# define tilt vector for molecule types: point to the outer of membrane
# this dictionary must be updated before every new calculation!!!
# resvedict = {
#     'DPPC': ['C214 C215 C216 C314 C315 C316', 'C2  P'],
#     'POPC': ['C216 C217 C218 C314 C315 C316', 'C2  P'],
#     'PSM' : ['C14F C15F C16F C16S C17S C18S', 'C2S P'],
#     'CHL1': ['C17',                           'C3'],
# }
try:
    dictionary_data = parser.get('User config', 'dictionary_data')
except configparser.Error:
    dictionary_data = 'dictionary.dat'

with open(dictionary_data, 'r') as f: d_data = f.read()

resvedict = ast.literal_eval(d_data)

# define ratio threshold if a molecule type can be neglected
try:
    threshold_ratio = parser.getfloat('Program config', 'threshold_ratio')
except configparser.Error:
    threshold_ratio = 0.03     # 3%

# define direction of membrane
try:
    Oaxis = ast.literal_eval(parser.get('User config', 'membrane_axis'))
except configparser.Error:
    Oaxis = [0, 0, 1]     # z-axis


# number of frame to calculate bending modulus!!!
try:
    n_frame = parser.getint('Program config', 'n_frame')
except configparser.Error:
    n_frame = lipsys.trajectory.n_frames


# file to write system infomation during the run
try:
    file_log = parser.get('User config', 'file_log')
except configparser.Error:
    file_log = 'Kc.log'


# the temperature of the system
try:
    Temp = parser.getfloat('User config', 'temperature')
except configparser.Error:
    Temp = 323.15 # Kelvin

# Boltzmann constant
kB = 1.38064852e-23 # J per Kelvin


# molecules in range radius are counted as neighbors
try:
    neighbor_radius = parser.getfloat('Program config', 'neighbor_radius')
except configparser.Error:
    neighbor_radius = 20

# neighbor_list is updated every update_rate frames
try:
    update_rate = parser.getint('Program config', 'update_rate')
except configparser.Error:
    update_rate = 10 


# only alpha in a specific range are fitted
try:
    fit_range = ast.literal_eval(parser.get('User config', 'fit_range'))
except configparser.Error:
    fit_range = [5, 20] # in degree


def fit_func(x, chi_12, c):
    '''quadratic function for fitting'''
    
    return chi_12 * x ** 2 + c


# unwrap frames to make molecules whole
# https://userguide.mdanalysis.org/stable/trajectories/transformations.html

workflow = [trans.unwrap(lipsys.atoms)]
lipsys.trajectory.add_transformations(*workflow)


print('maximum n_frame = ', lipsys.trajectory.n_frames)


print('n_residues = ', lipsys.atoms.n_residues)


print('n_atoms = ', lipsys.atoms.n_atoms)


print('box =', lipsys.dimensions[:3])
box = lipsys.dimensions[:3]
half_box = box * .5


if debug_mode:
    tmp = np.array([i for i in lipsys.atoms.resnames])
    print(tmp)
    print(tmp.shape)

    tmp = np.array([i for i in lipsys.atoms.types])
    print(tmp)
    print(tmp.shape)

    tmp = np.array([i for i in lipsys.atoms.names])
    print(tmp)
    print(tmp.shape)

    tmp = np.array([coor for coor in lipsys.trajectory.ts.positions])
    print(tmp)
    print(tmp.shape)


residict = {} # dictionary of number of residue type
i_old = -1
res_total = 0

reslisdict = {} # dictionary of id list of residue type

for i in lipsys.residues.resids:
    resname = lipsys.residues.resnames[i]
    
    if (not residict.get(resname)):
        residict[resname] = 1
        reslisdict[resname] = [i]
    elif (i != i_old):
        i_old = i
        residict[resname] += 1
        reslisdict[resname].append(i)

        
for i in residict:
    res_total += residict[i]
    
    if debug_mode:
        print(str(i) + ' \t= ' + str(residict[i]))
        print(reslisdict[i])
    
    
print('total residues = ' + str(res_total))


def TCVN(ares):
    '''return: (T)ilt angle, (C)enter of geometry, tilt (V)ector, and (N)orm of tilt vector of molecule'''
    
    center = ares.center_of_geometry()
    
    name = ares.resnames[0]
    
    A = ares.select_atoms('name ' + resvedict[name][0]).center_of_geometry()
    B = ares.select_atoms('name ' + resvedict[name][1]).center_of_geometry()
    
    vecAB = B - A
    
    norAB = norm(vecAB)
    
    angle = np.arccos(np.dot(vecAB, Oaxis) / (norAB * norm(Oaxis))) # in radian
    angle = np.rad2deg(angle) # in degree
    
    return angle, center, vecAB, norAB


# leaflets
reslisdic = [{}, {}] # for Oaxis+ and Oaxis-
leaflet = [[], []] # contain all residue id of a single leaflet

res_total_new = 0

fig_num = 0

iglist = [] # ignored residue types


# check the distribution of tilt angle of all residue type

for i in residict:
    
    if (residict[i] / res_total < threshold_ratio): 
        iglist.append(i)
        continue
    
    reslisdic[0][i] = []
    reslisdic[1][i] = []
    
    res_total_new += residict[i]
    cnt1 = 0
    cnt2 = 0
    tmp1 = []
    tmp2 = []
    
    for idx in reslisdict[i]:
        ares = lipsys.select_atoms('resid ' + str(idx))
        angle = TCVN(ares)[0]
        
        if (angle < 90): 
            reslisdic[0][i].append(idx)
            cnt1 += 1
            tmp1.append(angle)
        else:
            reslisdic[1][i].append(idx)
            cnt2 += 1
            tmp2.append(angle)
            
        if (60 < angle and angle < 120):
            print('Warning: Tilt angle is too big at ', i, idx, angle)
        
        
    if (cnt1 != cnt2):
        print('Error: Leaflets unbalanced!')
        sys.exit(1)
    
    leaflet[0].extend(reslisdic[0][i])
    leaflet[1].extend(reslisdic[1][i])
    
    if debug_mode:
        print('Plotting ' + i + ' ...')

        fig_num += 1
        plt.figure(fig_num)
        
        plt.hist(np.concatenate((tmp1, tmp2)), bins=np.linspace(0, 180, 90), density=False)
        plt.title('Tilt angle distribution of ' + i)
        plt.xlabel('Tilt angle, degree')
        plt.ylabel('Probability density (raw count)')
        
#         plt.subplot(1, 2, 1)
#         plt.hist(tmp1, bins=np.linspace(0, 90, 50), density=True)
        
#         plt.subplot(1, 2, 2)
#         plt.hist(tmp2, bins=np.linspace(90, 180, 50), density=True)
        
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig('Tilt angle distribution of ' + i + ' at the first frame.png')
        
#         plt.show()


# set the current frame of trajectory to the beginning
lipsys.trajectory[0]


# https://www.mdanalysis.org/MDAnalysisTutorial/writing.html

# W = lipsys.select_atoms("all")
# W.write('test.gro')

# lipsys.trajectory.next()

# W = lipsys.select_atoms("all")
# W.write('test2.gro')

# ... to check wrap function!


def real_distance(A, B):
    '''return distance between residue A and B concerning PBC'''
    
    vecAB = abs(B - A)
    
    for i in range(3):
        if (vecAB[i] > half_box[i]):
            vecAB[i] = box[i] - vecAB[i]
    
    return norm(vecAB)


resname = list(reslisdic[0].keys())
N = len(resname)

print('Ignoring for residues:', iglist)
print('Calculating for residues:', resname)


til = [None] * res_total
cen = [None] * res_total
vec = [None] * res_total
nor = [None] * res_total


def brute_force_loop():
    '''calculate the distributions of the current frame so detailed with no tricks'''

    # initialize values to reduce calculations

    for I in range(N):
        res = resname[I]

        for leaf in range(2):
            n = len(reslisdic[leaf][res])

            for i in range(n):
                r_i = reslisdic[leaf][res][i]
                ares = lipsys.select_atoms('resid ' + str(r_i))
                til[r_i], cen[r_i], vec[r_i], nor[r_i] = TCVN(ares)


    # the main loop for 1 frame!
    # method : Brute-force
    
    fig_num += 1
    plt.figure(fig_num)
    
    gs0 = grs.GridSpec(2, 1)
    gs0.update(wspace=0.025, hspace=0.025)
    
    for I in range(N):
        res1 = resname[I]

        for J in range(I, N):
            res2 = resname[J]

            alpha = []
            distance = []

            print('Calculating alpha of pairs of', res1, 'and', res2, '...')

            for leaf in range(2):
                print('  leaflet', leaf, '...')

                n1 = len(reslisdic[leaf][res1])
                n2 = len(reslisdic[leaf][res2])

                for i in range(n1):

                    for j in range(n2):

                        if (res1 == res2 and j <= i): continue

                        r_i = reslisdic[leaf][res1][i]
                        r_j = reslisdic[leaf][res2][j]

                        dis = real_distance(cen[r_i], cen[r_j])
                        distance.append(dis)

                        if (dis < 10 and (til[r_i] < 10 or til[r_j] < 10)):
                            angle = np.arccos(np.dot(vec[r_i], vec[r_j]) / (nor[r_i] * nor[r_j])) # in radian
                            angle = np.rad2deg(angle) # in degree
                            alpha.append(angle)


            print('Complete!')

            print(len(alpha), 'alpha')
            print(len(distance), 'distance')
            

            plt.subplot(gs0[0])
            P_alpha, bins = np.histogram(alpha, bins=np.linspace(0, 90, 91), density=True)
            alpha_new = .5 * (bins[1:] + bins[:-1])
            plt.plot(alpha_new, P_alpha, label=res1 + '/' + res2)

            plt.subplot(gs0[1])
            PMF = -1 * np.log(P_alpha / np.sin(np.deg2rad(alpha_new)))
            plt.plot(alpha_new, PMF, label=res1 + '/' + res2)
            
            plt.tight_layout()

    
    plt.subplot(gs0[0])
    plt.ylabel('P(alpha)')
    plt.legend(loc='upper right')
    plt.xlim([0, 90])

    plt.subplot(gs0[1])
    plt.ylabel('Potential of Mean force, kBT')
    plt.xlabel('splay angle, alpha, degree')
    plt.legend(loc='upper right')
    plt.xlim([0, 90])

    fig = plt.gcf()
    fig.set_size_inches(12.5, 10.5)
    fig.savefig('P_alpha and PMF at the first frame.png')
    
#     plt.show()


if debug_mode: brute_force_loop()


def check_displacement():
    '''check the distribution of displacement of all molecules between the first and last frame'''
    
    lipsys.trajectory[-1]

    max_displacement = []

    for I in range(N):
        res = resname[I]

        for leaf in range(2):
            n = len(reslisdic[leaf][res])

            for i in range(n):
                r_i = reslisdic[leaf][res][i]
                ares = lipsys.select_atoms('resid ' + str(int(r_i)))
                c= TCVN(ares)[1]
                max_displacement.append(real_distance(c, cen[r_i]))

                
    fig_num += 1
    plt.figure(fig_num)            
    
    plt.hist(max_displacement, bins=np.linspace(0, 50, 100), density=False)
    plt.xlabel('Displacement distance, angstrom')
    plt.ylabel('Probability density (raw count)')
    
    fig = plt.gcf()
    fig.savefig('Displacement of residues at the first frame.png')
#     plt.show()

    lipsys.trajectory[0]


if debug_mode: check_displacement()


neighbor_list = [None] * res_total


def check_neighbor(radius):
    '''check the number of neighbor of a molecules in a specific radius of the current frame'''

    count_neighbor = []

    for leaf in range(2):

        for i in leaflet[leaf]:
            count = 0;

            for j in leaflet[leaf]:
                if (i != j and real_distance(cen[i], cen[j]) < radius): count += 1

            count_neighbor.append(count)

    
    fig_num += 1
    plt.figure(fig_num)
        
    plt.hist(count_neighbor, bins=np.linspace(0, 10, 21), density=False)
    plt.xlabel('Number of neighbor')
    plt.ylabel('Probability density (raw count)')
    
    fig = plt.gcf()
    fig.savefig('Number of neighbor of residues at the first frame.png')
    
#     plt.show()


if debug_mode: check_neighbor(10)


def initialize():
    '''initialize tilt angle, center, tilt vector, norm of tilt vector of all molecules of the current frame'''
    
    for leaf in range(2):
        
        for i in leaflet[leaf]:
            
            ares = lipsys.select_atoms('resid ' + str(i))
            til[i], cen[i], vec[i], nor[i] = TCVN(ares)


# update method : Brute-force

def brute_update_neighbor():
    '''update neighbor list of all molecules of the current frame'''
    
    print('Brutely updating neighbor list ...')
    
    for leaf in range(2):
    
        for i in leaflet[leaf]:
            neighbor_list[i] = []

            for j in leaflet[leaf]:

                if (i != j and real_distance(cen[i], cen[j]) < neighbor_radius):
                    neighbor_list[i].append(j)
                


# update method : List of neighbor list

def update_neighbor():
    '''update neighbor list of all molecules of the current frame'''
    
    print('Updating neighbor list ...')
    
    for leaf in range(2):
    
        for i in leaflet[leaf]:
            neilist = []

            for j1 in neighbor_list[i]:
                
                for j2 in neighbor_list[j1]:

                    if (i != j2 and real_distance(cen[i], cen[j2]) < neighbor_radius):
                        neilist.append(j2)
                        
                        
            neighbor_list[i] = np.unique(neilist)


all_alpha = {}

for i in range(N):
    
    for j in range(N):
        all_alpha[resname[i] + resname[j]] = []


# https://stackabuse.com/parallel-processing-in-python/

for i in range(n_frame):
    
    lipsys.trajectory[i]

    # the main loop for 1 frame!
    # method : Neighbor list
    
    print('Analyzing frame', i, '...')

    initialize()
    
    if (i == 0): brute_update_neighbor() # initialize the very first neighbor list

    if (i % update_rate == 0): update_neighbor()
    

    marked = {}

    for leaf in range(2):
        print('  leaflet', leaf, '...')

        for i in leaflet[leaf]:
            name_i = lipsys.residues.resnames[i]

            for j in neighbor_list[i]:

                if (til[j] >= 10 and til[i] >= 10): continue
                if (marked.get(str(i) + ' ' + str(j))): continue
                if (real_distance(cen[i], cen[j]) >= 10): continue

                # calculate alpha of i j
                angle = np.arccos(np.dot(vec[i], vec[j]) / (nor[i] * nor[j])) # in radian
                angle = np.rad2deg(angle) # in degree

                # save new alpha to array
                name_j = lipsys.residues.resnames[j]
                all_alpha[name_i + name_j].append(angle)

                marked[str(j) + ' ' + str(i)] = 1 # this "marked" dictionary is tricky!!!


print('Complete!')


# post-processing

gs1 = grs.GridSpec(2, 1)
gs1.update(wspace=0.025, hspace=0.025)
fig_num += 1

sum_splay = 0
phi_total = 0
sum_splay_sqr = 0

all_alpha_merge = {}

for i in range(N):
    
    for j in range(i, N):
        
        name1 = resname[i] + resname[j]
        name2 = resname[j] + resname[i]
        
        all_alpha_merge[name1] = []
        all_alpha_merge[name1].extend(all_alpha[name1])
        if (name1 != name2): all_alpha_merge[name1].extend(all_alpha[name2])
        
        phi_ij = len(all_alpha_merge[name1])
        
#         print(len(all_alpha[name1]))
#         print(len(all_alpha[name2]))
        with open(file_log, 'a') as flog:
            flog.write(resname[i] + ' ' + resname[j] + ' : ' + str(phi_ij) + ' alpha\n')
        
        plt.figure(fig_num)
        
        plt.subplot(gs1[0])
        P_alpha, bins = np.histogram(all_alpha_merge[name1], bins=np.linspace(0, 90, 91), density=True)
        alpha_new = .5 * (bins[1:] + bins[:-1])
        plt.plot(alpha_new, P_alpha, label=resname[i] + '/' + resname[j])
        
        spl = plt.subplot(gs1[1])
        spl.set_ylim([-1, 10])
        PMF = -1 * np.log(P_alpha / np.sin(np.deg2rad(alpha_new)))
        plt.plot(alpha_new, PMF, label=resname[i] + '/' + resname[j])
        
        plt.tight_layout()
        
        
        # extract data from fit range
        X_fit = []
        Y_fit = []
        
        for k, alpha in enumerate(alpha_new):
            
            if (fit_range[0] <= alpha and alpha <= fit_range[1]):
                
                if (P_alpha[k] == 0): continue
                
                X_fit.append(np.deg2rad(alpha))
                Y_fit.append(PMF[k])
                
        # quadratic fit alpha to calculate chi_12
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

        popt, pcov = curve_fit(fit_func, X_fit, Y_fit)
        perr = np.sqrt(np.diag(pcov)) # standard deviation errors
        
        plt.plot(alpha_new, fit_func(np.deg2rad(alpha_new), *popt), '--')
        
        # update splay contribution and number of encounter pair
        phi_total += phi_ij
        chi_12_ij = popt[0]
        d_chi_12_ij = perr[0]
        
        sum_splay += phi_ij / chi_12_ij
        sum_splay_sqr += phi_ij / chi_12_ij ** 2 * d_chi_12_ij
        
        
        # write data to file for further postprocessing
        # https://docs.python.org/3/library/io.html
        prefix = resname[i] + '-' + resname[j]
        subfix = '.dat'
        # file_raw writes all existing alpha
        # file_all writes alpha, P(alpha), PMF(alpha)
        file_raw = prefix + '_raw' + subfix
#         file_raw = prefix + '_' + str(fit_range[0]) + '-' + str(fit_range[1]) + subfix
        file_all = prefix + '_P_and_PMF' + subfix
        
        with open(file_raw, 'w') as filehandle:
            filehandle.write('# all existing alpha of ' + prefix + ' (in units degree)\n')
            filehandle.write('# first line: number of alpha\n%d\n' % phi_ij)
            
            for alpha in all_alpha_merge[name1]:
                filehandle.write('%f ' % alpha)
        
        
        with open(file_all, 'w') as filehandle:
            filehandle.write('# alpha(in degree) P(alpha)() PMF(alpha)(in kBT)\n')
            
            for k, alpha in enumerate(alpha_new):
                if (fit_range[0] <= alpha and alpha <= fit_range[1]):
                    filehandle.write('%f\t%f\t%f\n' % (alpha, P_alpha[k], PMF[k]))
        
        
plt.subplot(gs1[0])
plt.ylabel('P(alpha)')
plt.legend(loc='upper right')
plt.xlim([0, 90])

plt.subplot(gs1[1])
plt.ylabel('Potential of Mean force, kBT')
plt.xlabel('splay angle, alpha, degree')
plt.legend(loc='upper right')
plt.xlim([0, 90])

fig = plt.gcf()
fig.set_size_inches(12.5, 10.5)
fig.savefig('P_alpha and PMF of ' + str(n_frame) + ' frames.png')

# plt.show()


# calculate bending modulus
Km = phi_total / sum_splay
Kc = 2 * Km
d_Kc = 2 * phi_total / sum_splay ** 2 * sum_splay_sqr

with open(file_log, 'a') as flog:
    print('Bending modulus of system (in kBT Units) =', Kc, u"\u00B1", d_Kc)
    print('Bending modulus of system (in J Units)   =', Kc * kB * Temp, u"\u00B1", d_Kc * kB * Temp)

    flog.write('Bending modulus of system (in kBT Units) = %f \u00B1 %f\n' % (Kc, d_Kc))
    flog.write('Bending modulus of system (in J Units)   = %e \u00B1 %e\n' % (Kc * kB * Temp, d_Kc * kB * Temp))  
    flog.write('Finished on %s\n' %datetime.datetime.now())
    flog.write('\n')







