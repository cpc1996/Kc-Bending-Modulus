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
    Last update:  04/04/2021
'''

print(info)


import sys
sys.path.append('/home/cao/anaconda3/envs/hus/lib/python3.8/site-packages')


import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt 

from scipy.optimize import curve_fit


import MDAnalysis as mda
from MDAnalysis import transformations as trans


import warnings


# debug flag
debug_mode = False


# BE CAREFUL OF THESE LINES!!!
np.seterr(divide = 'ignore')

warnings.filterwarnings("ignore",category=UserWarning)


# read the data from tpr and xtc
# https://www.mdanalysis.org/MDAnalysisTutorial/basics.html

input_tpr = 'data/step7_new.tpr'
input_xtc = 'data/d1_step7_mem_500.xtc'

lipsys = mda.Universe(input_tpr, input_xtc)
print(lipsys)


# define tilt vector for molecule types: point to the outer of membrane
# this dictionary must be updated before every new calculation!!!
resvedict = {
    'DPPC': ['C214 C215 C216 C314 C315 C316', 'C2  P'],
    'POPC': ['C216 C217 C218 C314 C315 C316', 'C2  P'],
    'PSM' : ['C14F C15F C16F C16S C17S C18S', 'C2S P'],
    'CHL1': ['C17',                           'C3'],
}

# define ratio threshold if a molecule type can be neglected
threshold_ratio = 0.03     # 3%

# define direction of membrane
Oaxis = np.array([0, 0, 1])     # z-axis


# number of frame to calculate bending modulus!!!
n_frame = lipsys.trajectory.n_frames
#n_frame = 10


# molecules in range radius are counted as neighbors
neighbor_radius = 20

# neighbor_list is updated every update_rate frames
update_rate = 10 


# only alpha in a specific range are fitted
fit_range = [5, 20] # in degree



#================================================================================================================


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
        
        print(i)

        fig_num += 1
        plt.figure(fig_num)
        
        plt.subplot(1, 2, 1)
        plt.hist(tmp1, bins=np.linspace(0, 90, 50), density=True)
        
        plt.subplot(1, 2, 2)
        plt.hist(tmp2, bins=np.linspace(90, 180, 50), density=True)
        
        plt.tight_layout()
        
        plt.show()


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
            

            plt.subplot(2, 1, 1)
            P_alpha, bins = np.histogram(alpha, bins=np.linspace(0, 90, 90), density=True)
            alpha_new = .5 * (bins[1:] + bins[:-1])
            plt.plot(alpha_new, P_alpha)

            plt.subplot(2, 1, 2)
            PMF = -1 * np.log(P_alpha / np.sin(np.deg2rad(alpha_new)))
            plt.plot(alpha_new, PMF)
            
            plt.tight_layout()


    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.show()


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
                ares = lipsys.select_atoms('resid ' + str(r_i))
                c= TCVN(ares)[1]
                max_displacement.append(real_distance(c, cen[r_i]))

                
    plt.hist(max_displacement, bins=np.linspace(0, 50, 100), density=True)
    
    plt.show()

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

            
    plt.hist(count_neighbor, bins=np.linspace(0, 70, 100), density=True)
    
    plt.show()


if debug_mode: check_neighbor(20)


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
        print(resname[i], resname[j], '\t:', phi_ij, 'alpha')
        
        plt.figure(fig_num + 1)
        
        plt.subplot(211)
        P_alpha, bins = np.histogram(all_alpha_merge[name1], bins=np.linspace(0, 90, 91), density=True)
        alpha_new = .5 * (bins[1:] + bins[:-1])
        plt.plot(alpha_new, P_alpha)
        
        spl = plt.subplot(212)
        spl.set_ylim([-1, 10])
        PMF = -1 * np.log(P_alpha / np.sin(np.deg2rad(alpha_new)))
        plt.plot(alpha_new, PMF)
        
        plt.tight_layout()
        
        
        # extract data from fit range
        X_fit = []
        Y_fit = []
        
        for k, alpha in enumerate(alpha_new):
            
            if (fit_range[0] <= alpha and alpha <= fit_range[1]):
                
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
        
#         print(phi_ij, chi_12_ij)
        

fig = plt.gcf()
fig.set_size_inches(12.5, 10.5)

plt.show()
fig_num += 1


# calculate bending modulus

# print(phi_total)
Km = phi_total / sum_splay
Kc = 2 * Km
d_Kc = 2 * phi_total / sum_splay ** 2 * sum_splay_sqr

print('Bending modulus of system (in kBT Units) =', Kc, u"\u00B1", d_Kc)




