#!/bin/env python3
import argparse
from datetime import datetime
import os
import pathlib
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from scipy import sparse

from lanelet_xml import LaneletMap
import numpy as np
import math
import matplotlib.pyplot as plt
import osqp

prob = osqp.OSQP()

def distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

def smooth_3d_curve(points, poly_order=2, window_size=10):
    points = np.asarray(points)
    # print(points.shape)
    tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]])
    new_u = np.linspace(0, 1, len(points))
    smoothed = np.array(splev(new_u, tck)).T
    
    if window_size > 1 and poly_order < window_size:
        smoothed = np.array([savgol_filter(smoothed[:, i], window_size, poly_order) for i in range(3)]).T
    return smoothed

def setup_fem_smooth_solver(points, w1, w2, w3, constraint_l, constraint_u):
    global prob
    # plt.scatter(points[:,0],points[:,1],color = 'b', s = 5)
    mat_len = 2 * len(points)
    P = np.zeros((mat_len,mat_len))
    q = np.zeros((mat_len))
    A = np.zeros((mat_len,mat_len))
    l = np.zeros((mat_len))
    u = np.zeros((mat_len))
    # construct P 上三角矩阵
    P[0,0] = w1 + w2 + w3
    P[1,1] = w1 + w2 + w3
    P[0,2] = -2*w1 - w2
    P[1,3] = -2*w1 - w2
    P[2,2] = 5*w1 + 2*w2 + w3
    P[3,3] = 5*w1 + 2*w2 + w3
    for i in range(2,len(points)-2):
        P[2*i-4,2*i] = w1
        P[2*i-3,2*i+1] = w1
        P[2*i-2,2*i] = -4*w1 - w2
        P[2*i-1,2*i+1] = -4*w1 - w2
        P[2*i-0,2*i] = 6*w1 + 2*w2 + w3
        P[2*i+1,2*i+1] = 6*w1 + 2*w2 + w3
    P[mat_len-8,mat_len-4] = w1
    P[mat_len-7,mat_len-3] = w1
    P[mat_len-6,mat_len-4] = -4*w1 - w2
    P[mat_len-5,mat_len-3] = -4*w1 - w2
    P[mat_len-4,mat_len-4] = 5*w1 + 2*w2 + w3
    P[mat_len-3,mat_len-3] = 5*w1 + 2*w2 + w3
    P[mat_len-6,mat_len-2] = w1
    P[mat_len-5,mat_len-1] = w1
    P[mat_len-4,mat_len-2] = -2*w1 - w2
    P[mat_len-3,mat_len-1] = -2*w1 - w2
    P[mat_len-2,mat_len-2] = w1 + w2 + w3
    P[mat_len-1,mat_len-1] = w1 + w2 + w3

    # construct q
    for i in range(len(points)):
        q[2*i] = -1 * w3 * points[i, 0]
        q[2*i+1] = -1 * w3 * points[i, 1]
    
    # construct A
    for i in range(mat_len):
        A[i,i] = 1

    # construct l,u
    for i in range(len(points)):
        l[2*i] = points[i, 0] - constraint_l
        l[2*i+1] = points[i, 1] - constraint_l 
        u[2*i] = points[i, 0] + constraint_u
        u[2*i+1] = points[i, 1] + constraint_u 

    # constuct sparse matrix
 
    P = sparse.csr_matrix(P)
    A = sparse.csr_matrix(A)

    prob.setup(P, q, A, l, u, rho = 1e-2)
    # Solve problem
    res = prob.solve()
    
    # result output
    for i in range(len(points)):
        points[i,0] = res.x[2*i]
        points[i,1] = res.x[2*i+1]
    # plt.scatter(points[:,0],points[:,1],color = 'r', s = 1)
    # plt.show()
    
    return points

def update_fem_smooth_solver(points, w1, w2, w3, constraint_l, constraint_u):
    global prob
    plt.scatter(points[:,0],points[:,1],color = 'b', s = 5)
    mat_len = 2 * len(points)
    q = np.zeros((mat_len))  
    l = np.zeros((mat_len))
    u = np.zeros((mat_len))

    # construct q
    for i in range(len(points)):
        q[2*i] = -1 * w3 * points[i, 0]
        q[2*i+1] = -1 * w3 * points[i, 1]

    # construct l,u
    for i in range(len(points)):
        l[2*i] = points[i, 0] - constraint_l
        l[2*i+1] = points[i, 1] - constraint_l 
        u[2*i] = points[i, 0] + constraint_u
        u[2*i+1] = points[i, 1] + constraint_u 

    prob.update(q = q, l = l, u = u)
    # Solve problem
    res = prob.solve()
    
    # result output
    for i in range(len(points)):
        points[i,0] = res.x[2*i]
        points[i,1] = res.x[2*i+1]

    plt.scatter(points[:,0],points[:,1],color = 'r', s = 1)
    plt.show()
    return points

# params:
# w1: the smooth cost weight, w2: the length cost weight, w3: the drift cost weight
# window_size: the slide window length
# interval: the interval from this window to next window
# constraint_l < xi - xi_ref < constraint_r
# considering adding curvature constraint to this optimizer
def fem_smooth_slide_window(traj_array, window_size, interval, w1, w2, w3, constraint_l, constraint_u):
    pt_total_amount = len(traj_array)
    modified_array = traj_array
    for i in range(0, pt_total_amount - window_size + 1, interval):
        pt_optimize = traj_array[i:i+window_size, :]
        if i == 0:
            pt_optimize = setup_fem_smooth_solver(pt_optimize,w1,w2,w3,constraint_l,constraint_u)
        else:
            pt_optimize = update_fem_smooth_solver(pt_optimize,w1,w2,w3,constraint_l,constraint_u)
        modified_array[i:i+window_size, :] = pt_optimize
    return modified_array

def generate(width, mgrs, interval, offset, use_centerline=False):

    left = np.load("/home/gjd/autoware/src/tools/bag2lanelet/scripts/left_nodes.npy")
    right = np.load("/home/gjd/autoware/src/tools/bag2lanelet/scripts/right_nodes.npy")

    plt.scatter(left[:,0],left[:,1],color = 'b', s = 5)
    origin_vis = plt.scatter(right[:,0],right[:,1],color = 'b', s = 5)
    # smooth process

    # left = smooth_3d_curve(left, 2, 10)
    # right = smooth_3d_curve(right, 2, 10)
    # left = fem_smooth_slide_window(left,200,6,1,1,1,0.3,0.3)
    # right = fem_smooth_slide_window(right,30,6,1,1,1,0.3,0.3)

    left = setup_fem_smooth_solver(left,5,10,1,0.5,0.5)
    right = setup_fem_smooth_solver(right,5,10,1,0.5,0.5)

    plt.scatter(left[:,0],left[:,1],color = 'r', s = 5)
    smooth_vis = plt.scatter(right[:,0],right[:,1],color = 'r', s = 5)
    plt.legend(handles=[origin_vis,smooth_vis],labels=['original curve','smooth curve'],loc='best')
    plt.show()

    if(len(left)>len(right)):
        left = left[:len(right),:]

    center = (left+right)/2
    print(left.shape,right.shape,center.shape)
    m = LaneletMap(mgrs=mgrs)
    left_nodes = [m.add_node(*node) for node in left]
    right_nodes = [m.add_node(*node) for node in right]
    center_nodes = [m.add_node(*node) for node in center]

    left_line = m.add_way(left_nodes)
    right_line = m.add_way(right_nodes)
    center_line = m.add_way(center_nodes) if use_centerline else None

    m.add_relation(left_line, right_line, center_line)
    m.save(
        "/home/gjd/autoware/src/tools/bag2lanelet/scripts/line2lanelet.osm"
    )


def main():
    parser = argparse.ArgumentParser(description="Create lanelet2 file from rosbag2")
    parser.add_argument("-l", "--width", type=float, default=2.0, help="lane width[m]")
    parser.add_argument("-m", "--mgrs", default="54SUE", help="MGRS code")
    parser.add_argument(
        "--interval",
        type=float,
        nargs=2,
        default=[0.1, 2.0],
        help="min and max interval between tf position",
    )
    parser.add_argument(
        "--offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="offset[m] from base_link"
    )
    parser.add_argument("--center", action="store_true", help="add centerline to lanelet")
    args = parser.parse_args()   
    generate(
        args.width, args.mgrs, args.interval, args.offset, args.center
    )


if __name__ == "__main__":
    main()