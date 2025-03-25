#!/bin/env python3
import argparse
from datetime import datetime
import os
import pathlib
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev

from lanelet_xml import LaneletMap
import numpy as np
import math
import matplotlib.pyplot as plt

def distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

def smooth_3d_curve(points, poly_order, window_size):
    points = np.asarray(points)
    # print(points.shape)
    tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]])
    new_u = np.linspace(0, 1, len(points))
    smoothed = np.array(splev(new_u, tck)).T
    
    if window_size > 1 and poly_order < window_size:
        smoothed = np.array([savgol_filter(smoothed[:, i], window_size, poly_order) for i in range(3)]).T
    return smoothed

def fem_smooth_slide_window(traj_array, window_size):

    return 0

def generate(width, mgrs, interval, offset, use_centerline=False):

    left = np.load("/media/gjd/TOSHIBA EXT/bag2lanelet/scripts/left_nodes.npy")
    right = np.load("/media/gjd/TOSHIBA EXT/bag2lanelet/scripts/right_nodes.npy")

    # left = smooth_3d_curve(left, 10, 25)
    # right = smooth_3d_curve(right, 10, 25)

    plt.scatter(left[0],left[1])
    plt.scatter(right[0],right[1])

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
        "/media/gjd/TOSHIBA EXT/bag2lanelet/scripts/line2lanelet.osm"
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