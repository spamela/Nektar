#!/usr/bin/env python3
import re
import sys
import numpy
import pylab
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib




# --- Main script

def main():
    # --- EQDSK filename
    if (len(sys.argv) < 3):
        print("I need filenames! Usage:")
        print("./convert_eqdsk2gmsh.py /path/to/eqdsk.file /path/to/output.geo")
        return 1
    else:
        filename_eqdsk = str(sys.argv[1])
        filename_geo   = str(sys.argv[2])

    # --- Convert to .geo for gmsh
    convert_to_gmsh(filename_eqdsk, filename_geo)








# --- Routines


def convert_to_gmsh(filename_eqdsk, filename_geo):
    # --- Get Wall and Separatrix
    n_sep, R_sep, Z_sep, n_wall, R_wall, Z_wall = get_wall_and_separatrix(filename_eqdsk)

    # --- Create a core surface simply by contracting the separatrix around magnetic axis
    rExt, zExt, rRef, rMin, zMid, R_axis, Z_axis, psi_axis, psi_bnd, bRef, Ip = get_eqdsk_parameters(filename_eqdsk)
    n_core, R_core, Z_core = create_contracted_separatrix(n_sep, R_sep, Z_sep, R_axis, Z_axis, 0.6)

    # --- Reduce surface resolutions?
    n_max = 50
    n_sep,  R_sep,  Z_sep  = reduce_surface_size(n_sep,  R_sep,  Z_sep,  n_max)
    n_core, R_core, Z_core = reduce_surface_size(n_core, R_core, Z_core, n_max)

    # --- Write .geo file for gmsh
    with open(filename_geo, "w") as file_write:
        # --- Headers with grid resolutions
        file_write.write("// --- Length scale of grid\n")
        file_write.write("lc_SOL  = 0.4;\n")
        file_write.write("lc_sep  = 0.05;\n")
        file_write.write("lc_core = 1.5;\n")
        file_write.write("\n")
        # --- Wall points
        file_write.write("// --- Wall points\n")
        for i in range(n_wall):
            file_write.write("Point(%d)  = { %+11.9e, %+11.9e, %+11.9e, lc_SOL };\n" % (i+1,R_wall[i],Z_wall[i],0.0) )
        file_write.write("\n")
        # --- Sep points
        file_write.write("// --- Separatrix points\n")
        for i in range(n_sep):
            file_write.write("Point(%d)  = { %+11.9e, %+11.9e, %+11.9e, lc_sep };\n" % (n_wall+i+1,R_sep[i],Z_sep[i],0.0) )
        file_write.write("\n")
        # --- Core points
        file_write.write("// --- Core surface points\n")
        for i in range(n_core):
            file_write.write("Point(%d)  = { %+11.9e, %+11.9e, %+11.9e, lc_core };\n" % (n_wall+n_sep+i+1,R_core[i],Z_core[i],0.0) )
        file_write.write("\n")
        # --- Wall lines
        file_write.write("// --- Straight curves between points for wall\n")
        for i in range(n_wall):
            n_beg = i+1
            n_end = (i+1)%(n_wall)+1
            file_write.write("Line(%d)  = {%d ,%d};\n" % (n_beg,n_beg,n_end) )
        file_write.write("\n")
        # --- Sep lines
        file_write.write("// --- Straight curves between points for separatrix\n")
        for i in range(n_sep):
            n_beg = n_wall + i+1
            n_end = n_wall + (n_sep+i+1)%(n_sep)+1
            file_write.write("Line(%d)  = {%d ,%d};\n" % (n_beg,n_beg,n_end) )
        file_write.write("\n")
        # --- Core lines
        file_write.write("// --- Straight curves between points for core surface\n")
        for i in range(n_core):
            n_beg = n_wall + n_sep + i+1
            n_end = n_wall + n_sep + (n_sep+n_core+i+1)%(n_core)+1
            file_write.write("Line(%d)  = {%d ,%d};\n" % (n_beg,n_beg,n_end) )
        file_write.write("\n")
        # --- Wall contour
        file_write.write("// --- Contour with all lines for wall\n")
        file_write.write("Curve Loop(1) = \n{")
        for i in range(n_wall):
            file_write.write("%d" % (i+1) )
            if (i < n_wall-1):
                file_write.write(",\n")
        file_write.write("};\n")
        file_write.write("\n")
        # --- Sep contour
        file_write.write("// --- Contour with all lines for separatrix\n")
        file_write.write("Curve Loop(2) = \n{")
        for i in range(n_sep):
            file_write.write("%d" % (n_wall+i+1) )
            if (i < n_sep-1):
                file_write.write(",\n")
        file_write.write("};\n")
        file_write.write("\n")
        # --- Core contour
        file_write.write("// --- Contour with all lines for core surface\n")
        file_write.write("Curve Loop(3) = \n{")
        for i in range(n_core):
            file_write.write("%d" % (n_wall+n_sep+i+1) )
            if (i < n_core-1):
                file_write.write(",\n")
        file_write.write("};\n")
        file_write.write("\n")
        # --- Construct grid
        file_write.write("// --- 2D surface between wall and separatrix\n")
        file_write.write("Plane Surface(1) = {1,2};\n")
        file_write.write("// --- 2D surface between separatrix and middle surface\n")
        file_write.write("Plane Surface(2) = {2,3};\n")
        file_write.write("// --- 2D surface inside middle surface\n")
        file_write.write("Plane Surface(3) = {3};\n")




def get_eqdsk_dim(filename):
    file_read = open(filename, 'r')
    with open(filename, 'r') as file_read:
        headers = file_read.readline()
    array_tmp = headers.split()
    nR = int(array_tmp[len(array_tmp)-2])
    nZ = int(array_tmp[len(array_tmp)-1])
    return nR, nZ


def reduce_surface_size(n_surf, R_surf, Z_surf, n_max):
    # we just reduce surface resolution by two until < n_max
    n_old = n_surf
    R_old = R_surf
    Z_old = Z_surf
    n_surf_reduced = n_max + 1
    while (n_surf_reduced > n_max):
        count = 0
        R_surf_new = []
        Z_surf_new = []
        for i in range(n_old):
            if (i%2 == 0):
                count = count + 1
                R_surf_new.append(R_old[i])
                Z_surf_new.append(Z_old[i])
        n_surf_reduced = count
        n_old = count
        R_old = R_surf_new
        Z_old = Z_surf_new
    return n_surf_reduced, R_surf_new, Z_surf_new




def get_eqdsk_parameters(filename):
    # --- Counters
    count_lines       = 0
    # --- Initialise
    rExt     = 0.0
    zExt     = 0.0
    rRef     = 0.0
    rMin     = 0.0
    zMid     = 0.0
    R_axis   = 0.0
    Z_axis   = 0.0
    psi_axis = 0.0
    psi_bnd  = 0.0
    bRef     = 0.0
    Ip       = 0.0
    # --- Read file
    file_read = open(filename, 'r')
    for line in file_read:
        count_lines = count_lines + 1
        # 1 line of headers + 4 lines of parameters
        if (count_lines > 1):
            # edsk lines are made of upto 5 floats in format +0.123456789e+01, ie. chunks of 16 characters
            array_tmp = chunkstring(line, 16)
            if (count_lines == 2):
                rExt = array_tmp[0]
                zExt = array_tmp[1]
                rRef = array_tmp[2]
                rMin = array_tmp[3]
                zMid = array_tmp[4]
                continue
            if (count_lines == 3):
                R_axis   = array_tmp[0]
                Z_axis   = array_tmp[1]
                psi_axis = array_tmp[2]
                psi_bnd  = array_tmp[3]
                bRef     = array_tmp[4]
                continue
            if (count_lines == 4):
                Ip       = array_tmp[0]
                break
    file_read.close()
    return rExt, zExt, rRef, rMin, zMid, R_axis, Z_axis, psi_axis, psi_bnd, bRef, Ip



def create_contracted_separatrix(n_sep, R_sep, Z_sep, R_axis, Z_axis, contract):
    R_surf = []
    Z_surf = []
    for i in range(len(R_sep)):
        R_surf.append(R_axis + (R_sep[i]-R_axis) * contract)
        Z_surf.append(Z_axis + (Z_sep[i]-Z_axis) * contract)
    n_surf = n_sep
    return n_surf, R_surf, Z_surf 


def get_wall_and_separatrix(filename):
    # --- Get dimensions
    nR, nZ = get_eqdsk_dim(filename)
    # --- Counters
    count_lines       = 0
    count_profiles    = 0
    count_prof_length = 0
    count_psi_length  = 0
    finished_psi      = False
    n_sep             = 0
    n_wall            = 0
    count_sep_wall    = 0
    RZ_sep_wall       = []
    # --- Read file
    file_read = open(filename, 'r')
    for line in file_read:
        count_lines = count_lines + 1
        # 1 line of headers + 4 lines of parameters
        if (count_lines > 5):
            # --- Count 4 profiles after headers: F-profile, P-profile, FFprime-profile, Pprime-profile
            if (   ((count_profiles < 4) and (not finished_psi)) \
                or ((count_profiles < 5) and (finished_psi)    ) ):
                # edsk lines are made of upto 5 floats in format +0.123456789e+01, ie. chunks of 16 characters
                array_tmp = chunkstring(line, 16)
                count_prof_length = count_prof_length + len(array_tmp)
                if (count_prof_length == nR):
                    count_prof_length = 0
                    count_profiles = count_profiles + 1
                    continue
            # --- Count 2D psi-map: nR*nZ floats
            if ( (count_profiles == 4) and (not finished_psi) ):
                # edsk lines are made of upto 5 floats in format +0.123456789e+01, ie. chunks of 16 characters
                array_tmp = chunkstring(line, 16)
                count_psi_length = count_psi_length + len(array_tmp)
                if (count_psi_length == nR*nZ):
                    finished_psi = True
                    continue
            # --- Read wall and separatrix dimensions
            if ( (n_sep == 0) and (n_wall == 0) and (count_profiles == 5) ):
                array_tmp = line.split()
                n_sep  = int(array_tmp[0])
                n_wall = int(array_tmp[1])
                continue
            # --- Read wall and separatrix
            if ( (n_sep != 0) and (n_wall != 0) ):
                # edsk lines are made of upto 5 floats in format +0.123456789e+01, ie. chunks of 16 characters
                array_tmp = chunkstring(line, 16)
                for i in range(len(array_tmp)):
                    RZ_sep_wall.append(array_tmp[i])
                count_sep_wall = count_sep_wall + len(array_tmp)
                if (count_sep_wall >= 2*n_sep + 2*n_wall):
                    break
    file_read.close()
    R_sep = []
    Z_sep = []
    for i in range(n_sep):
        R_sep.append(RZ_sep_wall[2*i  ])
        Z_sep.append(RZ_sep_wall[2*i+1])
    R_wall = []
    Z_wall = []
    for i in range(n_wall):
        R_wall.append(RZ_sep_wall[2*n_sep + 2*i  ])
        Z_wall.append(RZ_sep_wall[2*n_sep + 2*i+1])
    # --- Sanity check, we don't want the last point to be the same as the first!
    if ( (R_sep[0] == R_sep[len(R_sep)-1]) and (Z_sep[0] == Z_sep[len(Z_sep)-1]) ):
        n_sep = n_sep - 1
        R_sep.pop()
        Z_sep.pop()
    if ( (R_wall[0] == R_wall[len(R_wall)-1]) and (Z_wall[0] == Z_wall[len(Z_wall)-1]) ):
        n_wall = n_wall - 1
        R_wall.pop()
        Z_wall.pop()
    # --- Sanity check, you never know if EQDSK does cm or m...
    rExt, zExt, rRef, rMin, zMid, R_axis, Z_axis, psi_axis, psi_bnd, bRef, Ip = get_eqdsk_parameters(filename)
    if (min(R_sep) > rMin+rExt):
        for i in range(n_sep):
            R_sep[i] = R_sep[i] * 0.01
            Z_sep[i] = Z_sep[i] * 0.01
    if (max(R_sep) < rMin):
        for i in range(n_sep):
            R_sep[i] = R_sep[i] * 100.0
            Z_sep[i] = Z_sep[i] * 100.0
    if (min(R_wall) > rMin+rExt):
        for i in range(n_wall):
            R_wall[i] = R_wall[i] * 0.01
            Z_wall[i] = Z_wall[i] * 0.01
    if (max(R_wall) < rMin):
        for i in range(n_wall):
            R_wall[i] = R_wall[i] * 100.0
            Z_wall[i] = Z_wall[i] * 100.0
    return n_sep, R_sep, Z_sep, n_wall, R_wall, Z_wall




def chunkstring(string, length):
    array_tmp = re.findall('.{%d}' % length, string)
    for i in range(len(array_tmp)):
        array_tmp[i] = float(array_tmp[i])
    return array_tmp






# --- Run as executable directly
main()

