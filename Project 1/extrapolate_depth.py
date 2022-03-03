import scipy as sp
import numpy as np
from scipy.interpolate import griddata
from matplotlib import path
import matplotlib.pyplot as plt


def readData(depth, colx, coly):
    contour = np.genfromtxt(
        "Nechelik_Data.csv",
        delimiter=",",
        skip_header=2,
        missing_values="",
        usecols=(colx, coly),
    )
    contour = contour[~np.isnan(contour)].reshape(-1, 2)
    if depth != 0:
        contour = np.append(contour, np.full(contour.shape, depth), axis=1)
    return contour


def extrapolate_depth(filename, Nx, Ny):

    boundary = readData(0, 0, 1)
    c_7400 = readData(7400, 2, 3)
    c_7200 = readData(7200, 4, 5)
    c_7000 = readData(7000, 6, 7)
    c_6800 = readData(6800, 8, 9)
    c_6600 = readData(6600, 10, 11)
    contours = np.concatenate((c_7400, c_7200, c_7000, c_6800, c_6600), axis=0)
    # print(contours,"\n")

    # create meshgrid
    min_x = 0
    max_x = np.amax(boundary[:, 0])
    max_y = np.amax(boundary[:, 1])
    min_y = np.amin(boundary[:, 1])
    xv = np.linspace(min_x, max_x, num=Nx)
    yv = np.linspace(min_y, max_y, num=Ny)
    gridx, gridy = np.meshgrid(xv, yv)
    mesh = (gridx, gridy)
    # print(mesh[0], mesh[1])

    # calculate interpolated depth value
    depth = sp.interpolate.griddata(
        contours[:, [0, 1]], contours[:, 2], mesh, method="cubic"
    )
    # print(depth)
    gridx = mesh[0].reshape(-1, 1)
    gridy = mesh[1].reshape(-1, 1)
    depth = depth.reshape(-1, 1)
    point_depth = np.concatenate((gridx, gridy, depth), axis=1).reshape(-1, 3)
    point_depth = point_depth[~np.isnan(point_depth).any(axis=1)]
    # print(point_depth)

    depth2 = sp.interpolate.griddata(
        point_depth[:, [0, 1]], point_depth[:, 2], mesh, method="nearest"
    )
    # print(depth2)

    shape = path.Path(boundary)
    # print(shape)

    for i in range(0, depth2.shape[0]):
        for j in range(0, depth2.shape[1]):
            if not shape.contains_point((xv[j], yv[i])):
                depth2[i, j] = float("NaN")

    return depth2
