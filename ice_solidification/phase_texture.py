"""Padded slice atlas for trilinear sampling of a particle phase volume."""
import math
import numpy as np


def pack_phase_volume(phase):
    """Pack xyz lattice samples into a bottom-up, single-channel image."""
    nx, ny, nz = phase.shape
    columns = max(1, math.ceil(math.sqrt(nz * (ny + 2) / (nx + 2))))
    rows = math.ceil(nz / columns)
    atlas = np.zeros((rows * (ny + 2), columns * (nx + 2)), np.float32)
    for z in range(nz):
        x0, y0 = (z % columns) * (nx + 2), (z // columns) * (ny + 2)
        atlas[y0:y0 + ny + 2, x0:x0 + nx + 2] = np.pad(phase[:, :, z].T, 1, mode='edge')
    return atlas, columns


def sample_atlas(atlas, shape, columns, points):
    """CPU reference for the two bilinear shader lookups, in lattice coordinates."""
    points = np.clip(points, 0, np.asarray(shape) - 1)
    x, y, z = points.T
    z0 = np.floor(z).astype(int)
    values = []
    for layer in [z0, np.minimum(z0 + 1, shape[2] - 1)]:
        u = (layer % columns) * (shape[0] + 2) + 1 + x
        v = (layer // columns) * (shape[1] + 2) + 1 + y
        i, j = np.floor(u).astype(int), np.floor(v).astype(int)
        a, b = u - i, v - j
        values.append((1 - b) * ((1 - a) * atlas[j, i] + a * atlas[j, i + 1])
                      + b * ((1 - a) * atlas[j + 1, i] + a * atlas[j + 1, i + 1]))
    return (1 - z + z0) * values[0] + (z - z0) * values[1]
