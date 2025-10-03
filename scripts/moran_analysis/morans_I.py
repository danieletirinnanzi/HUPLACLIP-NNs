import numpy as np
from numba import jit

# MORAN'S I CALCULATION (written on 30/07/2025 @ SISSA)
# Implementation progression described in "moransI_implementation_progression.ipynb£
@jit(nopython=True)
def morans_I_numba(adj_matrix, max_radius):
    """
    Calculates Moran's I spatial autocorrelation statistic for a square adjacency matrix
    using rook's case neighbors up to a given radius.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Square (N x N) matrix representing an input graph.
    max_radius : int
        Maximum radius to consider for neighbor relationships.

    Returns
    -------
    I : np.ndarray
        Array of Moran's I values for each radius from 0 to max_radius.

    Notes
    -----
    - Uses rook's case (up, down, left, right) neighbors at each radius.
    - I is set to 0 when radius is 0 (I[0]=0).
    - Optimized with numba for performance.
    """    
    N = adj_matrix.shape[0]
    x = adj_matrix
    x_mean = np.mean(x)

    cov = np.zeros(max_radius+1)
    w = np.zeros(max_radius+1, dtype='int')

    x = x - x_mean
    var = np.sum(x**2)


    for dr in range(1,max_radius+1):
        for i in range(N):
            for j in range(N):
                # Loop over the four rook directions at distance "dr"
                for d in range(4):
                    if d == 0:
                        di, dj = -dr, 0 # Up
                    elif d == 1:
                        di, dj = dr, 0  # Down
                    elif d == 2:
                        di, dj = 0, -dr # Left
                    else:
                        di, dj = 0, dr  # Right
                    # Compute exact coordinates of neighoring cell
                    ni, nj = i+di, j+dj

                    # Check if current neighbor is within matrix bounds
                    if 0 <= ni < N and 0 <= nj < N:
                        # Update covariance and increase neighbor count
                        cov[dr] += x[i,j] * x[ni,nj]
                        w[dr] += 1
        if dr > 1:
            cov[dr] += cov[dr-1]
            w[dr] += w[dr-1]

    # Normalization
    I = (N*N / w) * (cov / var)
    return I