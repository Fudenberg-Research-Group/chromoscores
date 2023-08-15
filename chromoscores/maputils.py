import numpy as np


def get_diagonal_pileup(contact_map, boundary_list, window_size = 10):
    """
    parameters
    ----------
    contact_map: contact map
    boundary_list: list of the boundary elements positions on the diagonal
    window_size: size of the window

    Returns
    -------
    a stackup of snippts around the boundary elements
    """

    if window_size <= 0 or window_size > len(contact_map):
        raise ValueError("window_size must be larger than 0 and smaller than the size of the contact map")
    
    mat = np.zeros((window_size, window_size))
    for i in range(len(boundary_list)):
        mat += contact_map[
            boundary_list[i] - window_size // 2 : boundary_list[i] + window_size // 2,
            boundary_list[i] - window_size // 2 : boundary_list[i] + window_size // 2,
        ]
    return mat


def get_offdiagonal_pileup(
    contact_map, boundary_list, min_dist, max_dist, bin_num = 5, window_size = 10
):
    """
    parameters
    ----------
    contact_map: contact map
    boundary_list: list of the boundary elements positions on the diagonal
    min_dist: minimum distance from the diagonal
    max_dist: maximum distance from the diagonal
    bin_num: number of bins
    window_size: size of the window for the pileup

    Returns
    -------
    a list of pileups as numpy arrays around the feature (e.g., peaks) as a function of distance from the diagonal
    """
    
    if window_size <= 0 or window_size > len(contact_map):
        raise ValueError("window_size must be larger than 0 and smaller than the size of the contact map")
    
    interval = [min_dist, max_dist]
    bin_borders = np.histogram(interval, bins=bin_num + 1)[1]
    bin_border_int = [int(x) for x in bin_borders]

    pile_ups = []
    for i in range(bin_num):
        mat = np.zeros((window_size, window_size))
        dist = (bin_border_int[i] + bin_border_int[i + 1]) / 2

        for i_element in boundary_list:
            for j_element in boundary_list:
                if bin_border_int[i] <= (j_element - i_element) < bin_border_int[i + 1]:
                    mat += contact_map[
                        i_element - window_size // 2 : i_element + window_size // 2,
                        j_element - window_size // 2 : j_element + window_size // 2,
                    ]
        pile_ups.append([dist, mat])

    return pile_ups


def get_observed_over_expected(contact_map):
    """
    parameters
    ----------
    contact_map: contact map

    Returns
    -------
    a normalized contact map based on the average of each diagonal from the main diagonal

    note: compare it with cooltools implementation. 
    """
    mat = np.zeros(np.shape(contact_map))
    for i in range(len(contact_map)):
        for j in range(len(contact_map) - i):
            mat[i, i + j] = contact_map[i, i + j] / (np.mean(np.diag(contact_map, k=j)))
            mat[i + j, i] = mat[i, i + j]
    return mat
