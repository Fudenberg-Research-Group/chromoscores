import numpy as np


def get_diagonal_pileup(contact_map, boundary_list, window_size = 10):
    """
    parameters
    ----------
    contact_map: contact map (2D array)
    boundary_list: list of the boundary elements' positions on the diagonal
    window_size: size of the window (must be odd for center)

    Returns
    -------
    a stackup of snippets around the boundary elements
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

def get_offdiagonal_pileup_binlist(
    contact_map, boundary_list, binlist, window_size=10
):
    """
    parameters
    ----------
    contact_map: contact map
    boundary_list: list of the boundary elements positions on the diagonal
    binlist : exact list of bin boundaries 
    window_size: size of the window for the pileup

    Returns
    -------
    a list of pileups as numpy arrays around the feature (e.g., peaks) as a function of distance from the diagonal
    """

    bin_border_int=binlist
    bin_num=len(bin_border_int)

    pile_ups = []
    for i in range(bin_num-1):
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

def get_offdiagonal_pileup_binlist_orientation(
    contact_map, boundary_list, orientation, binlist, window_size=10
):
    """
    parameters
    ----------
    contact_map: contact map
    boundary_list: list of the boundary elements positions on the diagonal
    orientation: list of the boundary element orientations
    binlist: exact list of bins boundaries
    window_size: size of the window for the pileup

    Returns
    -------
    a list of pileups as numpy arrays around the feature (e.g., peaks) as a function of distance from the diagonal,
    orientation between barriers, and the number of snippets at each range.
    """
    bin_border_int = binlist
    bin_num = len(bin_border_int)
    
    pile_ups = []
    
    for i in range(bin_num-1):
        mat = np.zeros((window_size, window_size))
        mat_conv = np.zeros((window_size, window_size))
        mat_dive = np.zeros((window_size, window_size))
        mat_tandp = np.zeros((window_size, window_size))
        mat_tandn = np.zeros((window_size, window_size))
        
        dist = (bin_border_int[i] + bin_border_int[i + 1]) / 2
        n_conv = 0
        n_dive = 0
        n_tand_p = 0
        n_tand_n = 0
        for i_element in boundary_list:
                for j_element in boundary_list:
                    if bin_border_int[i] <= (j_element - i_element) < bin_border_int[i + 1]:
                        mat += contact_map[
                            i_element - window_size // 2 : i_element + window_size // 2,
                            j_element - window_size // 2 : j_element + window_size // 2,
                        ]
                        if orientation[np.flatnonzero(boundary_list==np.max([i_element, j_element]))] == '+':
                            if orientation[np.flatnonzero(boundary_list==np.min([i_element, j_element]))] == '-':
                                n_conv += 1 
                                mat_conv += contact_map[
                                    i_element - window_size // 2 : i_element + window_size // 2,
                                    j_element - window_size // 2 : j_element + window_size // 2,
                                ]
                            else:
                                n_tand_p +=1 
                                mat_tandp += contact_map[
                                    i_element - window_size // 2 : i_element + window_size // 2,
                                    j_element - window_size // 2 : j_element + window_size // 2,
                                ]
                        else:
                            if orientation[np.flatnonzero(boundary_list==np.min([i_element, j_element]))] == '+':
                                n_dive +=1 
                                mat_dive += contact_map[
                                    i_element - window_size // 2 : i_element + window_size // 2,
                                    j_element - window_size // 2 : j_element + window_size // 2,
                                ]
                            else:
                                n_tand_n +=1 
                                mat_tandn += contact_map[
                                    i_element - window_size // 2 : i_element + window_size // 2,
                                    j_element - window_size // 2 : j_element + window_size // 2,
                                ]
        n_tot = n_conv + n_dive + n_tand_p + n_tand_n
        pile_ups.extend([[['+-',dist,mat_conv, n_conv],['-+',dist,mat_dive, n_dive],['++',dist,mat_tandp, n_tand_p],['--',dist,mat_tandn, n_tand_n],['all',dist,mat, n_tot]]])
        
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
