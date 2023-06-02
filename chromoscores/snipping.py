import numpy as np



def get_diagonal_pile_up(contact_map, boundary_list, window_size):
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
    mat = np.zeros((window_size, window_size))
    for i in range(len(boundary_list)):
        mat += contact_map[
            boundary_list[i] - window_size // 2 : boundary_list[i] + window_size // 2,
            boundary_list[i] - window_size // 2 : boundary_list[i] + window_size // 2,
        ]
    return mat

def get_expected_map(contact_map):
    """
    parameters
    ----------
    contact_map: contact map

    Returns
    -------
    a normalized contact map based on the average of each diagonal from the main diagonal
    """
    mat = np.zeros(np.shape(contact_map))
    for i in range(len(contact_map)):
        for j in range(len(contact_map) - i):
            mat[i, i + j] = contact_map[i, i + j] / (
                np.mean(np.diag(contact_map, k=j))
            )
            mat[i + j, i] = mat[i, i + j]
    return mat

def peak_snipping(contact_map, window_size, peak_coordinate):
    """
    parameters
    ----------
    contact_map: contact map
    window_size: size of the window
    peak_coordinate: coordinate of the peak in (i,j) format

    returns
    -------
    a snippet of the contact map around the specified peak
    """
    if (peak_coordinate[1] + window_size) > len(contact_map) or (
        peak_coordinate[0] - window_size
    ) < 0:
        raise ValueError(
            "window window_size for peak coordinate exceeds window_size of the contact map"
        )
    
    snippet = contact_map[
        (peak_coordinate[0] - window_size) : (peak_coordinate[0] + window_size),
        (peak_coordinate[1] - window_size) : (peak_coordinate[1] + window_size),
    ]
    return snippet


def get_isolation_areas(
    contact_map, delta=1, diag_offset=3, max_distance=10, state=1
):
    """
    parameters
    ----------
    contact_map: snippet of a contact map around a boundary element
    delta: distance from the border between in_tad and out_tad
    diag_offset: distance from the diagonal. This also determines the size of the snippet.
    max_distance: maximum distance from the diagonal
    state: 1 for triangle snippets, 0 for square snippets

    returns
    -------
    areas with a size of diag_offset inside and outside a tad
    """
    csize = len(contact_map) // 2
    window_size = 4 * (diag_offset + delta) + 1
    pile_center = contact_map[
        csize - window_size // 2 : csize + window_size // 2 + 1,
        csize - window_size // 2 : csize + window_size // 2 + 1,
    ]

    out_tad = np.zeros(np.shape(pile_center))
    mask_out = np.zeros(np.shape(pile_center), dtype=bool)
    mask_out[
        window_size // 2 - diag_offset : window_size // 2,
        window_size // 2 + 1 : window_size // 2 + diag_offset + 1,
    ] = True
    out_tad[mask_out] = pile_center[mask_out]
    out_tad = np.tril(np.triu(out_tad, state * (diag_offset + 1)), max_distance)

    in_tad = np.zeros(np.shape(pile_center))
    mask = np.zeros(np.shape(pile_center), dtype=bool)
    mask[
        delta : delta + diag_offset,
        diag_offset + delta + 1 : 2 * diag_offset + delta + 1,
    ] = True
    in_tad[mask] = pile_center[mask]

    mask = np.zeros(np.shape(pile_center), dtype=bool)
    mask[
        window_size // 2 + delta : window_size // 2 + diag_offset + delta,
        window_size // 2
        + diag_offset
        + delta
        + 1 : window_size // 2
        + 2 * diag_offset
        + delta
        + 1,
    ] = True
    in_tad[mask] = pile_center[mask]
    in_tad = np.tril(np.triu(in_tad, state * (diag_offset + 1)), max_distance)

    return in_tad, out_tad, pile_center


def tad_snipping(contact_map, boundary_list, index):
    """
    parameters
    ----------
    contact_map: contact map
    boundary_list: list of the boundary elements positions on the diagonal.
    index: index of the boundary element in the boundary_list. This should be in the range of boundary_list.

    returns
    -------
    a snippet of the contact map around the specified boundary element
    """

    if index + 1 > len(boundary_list):
        raise ValueError("index + 1 should be in the range of boundary list")

    tads_snippet = contact_map[
        boundary_list[index] : boundary_list[index + 1] + 1,
        boundary_list[index] : boundary_list[index + 1] + 1,
    ]
    return tads_snippet


def tad_snippet_sectors(
    contact_map, boundary_list, index, delta, diag_offset, max_distance
):
    """
    parameters
    ----------
    contact_map: snippet of a contact map around a boundary element
    delta: distance from the border between in_tad and out_tad. is defined to exclude
           flames when extracting in_tad and out_tad areas.
    diag_offset: distance from the diagonal. This also determines the size of the snippet.
    max_distance: maximum distance from the diagonal
    state: 1 for triangle snippets, 0 for square snippets

    returns
    -------
    areas with a size of diag_offset inside and outside a tad
    """
    tad = tad_snipping(contact_map, boundary_list, index)
    tad_window_size = len(tad)

    pile_center = contact_map[
        boundary_list[index] : boundary_list[index + 2] + 1,
        boundary_list[index] : boundary_list[index + 2] + 1,
    ]

    if index + 1 > len(boundary_list):
        raise ValueError("index + 1 should be in the range of boundary list")

    if max_distance > len(pile_center) // 2:
        raise ValueError("max distance exceeds tad snippet window_size")
    

    out_tad = np.zeros(np.shape(pile_center))
    out_tad[delta : tad_window_size - delta, tad_window_size + delta : -delta] = 1
    out_tad = np.tril(np.triu(out_tad, diag_offset), max_distance) > 0

    in_tad = np.zeros(np.shape(pile_center))
    in_tad[delta : tad_window_size - delta, delta : tad_window_size - delta] = 1
    in_tad[tad_window_size + delta : -delta, tad_window_size + delta : -delta] = 1
    in_tad = np.tril(np.triu(in_tad, diag_offset), max_distance) > 0

    return in_tad, out_tad, pile_center



def flame_snipping_vertical(contact_map, boundary_list, index, width, edge):
    """
    parameters
    ----------
    contact_map: contact map
    boundary_list: list of the boundary elements positions on the diagonal.
    index: index of the boundary element in the boundary_list. This should be in the range of boundary_list.
    width: width of the flame
    edge: excluded areas at the ends of the flame
    
    returns
    -------
    a snippet of the contact map around the specified flame
    """
    snippet = contact_map[
        (boundary_list[index] + edge) : (boundary_list[index + 1] - edge),
        (boundary_list[index + 1] - width) : (boundary_list[index + 1] + width),
    ]
    return snippet


def flame_snipping_horizontal(contact_map, boundary_list, index, width, edge):
    """
    parameters
    ----------
    contact_map: contact map
    boundary_list: list of the boundary elements positions on the diagonal.
    index: index of the boundary element in the boundary_list. This should be in the range of boundary_list.
    width: width of the flame
    edge: excluded areas at the ends of the flame
    
    returns
    -------
    a snippet of the contact map around the specified flame
    """
    snippet = contact_map[
        (boundary_list[index] - width) : (boundary_list[index] + width),
        (boundary_list[index] + edge) : (boundary_list[index + 1] - edge),
    ]
    return snippet
