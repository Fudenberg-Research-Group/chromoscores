import numpy as np



def get_diagonal_pile_up(contact_map, BE_list, window_size):
    """
    contact_map: contact map
    stall_list: list of the boundary elements positions on the diagonal to be piled up
    window_size: size of the window
    """
    mat = np.zeros((window_size, window_size))
    for i in range(len(BE_list)):
        mat += contact_map[
            BE_list[i] - window_size // 2 : BE_list[i] + window_size // 2,
            BE_list[i] - window_size // 2 : BE_list[i] + window_size // 2,
        ]
    return mat

def get_expected_map(contact_map):
    """
    contact_map: contact map
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
    contact_map: contact map
    window_size: size of the window
    peak_coordinate: coordinate of the peak in (i,j) format
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


def get_isolation_snippets(
    contact_map, delta=1, diag_offset=3, max_distance=10, state=1
):
    """
    contact_map: contact map
    delta: distance from the border between in_tad and out_tad
    diag_offset: distance from the diagonal. This also determines the size of the snippet.
    max_distance: maximum distance from the diagonal
    state: 1 for triangle snippets, 0 for square snippets
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


def tad_snipping(contact_map, stall_list, index):
    """
    contact_map: contact map
    stall_list: list of the boundary elements positions on the diagonal.
    index: index of the boundary element in the stall_list. This should be in the range of stall_list.
    """

    if index + 1 > len(stall_list):
        raise ValueError("index + 1 should be in the range of stall list")

    tad = contact_map[
        stall_list[index] : stall_list[index + 1] + 1,
        stall_list[index] : stall_list[index + 1] + 1,
    ]
    return tad


def tad_snippet_sectors(
    contact_map, stall_list, index, delta, diag_offset, max_distance
):
    """
    Tad snippet sectors
    setting area in_tad vs area out_tad. "Delta" is defined to exclude
    flames when extracting in_tad and out_tad areas.
    --------------------------------------
    Function tad_snippet_sectors(contact_map, stall_list, index, delta,
    diag_offset, max_distance):

         begin function

         getting tad_matrix from tad_snippet function

         set adjacent_tads starting with stall_index:
             pile_center = contact_map[
             stall_list[index]:stall_list[index + 2] + 1,
             stall_list[index]:stall_list[index + 2] + 1,
             ]

         raise an error if max_distance is larger than snippet

         set in_tad and out_tad areas:
             out_tad = np.zeros(np.shape(pile_center))
             out_tad[delta:tad_window_size - delta, tad_window_size + delta:-delta] = 1
             out_tad = np.tril(np.triu(out_tad, diag_offset),
              max_distance) > 0

             in_tad = np.zeros(np.shape(pile_center))
             in_tad[delta:tad_window_size - delta, delta:tad_window_size - delta] = 1
             in_tad[tad_window_size + delta:-delta, tad_window_size + delta:-delta] = 1
             in_tad = np.tril(np.triu(in_tad, diag_offset), max_distance) > 0

    return in_tad, out_tad, adjacent matrices

    end function
    --------------------------------------
    """
    tad = tad_snippet(contact_map, stall_list, index)
    tad_window_size = len(tad)

    pile_center = contact_map[
        stall_list[index] : stall_list[index + 2] + 1,
        stall_list[index] : stall_list[index + 2] + 1,
    ]

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



def flame_snipping_vertical(contact_map, stall_list, index, width, edge):
    """
    Flame snippet (vertical)
    To extract snippets around a vertical flame due to a boundary element
    with index "index" from a list of boundary elements "stall_list":
    ("edge_length" is the excluded areas at the ends of the flame)
    ------------------------------------------
    Function flame_snippet_vertical(contact_map, stall_list, index, width,
     edge_length):

         begin function

         set snippet_matrix from contact_map containing flame with a selected
          width:
             snippet = contact_map[
             (stall_list[index] + edge_length):(stall_list[index + 1] -
             edge_length),
             (stall_list[index + 1] - width):(stall_list[index + 1] +
             width),]

    return snippet_matrix

    end function
    ------------------------------------------
    """
    snippet = contact_map[
        (stall_list[index] + edge) : (stall_list[index + 1] - edge),
        (stall_list[index + 1] - width) : (stall_list[index + 1] + width),
    ]
    return snippet


def flame_snipping_horizontal(contact_map, stall_list, index, width, edge):
    """
    Flame snippet (horizontal)
    To extract snippets of a horizontal flame due to a boundary element with
    index "index" from a list of boundary elements "stall_list":("edge_length"
    is the excluded areas at the end of the flame)
    ---------------------------------------
    Function flame_snippet_horizontal(contact_map, stall_list, index, width,
     edge_length):

         begin function

         set snippet_matrix from contact_map containing flame with a selected
          width:
             snippet = contact_map[
             (stall_list[index] - width):(stall_list[index] + width),
             (stall_list[index] + edge_length):(stall_list[index + 1] -
             edge_length),]

    return snippet_matrix

    end function
    ---------------------------------------
    """
    snippet = contact_map[
        (stall_list[index] - width) : (stall_list[index] + width),
        (stall_list[index] + edge) : (stall_list[index + 1] - edge),
    ]
    return snippet
