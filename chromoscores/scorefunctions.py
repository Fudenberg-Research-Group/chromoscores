###Funcitons###


#################################  snippets  ##################################


def peak_snippet(contact_map, stall_list, stall_list_index, peak_index, size):
    '''
    Peak snippet; snippet with a size of "size" around peak with diagonal index of
    "stall_list_index" and off-diagonal index of "peak_index" from a
    contact map, "contact_map", and list of boundary elements of "stall_list":
    -----------------
    Function peak_snippet(contact_map, stall_list, stall_index, peak_index, size):

    begin function

         raise an error if peak_index is out of stall list range

         set snippet_matrix from contact_map containing peak with a selected size:
             snippet = contact_map[
                       (stall_list[stall_list_index] - size) : (stall_list[stall_list_index] + size),
                       (stall_list[stall_list_index + peak_index] - size) : (
                        stall_list[stall_list_index + peak_index] + size
                        ),
                        ]

     return snippet_matrix

     end function
    ----------------
    '''

    if stall_list_index > len(stall_list):
        raise ValueError("peak index should be in the range of stall list")

    snippet = contact_map[
        (stall_list[stall_list_index] - size) : (stall_list[stall_list_index] + size),
        (stall_list[stall_list_index + peak_index] - size) : (
            stall_list[stall_list_index + peak_index] + size
        ),
    ]
    return snippet


def tad_snippet(contact_map, stall_list, index):
    """
    Tad snippet
    To extract Tad snippets adjacent to a boundary element with index of
    "index" from a list of "stall_list":
    -------------------------
    Function tad_snippet(contact_map, stall_list, index):

         begin function

         raise an error if index is out of list range

         set tad_matrix between sequential stalls starting with index:
             tad = contact_map[
                   stall_list[index] : stall_list[index + 1] + 1,
                   stall_list[index] : stall_list[index + 1] + 1,
                   ]

    return tad_matrix

    end function
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
    Function tad_snippet_sectors(contact_map, stall_list, index, delta, diag_offset, max_distance):

         begin function

         getting tad_matrix from tad_snippet function

         set adjacent_tads starting with stall_index:
             pile_center = contact_map[
             stall_list[index] : stall_list[index + 2] + 1,
             stall_list[index] : stall_list[index + 2] + 1,
             ]

         raise an error if max_distance is larger than snippet

         set in_tad and out_tad areas:
             out_tad = np.zeros(np.shape(pile_center))
             out_tad[delta : tad_size - delta, tad_size + delta : -delta] = 1
             out_tad = np.tril(np.triu(out_tad, diag_offset), max_distance) > 0

             in_tad = np.zeros(np.shape(pile_center))
             in_tad[delta : tad_size - delta, delta : tad_size - delta] = 1
             in_tad[tad_size + delta : -delta, tad_size + delta : -delta] = 1
             in_tad = np.tril(np.triu(in_tad, diag_offset), max_distance) > 0

    return in_tad, out_tad, adjacent matrices

    end function
    --------------------------------------
    """
    tad = tad_snippet(contact_map, stall_list, index)
    tad_size = len(tad)

    pile_center = contact_map[
        stall_list[index] : stall_list[index + 2] + 1,
        stall_list[index] : stall_list[index + 2] + 1,
    ]

    if max_distance > len(pile_center) // 2:
        raise ValueError("max distance exceeds tad snippet size")

    out_tad = np.zeros(np.shape(pile_center))
    out_tad[delta : tad_size - delta, tad_size + delta : -delta] = 1
    out_tad = np.tril(np.triu(out_tad, diag_offset), max_distance) > 0

    in_tad = np.zeros(np.shape(pile_center))
    in_tad[delta : tad_size - delta, delta : tad_size - delta] = 1
    in_tad[tad_size + delta : -delta, tad_size + delta : -delta] = 1
    in_tad = np.tril(np.triu(in_tad, diag_offset), max_distance) > 0

    return in_tad, out_tad, pile_center


def flame_snippet_vertical(contact_map, stall_list, index, width, edge_length):
    """
    Flame snippet (vertical)
    To extract snippets around a vertical flame due to a boundary element
    with index "index" from a list of boundary elements "stall_list":
    ("edge_length" is the excluded areas at the ends of the flame)
    ------------------------------------------
    Function flame_snippet_vertical(contact_map, stall_list, index, width, edge_length):

         begin function

         set snippet_matrix from contact_map containing flame with a selected width:
             snippet = contact_map[
             (stall_list[n] + edge_length) : (stall_list[n + 1] - edge_length),
             (stall_list[n + 1] - width) : (stall_list[n + 1] + width),
             ]

    return snippet_matrix

    end function
    ------------------------------------------
    """
    snippet = contact_map[
        (stall_list[n] + edge_length) : (stall_list[n + 1] - edge_length),
        (stall_list[n + 1] - width) : (stall_list[n + 1] + width),
    ]
    return snippet


def flame_snippet_horizontal(contact_map, stall_list, index, size, edge_length):
    """
    Flame snippet (horizontal)
    To extract snippets of a horizontal flame due to a boundary element with
    index "index" from a list of boundary elements "stall_list": ("edge_length"
    is the excluded areas at the end of the flame)
    ---------------------------------------
    Function flame_snippet_horizontal(contact_map, stall_list, index, width, edge_length):

         begin function

         set snippet_matrix from contact_map containing flame with a selected width:
             snippet = contact_map[
             (stall_list[n] - width) : (stall_list[n] + width),
             (stall_list[n] + edge_length) : (stall_list[n + 1] - edge_length),
             ]

    return snippet_matrix

    end function
    ---------------------------------------
    """
    snippet = contact_map[
        (stall_list[n] - width) : (stall_list[n] + width),
        (stall_list[n] + edge_length) : (stall_list[n + 1] - edge_length),
    ]
    return snippet


################################### Scores ###################################


####### peak score ###########
def peak_lowerLeft(peak_snippet, peak_length, background_length, pseudo_count=1):
    mid = len(peak_snippet) // 2
    return (
        pseudo_count
        + np.mean(
            peak_snippet[
                mid - peak_length : mid + peak_length,
                mid - peak_length : mid + peak_length,
            ]
        )
    ) / (
        pseudo_count
        + np.mean(
            peak_snippet[
                mid + peak_length : mid + background_length :,
                mid - background_length : mid - peak_length,
            ]
        )
    )


def peak_lowerRight(peak_snippet, peak_length, background_length, pseudo_count=1):
    mid = len(peak_snippet) // 2
    return (
        pseudo_count
        + np.mean(
            peak_snippet[
                mid - peak_length : mid + peak_length,
                mid - peak_length : mid + peak_length,
            ]
        )
    ) / (
        pseudo_count
        + np.mean(
            peak_snippet[
                mid + peak_length : mid + background_length :,
                mid + peak_length : mid + background_length,
            ]
        )
    )


def peak_upperRight(peak_snippet, peak_length, background_length, pseudo_count=1):
    mid = len(peak_snippet) // 2
    return (
        pseudo_count
        + np.mean(
            peak_snippet[
                mid - peak_length : mid + peak_length,
                mid - peak_length : mid + peak_length,
            ]
        )
    ) / (
        pseudo_count
        + np.mean(
            peak_snippet[
                mid - background_length : mid - peak_length,
                mid + peak_length : mid + background_length,
            ]
        )
    )


def peak_upperLeft(peak_snippet, peak_length, background_length, pseudo_count=1):
    mid = len(peak_snippet) // 2
    return (
        pseudo_count
        + np.mean(
            peak_snippet[
                mid - peak_length : mid + peak_length,
                mid - peak_length : mid + peak_length,
            ]
        )
    ) / (
        pseudo_count
        + np.mean(
            peak_snippet[
                mid - background_length : mid - peak_length,
                mid - background_length : mid - peak_length,
            ]
        )
    )


def peak_score(peak_snippet, peak_length, background_length, pseudo_count=1):
    avg = (
        peak_upperRight(peak_snippet, peak_length, background_length)
        + peak_lowerRight(peak_snippet, peak_length, background_length)
        + peak_upperLeft(peak_snippet, peak_length, background_length)
        + peak_lowerLeft(peak_snippet, peak_length, background_length)
    ) / 4
    return avg


######### Tad score ########


def tad_score(contact_map, stall_list, index, delta, diag_offset, max_distance):
    """
    ----------------------
    Function tad_score(contact_map, stall_list, index, delta, diag_offset, max_distance)
    begin function

    set in_tad, out_tad, and adjacent matrices from tad_snippet_sectors function

    assert adjacent matrix to be in the shape of in_tad matrix 

    return score as average of in_tad matrix over out_tad score:
           tad_score=np.mean(pile_center[in_tad]) / np.mean(pile_center[out_tad])

    end function
    ----------------------
    """
    in_tad, out_tad, pile_center = tad_snippet_sectors(
        contact_map, stall_list, index, delta, diag_offset, max_distance
    )
    assert pile_center.shape == (len(in_tad), len(in_tad))
    return np.mean(pile_center[in_tad]) / np.mean(pile_center[out_tad])


######## Flame scores #########
def flame_score_v(flame_snippet, flame_thickness, background_thickness):
    """
    vertical flame score
    """
    mid = (np.shape(flame_snippet)[1]) // 2 + 1
    return np.mean(
        avg_peaks[:, mid - flame_thickness // 2 : mid + flame_thickness // 2]
    ) / np.mean(
        avg_peaks[:, mid - background_thickness // 2 : mid + background_thickness // 2]
    )


def flame_score_h(flame_snippet, flame_thickness, background_thickness):
    """
    horizontal flame score
    """
    mid = len(flame_snippet) // 2 + 1
    return np.mean(
        flame_snippet[mid - flame_thickness // 2 : mid + flame_thickness // 2, :]
    ) / np.mean(
        avg_peaks[mid - background_thickness // 2 : mid + background_thickness // 2, :]
    )


######
