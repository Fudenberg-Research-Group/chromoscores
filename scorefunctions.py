###Funcitons###


#################################  snippets  ##################################
########   Peak snippet
### snippet with a size of "size" around peak with diagonal index of
### "stall_list_index" and off-diagonal index of "peak_index" from a
###contact map, "contact_map", and list of boundary elements of "stall_list":


def peak_snippet(contact_map, stall_list, stall_list_index, peak_index, size):

    if stall_list_index > len(stall_list):
        raise ValueError("peak index should be in the range of stall list")

    snippet = contact_map[
        (stall_list[stall_list_index] - size) : (stall_list[stall_list_index] + size),
        (stall_list[stall_list_index + peak_index] - size) : (
            stall_list[stall_list_index + peak_index] + size
        ),
    ]
    return snippet


###########   Tad snippet
### To extract Tad snippets adjacent to a boundary element with index of
### "index" from a list of "stall_list":


def tad_snippet(contact_map, stall_list, index):

    if index + 1 > len(stall_list):
        raise ValueError("index + 1 should be in the range of stall list")

    tad = contact_map[
        stall_list[index] : stall_list[index + 1] + 1,
        stall_list[index] : stall_list[index + 1] + 1,
    ]
    return tad


#######   Tad snippet sectors
### setting area in_tad vs area out_tad. "Delta" is defined to exclude
### flames when extracting in_tad and out_tad areas.


def tad_snippet_sectors(
    contact_map, stall_list, index, delta, diag_offset, max_distance
):

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


###########   Flame snippet (vertical)
### To extract snippets around a vertical flame due to a boundary element
### with index "index" from a list of boundary elements "stall_list":
### ("edge_length" is the excluded areas at the ends of the flame)


def flame_snippet_vertical(contact_map, stall_list, index, width, edge_length):
    snippet = contact_map[
        (stall_list[n] + edge_length) : (stall_list[n + 1] - edge_length),
        (stall_list[n + 1] - width) : (stall_list[n + 1] + width),
    ]
    return snippet


###########   Flame snippet (horizontal)
### To extract snippets of a horizontal flame due to a boundary element with
### index "index" from a list of boundary elements "stall_list": ("edge_length"
### is the excluded areas at the end of the flame)


def flame_snippet_horizontal(contact_map, stall_list, index, size, edge_length):
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
    in_tad, out_tad, pile_center = tad_snippet_sectors(
        contact_map, stall_list, index, delta, diag_offset, max_distance
    )
    assert pile_center.shape == (len(in_tad), len(in_tad))
    return np.mean(pile_center[in_tad]) / np.mean(pile_center[out_tad])


######## Flame scores #########
########### vertical flame score ###############
def flame_score_v(flame_snippet, flame_thickness, background_thickness):
    mid = (np.shape(flame_snippet)[1]) // 2 + 1
    return np.mean(
        avg_peaks[:, mid - flame_thickness // 2 : mid + flame_thickness // 2]
    ) / np.mean(
        avg_peaks[:, mid - background_thickness // 2 : mid + background_thickness // 2]
    )


############ horizontal flame score ###########
def flame_score_h(flame_snippet, flame_thickness, background_thickness):
    ### 6x6 area shifted down & left towards the diagonal ###
    mid = len(flame_snippet) // 2 + 1
    return np.mean(
        flame_snippet[mid - flame_thickness // 2 : mid + flame_thickness // 2, :]
    ) / np.mean(
        avg_peaks[mid - background_thickness // 2 : mid + background_thickness // 2, :]
    )


######
