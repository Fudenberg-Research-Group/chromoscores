import numpy as np


def peak_snipping(contact_map, size, peak_coordinate):
    """
    Peak snippet; snippet with a size of "size" around peak with coordinate of "peak_coordinate"
    on a contact map, "contact_map". peak_coordinate should be in the format of (i,j)
    -----------------
    Function peak_snippet(contact_map, size, peak_coordinate):

    begin function

         raise an error if peak coordinate + window size is out of the map

         snippet = contact_map[
        (peak_coordinate[0] - size): (peak_coordinate[0] + size),
        (peak_coordinate[1] - size): (peak_coordinate[1] + size)]

     return snippet_matrix

     end function
    ----------------
    """

    if (peak_coordinate[1] + size) > len(contact_map) or (
        peak_coordinate[0] - size
    ) < 0:

        raise ValueError(
            "selected window size for peak coordinate exceeds size of the contact map"
        )

    snippet = contact_map[
        (peak_coordinate[0] - size): (peak_coordinate[0] + size),
        (peak_coordinate[1] - size): (peak_coordinate[1] + size),
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
                   stall_list[index]: stall_list[index + 1] + 1,
                   stall_list[index]: stall_list[index + 1] + 1,
                   ]

    return tad_matrix

    end function
    """

    if index + 1 > len(stall_list):
        raise ValueError("index + 1 should be in the range of stall list")

    tad = contact_map[
        stall_list[index]: stall_list[index + 1] + 1,
        stall_list[index]: stall_list[index + 1] + 1,
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
             stall_list[index]: stall_list[index + 2] + 1,
             stall_list[index]: stall_list[index + 2] + 1,
             ]

         raise an error if max_distance is larger than snippet

         set in_tad and out_tad areas:
             out_tad = np.zeros(np.shape(pile_center))
             out_tad[delta: tad_size - delta, tad_size + delta: -delta] = 1
             out_tad = np.tril(np.triu(out_tad, diag_offset), max_distance) > 0

             in_tad = np.zeros(np.shape(pile_center))
             in_tad[delta: tad_size - delta, delta: tad_size - delta] = 1
             in_tad[tad_size + delta: -delta, tad_size + delta: -delta] = 1
             in_tad = np.tril(np.triu(in_tad, diag_offset), max_distance) > 0

    return in_tad, out_tad, adjacent matrices

    end function
    --------------------------------------
    """
    tad = tad_snippet(contact_map, stall_list, index)
    tad_size = len(tad)

    pile_center = contact_map[
        stall_list[index]: stall_list[index + 2] + 1,
        stall_list[index]: stall_list[index + 2] + 1,
    ]

    if max_distance > len(pile_center) // 2:
        raise ValueError("max distance exceeds tad snippet size")

    out_tad = np.zeros(np.shape(pile_center))
    out_tad[delta: tad_size - delta, tad_size + delta: -delta] = 1
    out_tad = np.tril(np.triu(out_tad, diag_offset), max_distance) > 0

    in_tad = np.zeros(np.shape(pile_center))
    in_tad[delta: tad_size - delta, delta: tad_size - delta] = 1
    in_tad[tad_size + delta: -delta, tad_size + delta: -delta] = 1
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
             (stall_list[index] + edge_length): (stall_list[n + 1] - edge_length),
             (stall_list[n + 1] - width): (stall_list[n + 1] + width),
             ]

    return snippet_matrix

    end function
    ------------------------------------------
    """
    snippet = contact_map[
        (stall_list[index] + edge_length): (stall_list[n + 1] - edge_length),
        (stall_list[n + 1] - width): (stall_list[n + 1] + width),
    ]
    return snippet


def flame_snippet_horizontal(contact_map, stall_list, index, width, edge_length):
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
             (stall_list[index] - width): (stall_list[index] + width),
             (stall_list[index] + edge_length): (stall_list[n + 1] - edge_length),
             ]

    return snippet_matrix

    end function
    ---------------------------------------
    """
    snippet = contact_map[
        (stall_list[index] - width): (stall_list[index] + width),
        (stall_list[index] + edge_length): (stall_list[n + 1] - edge_length),
    ]
    return snippet
