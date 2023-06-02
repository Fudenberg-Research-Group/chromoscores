import numpy as np

from chromoscores.snipping import *

"""peak score"""


def peak_score_upperRight(
    peak_snippet, peak_width=3, background_width=10, pseudo_count=0
):
    """
    parameters
    ----------
    peak_snippet: snippet of the contact map around the peak
    peak_width: width of the peak
    background_width: width of the background outside the peak but inside the snippet on the upper right
    pseudo_count: pseudo count to avoid division by zero

    returns
    -------
    ratio of the mean of the peak and the mean of the background

    """
    mid = len(peak_snippet) // 2
    peak_interior = pseudo_count + np.mean(
        peak_snippet[
            mid - peak_width // 2 : mid + peak_width // 2 + 1,
            mid - peak_width // 2 : mid + peak_width // 2 + 1,
        ]
    )
    peak_background = pseudo_count + np.mean(
        peak_snippet[
            mid - background_width : mid - peak_width // 2,
            mid + peak_width // 2 + 1 : mid + background_width + 1,
        ]
    )

    return peak_interior / peak_background


def peak_score_lowerRight(
    peak_snippet, peak_width=3, background_width=10, pseudo_count=0
):
    """
    parameters
    ----------
    peak_snippet: snippet of the contact map around the peak
    peak_width: width of the peak
    background_width: width of the background outside the peak but inside the snippet on the lower right
    pseudo_count: pseudo count to avoid division by zero

    returns
    -------
    ratio of the mean of the peak and the mean of the background

    """
    mid = len(peak_snippet) // 2
    peak_interior = pseudo_count + np.mean(
        peak_snippet[
            mid - peak_width // 2 : mid + peak_width // 2 + 1,
            mid - peak_width // 2 : mid + peak_width // 2 + 1,
        ]
    )
    peak_background = pseudo_count + np.mean(
        peak_snippet[
            mid + peak_width // 2 + 1 : mid + background_width + 1,
            mid + peak_width // 2 + 1 : mid + background_width + 1,
        ]
    )

    return peak_interior / peak_background


def peak_score_upperLeft(
    peak_snippet, peak_width=3, background_width=10, pseudo_count=0
):
    """
    parameters
    ----------
    peak_snippet: snippet of the contact map around the peak
    peak_width: width of the peak
    background_width: width of the background outside the peak but inside the snippet on the upper left
    pseudo_count: pseudo count to avoid division by zero

    returns
    -------
    ratio of the mean of the peak and the mean of the background

    """
    mid = len(peak_snippet) // 2
    peak_interior = pseudo_count + np.mean(
        peak_snippet[
            mid - peak_width // 2 : mid + peak_width // 2 + 1,
            mid - peak_width // 2 : mid + peak_width // 2 + 1,
        ]
    )
    peak_background = pseudo_count + np.mean(
        peak_snippet[
            mid - background_width : mid - peak_width // 2,
            mid - background_width : mid - peak_width // 2,
        ]
    )

    return peak_interior / peak_background


def peak_score_lowerLeft(
    peak_snippet, peak_width=3, background_width=10, pseudo_count=0
):
    """
    parameters
    ----------
    peak_snippet: snippet of the contact map around the peak
    peak_width: width of the peak
    background_width: width of the background outside the peak but inside the snippet on the lower left
    pseudo_count: pseudo count to avoid division by zero

    returns
    -------
    ratio of the mean of the peak and the mean of the background

    """
    mid = len(peak_snippet) // 2
    peak_interior = pseudo_count + np.mean(
        peak_snippet[
            mid - peak_width // 2 : mid + peak_width // 2 + 1,
            mid - peak_width // 2 : mid + peak_width // 2 + 1,
        ]
    )
    peak_background = pseudo_count + np.mean(
        peak_snippet[
            mid + peak_width // 2 + 1 : mid + background_width + 1,
            mid - background_width : mid - peak_width // 2,
        ]
    )

    return peak_interior / peak_background


def peak_score(
    peak_snippet,
    peak_width,
    background_width,
    pseudo_count,
):
    """
    parameters
    ----------
    peak_snippet: snippet of the contact map around the peak
    peak_width: width of the peak
    background_width: width of the background outside the peak but inside the snippet 
    pseudo_count: pseudo count to avoid division by zero

    returns
    -------
    ratio of the mean of the peak and the mean of the background

    """
    avg = (
        peak_score_upperRight(
            peak_snippet,
            peak_width=peak_width,
            background_width=background_width,
            pseudo_count=pseudo_count,
        )
        + peak_score_lowerRight(
            peak_snippet,
            peak_width=peak_width,
            background_width=background_width,
            pseudo_count=pseudo_count,
        )
        + peak_score_upperLeft(
            peak_snippet,
            peak_width=peak_width,
            background_width=background_width,
            pseudo_count=pseudo_count,
        )
        + peak_score_lowerLeft(
            peak_snippet,
            peak_width=peak_width,
            background_width=background_width,
            pseudo_count=pseudo_count,
        )
    ) / 4
    return avg


"""Isolation score"""


def isolation_score(snippet, delta, diag_offset, max_dist, state, pseudo_count=1):
    """
    parameters
    ----------
    snippet: snippet of the contact map around the boundary element
    delta: distance from the border between in_tad and out_tad. It is defined to exclude
           flames when extracting in_tad and out_tad areas.
    diag_offset: distance from the diagonal. This also determines the size of the snippet.
    max_distance: maximum distance from the diagonal
    state: 1 for triangle snippets, 0 for square snippets
    pseudo_count: pseudo count to avoid division by zero

    returns
    -------
    ratio of the mean of the area inside tads and the area outside tad

    """
    in_tad, out_tad, pile_center = get_isolation_snippets(
        snippet, delta, diag_offset, max_dist, state
    )
    assert pile_center.shape == (len(in_tad), len(in_tad))
    return (pseudo_count + np.mean(pile_center[in_tad > 0])) / (
        pseudo_count + np.mean(pile_center[out_tad > 0])
    )




def tad_score(contact_map, boundary_list, index, delta, diag_offset, max_dist):
    """
    parameters
    ----------
    contact_map: contact map
    boundary_list: list of boundary elements
    delta: distance from the border between in_tad and out_tad. It is defined to exclude
           flames when extracting in_tad and out_tad areas.
    diag_offset: distance from the diagonal. This also determines the size of the snippet.
    max_dist: maximum distance from the diagonal

    returns
    -------
    ratio of the mean of the area inside tads and the area outside tad

    """
    in_tad, out_tad, pile_center = tad_snippet_sectors(
        contact_map, boundary_list, index, delta, diag_offset, max_dist
    )
    assert pile_center.shape == (len(in_tad), len(in_tad))
    return np.mean(pile_center[in_tad]) / np.mean(pile_center[out_tad])


"""Flame scores"""


def flame_score_vertical(flame_snippet, flame_thickness, background_thickness, pseudo_count=1):
    """
    parameters
    ----------
    flame_snippet: snippet of the contact map around the flame
    flame_thickness: thickness of the flame
    background_thickness: thickness of the background outside the flame but inside the snippet
    pseudo_count: pseudo count to avoid division by zero

    returns
    -------
    ratio of the mean of the flame and the mean of the background
    """
    mid = (np.shape(flame_snippet)[1]) // 2 + 1
    flame_interior = pseudo_count+np.mean(
        flame_snippet[:, mid - flame_thickness // 2 : mid + flame_thickness // 2]
    )
    flame_background = pseudo_count+np.mean(
        flame_snippet[
            :, mid - background_thickness // 2 : mid + background_thickness // 2
        ]
    )

    return flame_interior / flame_background


def flame_score_horizontal(flame_snippet, flame_thickness, background_thickness, pseudo_count=1):
    """
    parameters
    ----------
    flame_snippet: snippet of the contact map around the flame
    flame_thickness: thickness of the flame
    background_thickness: thickness of the background outside the flame but inside the snippet
    pseudo_count: pseudo count to avoid division by zero

    returns
    -------
    ratio of the mean of the flame and the mean of the background
    """
    mid = len(flame_snippet) // 2 + 1
    flame_interior = pseudo_count+np.mean(
        flame_snippet[mid - flame_thickness // 2 : mid + flame_thickness // 2, :]
    )
    flame_background = pseudo_count+np.mean(
        flame_snippet[
            mid - background_thickness // 2 : mid + background_thickness // 2, :
        ]
    )

    return flame_interior / flame_background
