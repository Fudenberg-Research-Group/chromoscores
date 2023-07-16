import numpy as np


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
    if background_width > len(peak_snippet // 2):
        raise ValueError("background_width exceeds the size of the snippet")

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
    if background_width > len(peak_snippet // 2):
        raise ValueError("background_width exceeds the size of the snippet")

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
    if background_width > len(peak_snippet // 2):
        raise ValueError("background_width exceeds the size of the snippet")

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
    if background_width > len(peak_snippet // 2):
        raise ValueError("background_width exceeds the size of the snippet")

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


def _get_isolation_areas(contact_map, delta=1, diag_offset=3, max_distance=10, snippet_shapes='triangle'):
    """
    parameters
    ----------
    contact_map: snippet of a contact map around a boundary element
    delta: distance from the border between in_tad and out_tad
    diag_offset: distance of the snippet from the diagonal. This also determines the size of the snippet.
    max_distance: maximum distance from the diagonal
    state: 1 for triangle snippets, 0 for square snippets

    returns
    -------
    areas with a size of diag_offset inside and outside a tad
    """
    if snippet_shapes == 'triangle':
        triu_num = 1
    elif snippet_shapes == 'square':
        triu_num = 0
    else:
        raise ValueError("snippet shape can be triangle or square") 


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
    in_tad = np.tril(np.triu(in_tad, triu_num * (diag_offset + 1)), max_distance)

    return in_tad, out_tad, pile_center


def isolation_score(snippet, delta, diag_offset, max_dist, snippet_shapes , pseudo_count=0):
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
    in_tad, out_tad, pile_center = _get_isolation_areas(
        snippet, delta, diag_offset, max_dist, snippet_shapes
    )
    assert pile_center.shape == (len(in_tad), len(in_tad))
    return (pseudo_count + np.mean(pile_center[in_tad > 0])) / (
        pseudo_count + np.mean(pile_center[out_tad > 0])
    )


"""Flame scores"""


def flame_score_vertical(
    flame_snippet, flame_thickness, background_thickness, pseudo_count=1
):
    """
    parameters
    ----------
    flame_snippet: snippet of the contact map around a boundary element
    flame_thickness: thickness of the flame
    background_thickness: thickness of the background outside the flame but inside the snippet
    pseudo_count: pseudo count to avoid division by zero

    returns
    -------
    ratio of the mean of the flame and the mean of the background
    """
    mid = (np.shape(flame_snippet)[1]) // 2 
    flame_interior = pseudo_count + np.mean(
        flame_snippet[:mid, mid - flame_thickness // 2 : mid + flame_thickness // 2]
    )
    flame_background = pseudo_count + np.mean(
        flame_snippet[
            : mid, mid - background_thickness // 2 : mid - flame_thickness // 2
        ] + flame_snippet[
            : mid, mid + flame_thickness // 2 : mid + background_thickness // 2
        ]
    ) / 2

    return flame_interior / flame_background


def flame_score_horizontal(
    snippet, flame_thickness, background_thickness, pseudo_count=1
):
    """
    parameters
    ----------
    flame_snippet: snippet of the contact map around a boundary element
    flame_thickness: thickness of the flame
    background_thickness: thickness of the background outside the flame but inside the snippet
    pseudo_count: pseudo count to avoid division by zero

    returns
    -------
    ratio of the mean of the flame and the mean of the background
    """
    mid = len(snippet) // 2
    flame_interior = pseudo_count + np.mean(
        snippet[mid - flame_thickness // 2 : mid + flame_thickness // 2, mid:]
    )
    flame_background = (
        pseudo_count
        + np.mean(
            snippet[mid - background_thickness // 2 : mid - flame_thickness // 2, mid:]
            + snippet[
                mid + flame_thickness // 2 : mid + background_thickness // 2, mid:
            ]
        )
        / 2
    )

    return flame_interior / flame_background
