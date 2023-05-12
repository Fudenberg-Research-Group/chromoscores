from chromoscores.scorefunctions import *


mat = [[1, 1, 2], [1, 4, 1], [2, 1, 1]]
mat = np.array(mat)
assert peak_score_upperRight(mat, peak_width=1, background_width=1, pseudo_count=0) == 2
assert peak_score_upperLeft(mat, peak_width=1, background_width=1, pseudo_count=0) == 4
assert peak_score_lowerRight(mat, peak_width=1, background_width=1, pseudo_count=0) == 4
assert peak_score_lowerLeft(mat, peak_width=1, background_width=1, pseudo_count=0) == 2
assert peak_score(mat, peak_width=1, background_width=1, pseudo_count=0) == 3


########non-zero pseudo count
mat = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
mat = np.array(mat)
assert peak_score(mat, peak_width=1, background_width=1, pseudo_count=1) == 2
