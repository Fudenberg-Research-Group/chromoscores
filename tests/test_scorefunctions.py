from chromoscores.scorefunctions import *


mat = [[1,1,2],[1,4,1],[2,1,1]]
assert peak_score_upperRight(mat, peak_length=0, background_length = 1, pseudo_count=0) == 2
assert peak_score_upperLeft(mat, peak_length=0, background_length = 1, pseudo_count=0) == 4
assert peak_score(mat, peak_length=0, background_length = 1, pseudo_count=0) == 3
