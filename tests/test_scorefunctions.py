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


""" isolation snippets/score """
example_A=np.array(np.ones((18,18)))
a,b,c=get_isolation_snippets(example_A,1,3,10,1)

mat=a+b
assert isolation_score(mat,delta=1,diag_offset=3,max_dist=10,state=1,pseudo_count=1)==1

mat=3*a+b
assert isolation_score(mat,delta=1,diag_offset=3,max_dist=10,state=1,pseudo_count=1)=2


mat=a+3*b
assert isolation_score(mat,delta=1,diag_offset=3,max_dist=10,state=1,pseudo_count=1)==0.5


