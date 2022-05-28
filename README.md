Proposed resolution of a set of non linear optimization exercises, mainly exercise 58 that deals with formulating the problem of finding the smallest elipse aligned with the canonical basis for R2 that covers a set of points S in the same space.

[pdf](NLO_HW3_lourenco_rodrigues_83830.pdf) shows the derivation of an SDP formulation, and [python](ex_58_b.py) implements that formulation and uses a randomly generated point cloud to show that the derived formulation is correct and solves the problem.

This exercise can also be taken as an example of how to implement SDP constraints with cvxpy.