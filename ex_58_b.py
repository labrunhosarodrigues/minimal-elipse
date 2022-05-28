"""
Resolution implementation of exercise 58.

Minimal elipse in R2, with elipse axis aligned with canonical vectors, that cover a set of points S.

This script formulates a convex SDP problem to solve the statement above.


author: Lourenço A. Rodrigues, 83830
"""

# imports
# built-in

# local

# 3rd-party
import numpy as np
import cvxpy as cp
import plotly.graph_objects as go
import plotly.offline as po


def problem(points):

    lambda_ = cp.Variable(1)
    B1 = cp.Variable(1)
    B2 = cp.Variable(1)
    d1 = cp.Variable(1)
    d2 = cp.Variable(1)

    objective = cp.Maximize(lambda_)

    M3 = []
    for i in range(3):
        for j in range(3):
            e_i = np.zeros((3, 3))
            e_i[i, j] = 1
            M3.append(e_i)
    
    M2 = []
    for i in range(2):
        for j in range(2):
            e_i = np.zeros((2, 2))
            e_i[i, j] = 1
            M2.append(e_i)
    
    constraints = []
    for point in points:
        # | B1  0 d1 |
        # |  0 B2 d2 |
        # | d1 d2 miu|
        temp_constraint = (
            B1*M3[0]+
            d1*(M3[2]+M3[6])+
            B2*M3[4]+
            d2*(M3[5]+M3[7])+
            (# miu
                1-B1*(point[0]**2)-B2*(point[1]**2)+
                2*(point[0]*d1+point[1]*d2)
            )*M3[8] >> 0
        )
        constraints.append(temp_constraint)
    # | B1  0 |
    # |  0 B2 |
    temp_constraint = (B1*M2[0]+B2*M2[3] >> 0)
    constraints.append(temp_constraint)

    # | (B1+B2)    0     2*lambda |
    # |    0     (B1+B2)  (B1-B2) |
    # | 2*lambda (B1-B2)  (B1+B2) |
    temp_constraint = (
        (B1+B2)*(M3[0]+M3[4]+M3[8])+
        (B1-B2)*(M3[5]+M3[7])+
        (2*lambda_)*(M3[2]+M3[6]) >> 0
    )
    constraints.append(temp_constraint)

    prob = cp.Problem(objective, constraints)

    return prob, lambda_, B1, B2, d1, d2


def reverse_parameters(B1, B2, d1, d2):
    A1 = 1/B1.value
    A2 = 1/B2.value
    c1 = d1.value/B1.value
    c2 = d2.value/B2.value

    return A1, A2, c1, c2


def make_elipse(A1, A2, c1, c2):
    theta = np.linspace(0, 1, 1000)*2*np.pi

    x = np.sqrt(A1)*np.cos(theta)+c1
    y = np.sqrt(A2)*np.sin(theta)+c2
    
    return x, y


def visualize(points, x, y):

    fig = go.Figure(
        data=[
            go.Scatter(x=points[:,0], y=points[:,1], mode="markers", name="Points"),
            go.Scatter(x=x, y=y, mode="lines", name="elipse!")
        ]
    )
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )

    po.plot(fig)


def generate_points(K, c):
    A = np.random.normal(0, 1, (2, 2))
    points = c@np.ones((1, K)) + A@np.random.normal(0, 1, (2, K))

    return points.T


if __name__ == "__main__":

    points = generate_points(100, np.array([[3], [1]]))

    prob, lambda_, B1, B2, d1, d2 = problem(points)

    prob.solve()

    print(lambda_.value**2, B1.value*B2.value)
    params = reverse_parameters(B1, B2, d1, d2)

    x, y = make_elipse(*params)

    visualize(points, x, y)

    
