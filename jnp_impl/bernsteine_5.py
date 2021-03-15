import cupy as cp
from scipy.special import binom

def bernstein_5_coeffs(t, tmin, tmax):
    l = tmax-tmin
    t = (t-tmin)/l
    n = 5

    P0 = binom(n,0)*((1-t)**(n-0))*t**0
    P1 = binom(n,1)*((1-t)**(n-1))*t**1
    P2 = binom(n,2)*((1-t)**(n-2))*t**2
    P3 = binom(n,3)*((1-t)**(n-3))*t**3
    P4 = binom(n,4)*((1-t)**(n-4))*t**4
    P5 = binom(n,5)*((1-t)**(n-5))*t**5

    P0dot = binom(n,0)*(-5*(1 - t)**4)
    P1dot = binom(n,1)*(-4*t*(1 - t)**3 + (1 - t)**4)
    P2dot = binom(n,2)*(-3*t**2*(1 - t)**2 + 2*t*(1 - t)**3)
    P3dot = binom(n,3)*(t**3*(2*t - 2) + 3*t**2*(1 - t)**2)
    P4dot = binom(n,4)*(-t**4 + 4*t**3*(1 - t))
    P5dot = binom(n,5)*(5*t**4)

    P0ddot = binom(n,0)*(-20*(t - 1)**3)
    P1ddot = binom(n,1)*(4*(t - 1)**2*(5*t - 2))
    P2ddot = binom(n,2)*(-2*(t - 1)*(3*t**2 + 6*t*(t - 1) + (t - 1)**2))
    P3ddot = binom(n,3)*(2*t*(t**2 + 6*t*(t - 1) + 3*(t - 1)**2))
    P4ddot = binom(n,4)*(-4*t**2*(5*t - 3))
    P5ddot = binom(n,5)*(20*t**3)

    P = cp.hstack((P0, P1, P2, P3, P4, P5 ))

    Pdot = cp.hstack((P0dot, P1dot, P2dot, P3dot, P4dot, P5dot))/l

    Pddot = cp.hstack((P0ddot, P1ddot, P2ddot, P3ddot, P4ddot, P5ddot))/(l**2)


    return P, Pdot, Pddot