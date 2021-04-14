import numpy as np 
from scipy.special import binom

def bernstein_20_coeffs(t,tmin,tmax):
    l = tmax-tmin

    t = (t-tmin)/l
    n = 20


    P0 = binom(n,0)*((1-t)**(n-0))*t**0

    P1 = binom(n,1)*((1-t)**(n-1))*t**1


    P2 = binom(n,2)*((1-t)**(n-2))*t**2

    P3 = binom(n,3)*((1-t)**(n-3))*t**3

    P4 = binom(n,4)*((1-t)**(n-4))*t**4

    P5 = binom(n,5)*((1-t)**(n-5))*t**5

    P6 = binom(n,6)*((1-t)**(n-6))*t**6

    P7 = binom(n,7)*((1-t)**(n-7))*t**7

    P8 = binom(n,8)*((1-t)**(n-8))*t**8

    P9 = binom(n,9)*((1-t)**(n-9))*t**9


    P10 = binom(n,10)*((1-t)**(n-10))*t**10

    P11 = binom(n,11)*((1-t)**(n-11))*t**11

    P12 = binom(n,12)*((1-t)**(n-12))*t**12

    P13 = binom(n,13)*((1-t)**(n-13))*t**13

    P14 = binom(n,14)*((1-t)**(n-14))*t**14

    P15 = binom(n,15)*((1-t)**(n-15))*t**15

    P16 = binom(n,16)*((1-t)**(n-16))*t**16

    P17 = binom(n,17)*((1-t)**(n-17))*t**17

    P18 = binom(n,18)*((1-t)**(n-18))*t**18

    P19 = binom(n,19)*((1-t)**(n-19))*t**19

    P20 = binom(n,20)*((1-t)**(n-20))*t**20


    P0dot = -20.0*binom(n,0)*(-t + 1)**19

    P1dot = binom(n,1)*(-19.0*t*(-t + 1)**18 + (-t + 1)**19)

    P2dot = binom(n,2)*(-18*t**2*(1 - t)**17 + 2*t*(1 - t)**18)

    P3dot = binom(n,3)*(-17*t**3*(1 - t)**16 + 3*t**2*(1 - t)**17)

    P4dot = binom(n,4)*(-16*t**4*(1 - t)**15 + 4*t**3*(1 - t)**16)

    P5dot = binom(n,5)*(-15*t**5*(1 - t)**14 + 5*t**4*(1 - t)**15)

    P6dot = binom(n,6)*(-14*t**6*(1 - t)**13 + 6*t**5*(1 - t)**14)

    P7dot = binom(n,7)*(-13*t**7*(1 - t)**12 + 7*t**6*(1 - t)**13)

    P8dot = binom(n,8)*(-12*t**8*(1 - t)**11 + 8*t**7*(1 - t)**12)

    P9dot = binom(n,9)*(-11*t**9*(1 - t)**10 + 9*t**8*(1 - t)**11)

    P10dot = binom(n,10)*(-10*t**10*(1 - t)**9 + 10*t**9*(1 - t)**10)

    P11dot = binom(n,11)*(-9*t**11*(1 - t)**8 + 11*t**10*(1 - t)**9)
    
    P12dot = binom(n,12)*(-8*t**12*(1 - t)**7 + 12*t**11*(1 - t)**8)

    P13dot = binom(n,13)*(-7*t**13*(1 - t)**6 + 13*t**12*(1 - t)**7)

    P14dot = binom(n,14)*(-6*t**14*(1 - t)**5 + 14*t**13*(1 - t)**6)

    P15dot = binom(n,15)*(-5*t**15*(1 - t)**4 + 15*t**14*(1 - t)**5)

    P16dot = binom(n,16)*(-4*t**16*(1 - t)**3 + 16*t**15*(1 - t)**4)

    P17dot = binom(n,17)*(-3*t**17*(1 - t)**2 + 17*t**16*(1 - t)**3)

    P18dot = binom(n,18)*(-2*t**18*(1 - t)**1 + 18*t**17*(1 - t)**2)

    P19dot = binom(n,19)*(-t**19*(1 - t)**0 + 19*t**18*(1 - t))

    P20dot = binom(n,20)*20*t**19


    P0ddot = binom(n,0)*380.0*(-t + 1)**18

    P1ddot = binom(n,1)*(-38*(t - 1)**17*(10*t - 1))

    P2ddot = binom(n,2)*(2*(t - 1)**16*(153*t**2 + 36*t*(t - 1) + (t - 1)**2))

    P3ddot = binom(n,3)*(-2*t*(t - 1)**15*(136*t**2 + 51*t*(t - 1) + 3*(t - 1)**2))

    P4ddot = binom(n,4)*(4*t**2*(t - 1)**14*(60*t**2 + 32*t*(t - 1) + 3*(t - 1)**2))

    P5ddot = binom(n,5)*(-10*t**3*(t - 1)**13*(21*t**2 + 15*t*(t - 1) + 2*(t - 1)**2))

    P6ddot = binom(n,6)*(2*t**4*(t - 1)**12*(91*t**2 + 84*t*(t - 1) + 15*(t - 1)**2))

    P7ddot = binom(n,7)*(-2*t**5*(t - 1)**11*(78*t**2 + 91*t*(t - 1) + 21*(t - 1)**2))

    P8ddot = binom(n,8)*(4*t**6*(t - 1)**10*(33*t**2 + 48*t*(t - 1) + 14*(t - 1)**2))

    P9ddot = binom(n,9)*(-2*t**7*(t - 1)**9*(55*t**2 + 99*t*(t - 1) + 36*(t - 1)**2))

    P10ddot = binom(n,10)*(10*t**8*(t - 1)**8*(9*t**2 + 20*t*(t - 1) + 9*(t - 1)**2))

    P11ddot = binom(n,11)*(-2*t**9*(t - 1)**7*(36*t**2 + 99*t*(t - 1) + 55*(t - 1)**2))

    P12ddot = binom(n,12)*(4*t**10*(t - 1)**6*(14*t**2 + 48*t*(t - 1) + 33*(t - 1)**2))

    P13ddot = binom(n,13)*(-2*t**11*(t - 1)**5*(21*t**2 + 91*t*(t - 1) + 78*(t - 1)**2))

    P14ddot = binom(n,14)*(2*t**12*(t - 1)**4*(15*t**2 + 84*t*(t - 1) + 91*(t - 1)**2))

    P15ddot = binom(n,15)*(-10*t**13*(t - 1)**3*(2*t**2 + 15*t*(t - 1) + 21*(t - 1)**2))

    P16ddot = binom(n,16)*(4*t**14*(t - 1)**2*(3*t**2 + 32*t*(t - 1) + 60*(t - 1)**2))

    P17ddot = binom(n,17)*(-2*t**15*(t - 1)*(3*t**2 + 51*t*(t - 1) + 136*(t - 1)**2))

    P18ddot = binom(n,18)*(2*t**16*(t**2 + 36*t*(t - 1) + 153*(t - 1)**2))

    P19ddot = binom(n,19)*(-38*t**17*(10*t - 9))

    P20ddot = binom(n,20)*(380*t**18)


    P = np.hstack((P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20))

    Pdot = np.hstack((P0dot, P1dot, P2dot, P3dot, P4dot, P5dot, P6dot, P7dot, P8dot, P9dot, P10dot, P11dot, P12dot, P13dot, P14dot, P15dot, P16dot, P17dot, P18dot, P19dot, P20dot ))/l

    Pddot = np.hstack((P0ddot, P1ddot, P2ddot, P3ddot, P4ddot, P5ddot, P6ddot, P7ddot, P8ddot, P9ddot, P10ddot, P11ddot, P12ddot, P13ddot, P14ddot, P15ddot, P16ddot, P17ddot, P18ddot, P19ddot, P20ddot))/(l**2)

    return P, Pdot, Pddot
