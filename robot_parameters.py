import math
import numpy as np

class Args():
    def __init__(self):

        # Constants

        self.alpha_x_try = -1
        self.npoints = 500
        self.L = 0.1255             # metres
        self.Fv = 0.0025            # N.m/(rad/sec)
        self.Kb = 0.0255            # V/(rad/sec)
        self.Kt = 0.0255            # N.m/A
        self.Mc = 1.59              # Kg
        self.Mw = 0.133             # Kg
        self.Vs = 24                # Volt
        self.r = 0.03               # metres
        self.tw = 0.045             # metres
        self.Ra = 7.9               # Ohm
        self.n = 1                  # Assumed Gear Ratio
        self.max_u = 1
        self.tol_angle = 1e-1
        self.x_tol = 1e-1
        self.v_tol = 1e-1

        # Matrices & constants used in calculations

        self.M = self.Mc + 3*self.Mw
        self.Jw = 0.5*self.Mw*self.r**2
        self.Jc = 0.5*self.Mc*self.L**2
        self.J = 3*(self.Jw + self.Mw*self.L**2 + 0.25*self.Mw*self.r**2) + self.Jc
        self.B = np.matrix([[0, 1, self.L], [-math.sin(math.pi/3), -math.cos(math.pi/3), self.L], [math.sin(math.pi/3), -math.cos(math.pi/3), self.L]])
        self.M = np.diag([self.M, self.M, self.J])
        self.k2 = self.Vs*self.Kt*self.n/(self.Ra*self.r)
        self.D = self.k2*self.B.T
        self.k1 = 1.5*(self.Fv+self.Kt*self.Kb*self.n**2/self.Ra)/self.r**2
        self.C = self.k1*np.diag([1, 1, 2*self.L**2])
        self.Q = self.B.T*self.B
        self.A = (self.Jw*self.Q)/self.r**2 + self.M
        self.w1 = (self.Vs**2)/self.Ra
        self.w2 = self.Kb*self.n*self.Vs/(self.Ra*self.r)

        self.A = np.array(self.A)
        self.Q = np.array(self.Q)

        # For differential equation of phi-dot
        self.C3 = (self.C[2][2]/self.A[2][2])**2 - self.k2*self.w2*self.Q[2][2]*self.C[2][2]/(self.w1*self.A[2][2]**2)
        self.C4 = self.Q[2][2]*(self.k2**2)/(2*self.w1*(self.A[2][2])**2)
        self.h3 = 3*(self.L**2)*(self.Fv+self.Kt*self.Kb*(self.n**2)/self.Ra)/self.r**2
        self.k = (self.h3**2 - 3*(self.L**2)*self.h3*self.Kt*self.Kb*(self.n**2)/(self.Ra*(self.r**2)))/(self.Jc + (3*self.Jw*self.L**2)/self.r**2)**2
        self.tau_w = 1/(self.k**0.5)

        # For differential equation of x_dot
        self.R = [[math.cos(0), -math.sin(0), 0 ], [math.sin(0), math.cos(0), 0], [0, 0, 1]]
        self.R_dot = [[-math.sin(0), -math.cos(0), 0], [math.cos(0), -math.sin(0), 0], [0, 0, 0]]
        self.C1 = (self.C[0][0]/self.A[0][0])**2 - self.k2*self.w2*self.Q[0][0]*self.C[0][0]/(self.w1*self.A[0][0]**2)
        self.C2 = (self.k2**2)*self.Q[0][0]/(2*self.w1*self.A[0][0]**2)