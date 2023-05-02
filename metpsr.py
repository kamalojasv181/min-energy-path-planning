import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import integrate
import mpmath as mp

class METPSR():
    def __init__(self, args):

        self.args = args
        mp.dps = 4

    def min_energy(self, start_coordinates, end_coordinates, phi_f):

        xf = mp.sqrt((start_coordinates[0] - end_coordinates[0])**2 + (start_coordinates[1] - end_coordinates[1])**2)
        sample_times = np.linspace(1, self.energy_for_given_tf(xf, phi_f, 1)[2], 100)


        energy_values = 0*sample_times

        for id, t in enumerate(sample_times):
            energy, validity, _ = self.energy_for_given_tf(xf, phi_f, t)

            if validity:
                energy_values[id] = energy

            else:
                energy_values[id] = 1e10

        minimum_energy, min_id = np.min(energy_values), np.argmin(energy_values)
        min_time = sample_times[min_id]

        return minimum_energy, min_time
  


    def energy_for_given_tf(self, xf, phi_f, tf):

        h2 = mp.sinh(tf/self.args.tau_w)
        h1 = 2*(1 - mp.cosh(tf/self.args.tau_w)) + h2


        t_span = np.linspace(0, tf, self.args.npoints)
        alpha = [0, 0, (-1*(self.args.C3**1.5)*phi_f/(self.args.C4*(self.args.C3**0.5)*tf - 2*self.args.C4*mp.tanh((self.args.C3**0.5)*tf/2)))]
        λ = [0, 0, 0]

        dt = t_span[1] - t_span[0]
        phi_dot = np.zeros(self.args.npoints)

        for id, timepoint in enumerate(t_span):
            phi_dot[id] = self.merv(alpha[2], timepoint, tf)

        ang_acc = np.zeros(self.args.npoints)


        for id, timepoint in enumerate(t_span):
            if id == t_span.shape[0] - 1:
                ang_acc[id] = -phi_dot[id]/dt
            
            else:
                ang_acc[id] = (phi_dot[id+1] -phi_dot[id])/dt


        phi = self.calc_dist(phi_dot, ang_acc, dt)

        
        t, v, a, alpha_x = self.metpsr(tf, xf, alpha, phi_f)
        alpha[0] = alpha_x

        x_dot = np.concatenate((v.reshape(-1, 1), 0*v.reshape(-1, 1), phi_dot.reshape(-1, 1)), axis = 1)
        x_dot_dot = np.concatenate((a.reshape(-1, 1), 0*a.reshape(-1, 1), ang_acc.reshape(-1, 1)), axis = 1)

        lmda = 0*x_dot
        for t in range(t_span.shape[0]):
            x_dot_cur = x_dot[t]
            x_dot_dot_cur = x_dot_dot[t]

            omega = phi_dot[t]

            R_dot = omega*self.return_r_dot(phi[t])
            R_ = self.return_r(phi[t])
            lmda[t] = self.get_lambda(x_dot_cur, x_dot_dot_cur, omega, R_, R_dot)


        u = 0*lmda
        for t in range(t_span.shape[0]):

            x_dot_cur = x_dot[t]
            R_ = self.return_r(phi[t])
            cur_lmda = lmda[t]
            u[t] = self.return_control_vector(R_, x_dot_cur, cur_lmda)

        E = 0*t_span

        for t in range(t_span.shape[0]):

            u_cur = u[t]
            x_dot_cur = x_dot[t]

            R_ = self.return_r(phi[t])
            E[t] = self.return_energy(u_cur, x_dot_cur, R_)

        max_positive_u = np.max(u)
        max_negative_u = abs(np.min(u))

        max_obtained_u = max(max_positive_u, max_negative_u)

        Energy = dt*integrate.cumtrapz(E)
        x = self.calc_dist(v, a, dt)

        return [Energy[-1], self.is_valid(phi, phi_f, x, xf, v), max_obtained_u - self.args.max_u]
        

    def merv(self, alpha, t, tf):
        return -(self.args.C4*alpha*mp.exp(-self.args.C3**(1/2)*t)*(mp.exp(self.args.C3**(1/2)*t) - 
        mp.exp(2*self.args.C3**(1/2)*t) - mp.exp(self.args.C3**(1/2)*tf) + 
        mp.exp(self.args.C3**(1/2)*t)*mp.exp(self.args.C3**(1/2)*tf)))/(self.args.C3*(mp.exp(self.args.C3**(1/2)*tf) + 1))

    def max_acc(self):
        
        c1 = np.matmul(np.array(self.args.B).T, np.array([0, -1, 1]))
        c2 = np.matmul(np.linalg.inv(self.args.A), c1)
        c3 = np.matmul(np.linalg.inv(np.array(self.args.R).T), c2)
        x_dd = self.args.k2*c3
        return x_dd[0]

    def calculate_vx(self, a0, alpha_x, tf, alpha_phi):

        h2 = mp.sinh(tf/self.args.tau_w)
        h1 = 2*(1 - mp.cosh(tf/self.args.tau_w)) + h2

        def dvdt(v, t):
            omega_star = self.merv(alpha_phi, t, tf)
            return [v[1], (omega_star**2 + self.args.C1)*v[0] + self.args.C2*alpha_x]
        ts = np.linspace(0, tf, self.args.npoints)

        ic = [0, a0]

        vs = odeint(dvdt, ic, ts)

        return ts, vs[:, 0], vs[:, 1]

    def calculate_distance(self, v, dt):

        x = 0

        for vt in v:
            x += dt*vt
        
        return x

    def calc_dist(self, v, a, dt):

        timepoints = v.shape[0]
        x = np.zeros(timepoints)
        
        for t in range(1, timepoints):
            x[t] = x[t-1] + v[t-1]*dt + 0.5*a[t-1]*dt**2

        return x

    def return_r_dot(self, phi):
        
        return np.array([[-mp.sin(phi), -mp.cos(phi), 0], [mp.cos(phi), -mp.sin(phi), 0], [0, 0, 0]])

    def return_r(self, phi):
        
        return np.array([[mp.cos(phi), -mp.sin(phi), 0], [mp.sin(phi), mp.cos(phi), 0], [0, 0, 1]])

    def metpsr(self, tf, xf, alpha, phi_f):
        fl = self.calculate_vx(0, self.args.alpha_x_try, tf, alpha[2])[1][-1]
        fu = self.calculate_vx(abs(self.max_acc()), self.args.alpha_x_try, tf, alpha[2])[1][-1]


        a_x0 = self.max_acc()*(fl/(fl - fu))

        t, v, _ = self.calculate_vx(a_x0, self.args.alpha_x_try, tf, alpha[2])
        dt = t[1] - t[0]

        cxf = self.calculate_distance(v, dt)

        scaling_factor = cxf/xf

        alpha_x_scaled = self.args.alpha_x_try/scaling_factor
        
        a_x0_scaled = a_x0/scaling_factor

        if a_x0_scaled > abs(self.max_acc()):
            a_x0_scaled = abs(self.max_acc())

        t, v, a = self.calculate_vx(a_x0_scaled, alpha_x_scaled, tf, alpha[2])

        return t, v, a, alpha_x_scaled

    def get_lambda(self, x_dot, x_dot_dot, omega, R, R_dot):

        return (self.args.w2/self.args.k2)*(np.matmul(self.args.A, x_dot)) - (2*self.args.w1/(self.args.k2**2))*np.matmul(np.matmul(np.matmul(self.args.A, self.args.A), np.linalg.inv(self.args.Q)), x_dot_dot) -  (2*self.args.w1/(self.args.k2**2))*np.matmul(np.matmul(self.args.A, self.args.C), np.matmul(np.linalg.inv(self.args.Q), x_dot)) + ((2*self.args.w1*omega)/self.args.k2**2)*np.matmul(np.matmul(np.matmul(self.args.A, self.args.A), np.linalg.inv(self.args.Q)), np.matmul(R, np.matmul(R_dot.T, x_dot)))
    
    def return_control_vector(self, R, x_dot, lmda):

        return (1/(2*self.args.w1))*(self.args.w2*np.matmul(self.args.B, np.matmul(R.T, x_dot)) - np.matmul(np.matmul(self.args.D.T, R.T), np.matmul(np.linalg.inv(self.args.A.T), lmda)))
        
    def return_energy(self, u, x_dot, R):

        return self.args.w1*(np.sum(u*u)) - self.args.w2*np.sum(np.array(np.matmul(np.matmul(x_dot, R), self.args.B.T))[0]*u)

    def is_valid(self, phi, phi_f, x, xf, v):
        if abs(phi[-1] - phi_f) <= self.args.tol_angle and abs(x[-1]-xf) <= self.args.x_tol and abs(v[-1]) <= self.args.v_tol:
            return 1
        
        return 0