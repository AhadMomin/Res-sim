import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import yaml


class BuckleyLeverett(object):
    def __init__(self, inputs):

        if isinstance(inputs, str):
            with open(inputs) as f:
                self.inputs = yaml.load(f)
        else:
            self.inputs = inputs

        self.Sor = self.inputs["reservoir"]["oil"]["residual saturation"]
        self.Swc = self.inputs["reservoir"]["water"]["critical saturation"]
        self.nw = self.inputs["reservoir"]["water"]["corey-brooks exponent"]
        self.no = self.inputs["reservoir"]["oil"]["corey-brooks exponent"]
        self.krw_max = self.inputs["reservoir"]["water"]["max relative permeability"]
        self.kro_max = self.inputs["reservoir"]["oil"]["max relative permeability"]

        self.mu_o = self.inputs["fluid"]["oil"]["viscosity"]
        self.mu_w = self.inputs["fluid"]["water"]["viscosity"]

        self.Swi = self.inputs["initial conditions"]["water saturation"]

        self.step = 0.01

    # chain rule equations

    def water_rel_perm(self, S):
        return self.krw_max * ((S - self.Swc) / (1 - self.Sor - self.Swc)) ** self.nw

    def d_water_rel_perm_dS(self, S):
        return (
            self.nw
            * self.krw_max
            * ((-S + self.Swc) / (-1 + self.Sor + self.Swc)) ** self.nw
            / (S - self.Swc)
        )

    def oil_rel_perm(self, S):
        return (
            self.kro_max * (((1 - S) - self.Sor) / (1 - self.Sor - self.Swc)) ** self.no
        )

    def d_oil_rel_perm_dS(self, S):
        return (
            self.no
            * self.kro_max
            * ((-1 + S + self.Sor) / (-1 + self.Swc + self.Sor)) ** self.no
            / (-1 + S + self.Sor)
        )

    def fractional_flow(self, S):
        krw = self.water_rel_perm(S)
        kro = self.oil_rel_perm(S)
        return (krw / self.mu_w) / (krw / self.mu_w + kro / self.mu_o)

    def d_fractional_flow_dkrw(self, S):

        kro = self.oil_rel_perm(S)
        krw = self.water_rel_perm(S)
        mu_o = self.mu_o
        mu_w = self.mu_w

        return kro * mu_o * mu_w / (krw * mu_o + kro * mu_w) ** 2.0

    def d_fractional_flow_dkro(self, S):

        kro = self.oil_rel_perm(S)
        krw = self.water_rel_perm(S)
        mu_o = self.mu_o
        mu_w = self.mu_w

        return -krw * mu_o * mu_w / (krw * mu_o + kro * mu_w) ** 2.0

    def d_fractional_flow_dS(self, S):

        df_dkro = self.d_fractional_flow_dkro(S)
        df_dkrw = self.d_fractional_flow_dkrw(S)

        dkro_dS = self.d_oil_rel_perm_dS(S)
        dkrw_dS = self.d_water_rel_perm_dS(S)

        return df_dkro * dkro_dS + df_dkrw * dkrw_dS

    def compute_saturation_front(self):
        # Add you implimentation here.
        # brents method (combines bisection and newtons method (bisecction to get close to root then switches to newtons to finish iterations))
        f = lambda Swf: self.fractional_flow(Swf) / (
            Swf - self.Swi
        ) - self.d_fractional_flow_dS(Swf)
        return scipy.optimize.brenth(f, 0, 1)

    def compute_saturation_profile(self):

        Swi = self.inputs["initial conditions"]["water saturation"]

        S = np.arange(self.Swi + self.step, (1 - self.Swc), self.step)

        x = self.d_fractional_flow_dS(S)

        return (x, S)

    def plot_fractional_flow(self):

        S = np.arange(self.Swi + self.step, (1 - self.Swc), self.step)

        f = self.fractional_flow(S)

        plt.plot(S, f)
        plt.xlabel("$S_w$")
        plt.ylabel("$f$")

    def plot_full_saturation_profile(self):

        x, S = self.compute_saturation_profile()

        plt.plot(x, S)
        plt.ylabel("$S_w$")
        plt.xlabel("$x$")

    def plot_saturation_profile(self, t):

        x, S = self.compute_saturation_profile()

        Swf = self.compute_saturation_front()

        S1 = S[S > Swf]
        x1 = x[S > Swf] * t

        xD = self.d_fractional_flow_dS(Swf) * t

        S = np.concatenate((S1[::-1], np.array([Swf, self.Swi]), np.array([self.Swi])))
        x = np.concatenate((x1[::-1], np.array([xD, xD]), np.array([1.0])))

        plt.plot(x, S)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$S_w$")
