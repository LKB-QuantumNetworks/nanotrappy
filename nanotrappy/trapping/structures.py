import nanotrappy.utils.physicalunits as pu
from nanotrappy.utils.materials import SiO2, air
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve, root
from scipy.special import (
    kn,
    jv,
    j0,
    j1,
    k0,
    k1,
    jvp,
    kvp,
)  # using the j0, j1, k0, k1 is faster; jvp, kvp are derivatives


def jp(n, x):
    return 0.5 * (jv(n - 1, x) - jv(n + 1, x))


def kp(n, x):
    return -0.5 * (kn(n - 1, x) + kn(n + 1, x))


class Nanofiber:
    def __init__(self, material, cladding, radius=200e-9):
        self.material = material
        self.cladding = cladding
        self.radius = radius
        self.width = 2 * radius
        self.thickness = 2 * radius
        self.L = 0

    def compute_beta(self, lmbda, method="lm", start=1.4, maxrec=10, tolerance=1e-4, **kwargs):
        a = self.radius
        n1 = self.material.n
        n2 = self.cladding.n

        k00 = 2 * np.pi / lmbda

        h = lambda beta: np.sqrt((k00 ** 2) * (n1 ** 2) - beta ** 2)
        q = lambda beta: np.sqrt(beta ** 2 - (k00 ** 2) * (n2 ** 2))

        func = (
            lambda beta: (
                jp(1, h(beta) * a) / (h(beta) * a * j1(h(beta) * a))
                + kp(1, q(beta) * a) / (q(beta) * a * k1(q(beta) * a))
            )
            * (
                n1 ** 2 * jp(1, h(beta) * a) / (h(beta) * a * j1(h(beta) * a))
                + n2 ** 2 * kp(1, q(beta) * a) / (q(beta) * a * k1(q(beta) * a))
            )
            - (beta ** 2 / k00 ** 2) * (1 / (h(beta) * a) ** 2 + 1 / (q(beta) * a) ** 2) ** 2
        )

        for i in range(maxrec):
            beta_full_solution = root(func, (1 + (start - 1) / (2 ** i)) * k00, method=method, **kwargs)
            if abs(beta_full_solution.fun[0]) < tolerance:
                self.beta = beta_full_solution.x[0]
                return self.beta / k00, func, beta_full_solution
            else:
                continue
        self.beta = k00
        return self.beta / k00, func, beta_full_solution

    def electric_field_circular(self, x, y, z, lmbda, Ptot, sign):
        a = self.radius
        n1 = self.material.n
        n2 = self.cladding.n

        k00 = 2 * np.pi / lmbda
        omega = pu.cc * k00
        beta, _, _ = self.compute_beta(lmbda)
        beta = beta * k00

        ## Coordinates
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        ### Function definition
        h = np.sqrt((k00 ** 2) * (n1 ** 2) - beta ** 2)
        q = np.sqrt(beta ** 2 - (k00 ** 2) * (n2 ** 2))
        s = (1 / (h * a) ** 2 + 1 / (q * a) ** 2) / (
            jp(1, h * a) / (h * a * j1(h * a)) + kp(1, q * a) / (q * a * k1(q * a))
        )
        x = beta ** 2 / (2 * h * pu.μ0 * omega)
        y = beta ** 2 / (2 * q * pu.μ0 * omega)

        ### Normalization
        Tin = (1 + s) * (1 + (beta ** 2 / h ** 2) * (1 + s)) * (jv(2, h * a) ** 2 - j1(h * a) * jv(3, h * a)) + (
            1 - s
        ) * (1 + (beta ** 2 / h ** 2) * (1 - s)) * (j0(h * a) ** 2 + j1(h * a) ** 2)
        Tout = (1 + s) * (1 - (beta ** 2 / q ** 2) * (1 + s)) * (kn(2, q * a) ** 2 - k1(q * a) * kn(3, q * a)) + (
            1 - s
        ) * (1 - (beta ** 2 / q ** 2) * (1 - s)) * (k0(q * a) ** 2 - k1(q * a) ** 2)
        C = np.sqrt(4 * omega * pu.μ0 * Ptot / (np.pi * beta * a ** 2)) * (
            Tout + Tin * (k1(q * a) / j1(h * a)) ** 2
        ) ** (-1 / 2)

        ### Fields inside the core
        if 0 < r < a:
            Ez = C * j1(h * r) * (k1(q * a) / j1(h * a)) * np.exp(sign * 1j * theta) * np.exp(-1j * beta * z)
            Er = (
                C
                * (k1(q * a) / j1(h * a))
                * (1j * beta / (2 * h))
                * (jv(2, h * r) * (1 + s) - j0(h * r) * (1 - s))
                * np.exp(sign * 1j * theta)
                * np.exp(-1j * beta * z)
            )
            Etheta = (
                sign
                * C
                * (k1(q * a) / j1(h * a))
                * (beta / (2 * h))
                * (jv(2, h * r) * (1 + s) + j0(h * r) * (1 - s))
                * np.exp(sign * 1j * theta)
                * np.exp(-1j * beta * z)
            )

        ### Fields outside the core
        elif r >= a:
            Ez = C * k1(q * r) * np.exp(sign * 1j * theta) * np.exp(-1j * beta * z)
            Er = (
                -C
                * (1j * beta / (2 * q))
                * (kn(2, q * r) * (1 + s) + k0(q * r) * (1 - s))
                * np.exp(sign * 1j * theta)
                * np.exp(-1j * beta * z)
            )
            Etheta = (
                sign
                * C
                * (beta / (2 * q))
                * (-kn(2, q * r) * (1 + s) + k0(q * r) * (1 - s))
                * np.exp(sign * 1j * theta)
                * np.exp(-1j * beta * z)
            )

        return Ez, Er, Etheta

    def electric_field_linear(self, x, y, z, lmbda, Ptot, axis):
        a = self.radius
        n1 = self.material.n
        n2 = self.cladding.n

        k00 = 2 * np.pi / lmbda
        omega = pu.cc * k00
        if hasattr(self, "beta"):
            beta = self.beta
        else:
            beta, _, _ = self.compute_beta(lmbda) * k00

        ## Coordinates
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        ### Function definition
        h = np.sqrt((k00 ** 2) * (n1 ** 2) - beta ** 2)
        q = np.sqrt(beta ** 2 - (k00 ** 2) * (n2 ** 2))
        s = (1 / (h * a) ** 2 + 1 / (q * a) ** 2) / (
            jp(1, h * a) / (h * a * j1(h * a)) + kp(1, q * a) / (q * a * k1(q * a))
        )
        x = beta ** 2 / (2 * h * pu.μ0 * omega)
        y = beta ** 2 / (2 * q * pu.μ0 * omega)

        ### Normalization
        Tin = (1 + s) * (1 + (beta ** 2 / h ** 2) * (1 + s)) * (jv(2, h * a) ** 2 - j1(h * a) * jv(3, h * a)) + (
            1 - s
        ) * (1 + (beta ** 2 / h ** 2) * (1 - s)) * (j0(h * a) ** 2 + j1(h * a) ** 2)
        Tout = (1 + s) * (1 - (beta ** 2 / q ** 2) * (1 + s)) * (kn(2, q * a) ** 2 - k1(q * a) * kn(3, q * a)) + (
            1 - s
        ) * (1 - (beta ** 2 / q ** 2) * (1 - s)) * (k0(q * a) ** 2 - k1(q * a) ** 2)
        C = np.sqrt(4 * omega * pu.μ0 * Ptot / (np.pi * beta * a ** 2)) * (
            Tout + Tin * (k1(q * a) / j1(h * a)) ** 2
        ) ** (-1 / 2)

        ### Fields inside the core
        if 0 <= r < a:
            Ex = (
                C
                * (1j * beta / (h * np.sqrt(2)))
                * (k1(q * a) / j1(h * a))
                * (jv(2, h * r) * (1 + s) * np.cos(2 * theta - axis) - j0(h * r) * (1 - s) * np.cos(axis))
                * np.exp(-1j * beta * z)
            )
            Ey = (
                C
                * (1j * beta / (h * np.sqrt(2)))
                * (k1(q * a) / j1(h * a))
                * (jv(2, h * r) * (1 + s) * np.sin(2 * theta - axis) - j0(h * r) * (1 - s) * np.sin(axis))
                * np.exp(-1j * beta * z)
            )
            Ez = np.sqrt(2) * C * j1(h * r) * (k1(q * a) / j1(h * a)) * np.cos(theta - axis) * np.exp(-1j * beta * z)

        ### Fields outside the core
        elif r >= a:
            Ex = (
                -C
                * (1j * beta / (q * np.sqrt(2)))
                * (kn(2, q * r) * (1 + s) * np.cos(2 * theta - axis) + k0(q * r) * (1 - s) * np.cos(axis))
                * np.exp(-1j * beta * z)
            )
            Ey = (
                -C
                * (1j * beta / (q * np.sqrt(2)))
                * (kn(2, q * r) * (1 + s) * np.sin(2 * theta - axis) + k0(q * r) * (1 - s) * np.sin(axis))
                * np.exp(-1j * beta * z)
            )
            Ez = np.sqrt(2) * C * k1(q * r) * np.cos(theta - axis) * np.exp(-1j * beta * z)

        return Ex, Ey, Ez

    def compute_E_linear(self, x, y, z, lmbda, P, theta):
        # P is given in Watts

        E = np.zeros((3, len(x), len(y), len(z)), dtype="complex")
        for k in range(len(x)):
            for i in range(len(y)):
                for j in range(len(z)):
                    E[:, k, i, j] = self.electric_field_linear(x[k], y[i], z[j], lmbda, P, theta)
        return E


if __name__ == "__main__":
    nanof = Nanofiber(SiO2(), air())
    print(nanof.compute_E_linear([0],[0],[0],780e-9,1e-3,0))
