# optim/adam.py

import numpy as np
from optim.optimizer import Optimizer


class Adam(Optimizer):

    def __init__(self,
                 layers,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8):

        super().__init__(layers, lr)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0

        self.m = {}
        self.v = {}

        self._init_states()

    def _init_states(self):

        for layer in self.layers:

            for name in vars(layer):

                if name in ["W", "b", "gamma", "beta"]:

                    param = getattr(layer, name)

                    self.m[id(param)] = np.zeros_like(param)
                    self.v[id(param)] = np.zeros_like(param)

    def step(self):

        self.t += 1

        for layer in self.layers:

            for param_name, grad_name in [
                ("W", "dW"),
                ("b", "db"),
                ("gamma", "dgamma"),
                ("beta", "dbeta")
            ]:

                if hasattr(layer, param_name):

                    param = getattr(layer, param_name)
                    grad = getattr(layer, grad_name)

                    pid = id(param)

                    # =========================
                    # moments
                    # =========================
                    self.m[pid] = (
                        self.beta1 * self.m[pid]
                        + (1 - self.beta1) * grad
                    )

                    self.v[pid] = (
                        self.beta2 * self.v[pid]
                        + (1 - self.beta2) * (grad ** 2)
                    )

                    # =========================
                    # bias correction
                    # =========================
                    m_hat = self.m[pid] / (
                        1 - self.beta1 ** self.t
                    )

                    v_hat = self.v[pid] / (
                        1 - self.beta2 ** self.t
                    )

                    # =========================
                    # update
                    # =========================
                    param -= (
                        self.lr
                        * m_hat
                        / (np.sqrt(v_hat) + self.eps)
                    )