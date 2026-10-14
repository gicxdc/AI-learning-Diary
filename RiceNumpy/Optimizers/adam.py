import numpy as np


class Adam:

    def __init__(

        self,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8

    ):

        self.lr = lr

        self.beta1 = beta1
        self.beta2 = beta2

        self.eps = eps

        self.m = {}
        self.v = {}

        self.t = 0

    def step(self, parameters):

        self.t += 1

        for i, (param, grad) in enumerate(parameters):

            if grad is None:
                continue

            if i not in self.m:

                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)

            self.m[i] = (

                self.beta1 * self.m[i]
                + (1 - self.beta1) * grad

            )

            self.v[i] = (

                self.beta2 * self.v[i]
                + (1 - self.beta2)
                * (grad ** 2)

            )

            m_hat = (
                self.m[i]
                / (1 - self.beta1 ** self.t)
            )

            v_hat = (
                self.v[i]
                / (1 - self.beta2 ** self.t)
            )

            update = (

                self.lr
                * m_hat
                / (np.sqrt(v_hat) + self.eps)

            )

            param -= update