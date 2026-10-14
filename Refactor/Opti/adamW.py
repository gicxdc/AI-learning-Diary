# optim/adamw.py

import numpy as np
from optim.adam import Adam


class AdamW(Adam):

    def __init__(self,
                 layers,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8,
                 weight_decay=0.0005):

        super().__init__(
            layers,
            lr,
            beta1,
            beta2,
            eps
        )

        self.weight_decay = weight_decay

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

                    m_hat = self.m[pid] / (
                        1 - self.beta1 ** self.t
                    )

                    v_hat = self.v[pid] / (
                        1 - self.beta2 ** self.t
                    )

                    # =========================
                    # adam update
                    # =========================
                    adam_update = (
                        m_hat
                        / (np.sqrt(v_hat) + self.eps)
                    )

                    # =========================
                    # decoupled weight decay
                    # =========================
                    decay_update = (
                        self.weight_decay * param
                    )

                    param -= self.lr * (
                        adam_update + decay_update
                    )