import numpy as np


class AdamW:

    def __init__(
        self,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-4
    ):

        self.lr = lr

        self.beta1 = beta1
        self.beta2 = beta2

        self.eps = eps

        self.weight_decay = weight_decay

        # momentum buffers
        self.m = {}
        self.v = {}

        # timestep
        self.t = 0

    # =====================================
    # STEP
    # =====================================

    def step(self, parameters):

        self.t += 1

        for param, grad in parameters:

            if grad is None:
                continue

            # ---------------------------------
            # SAFE PARAMETER KEY
            # ---------------------------------

            key = id(param)

            # ---------------------------------
            # INIT MOMENTUM BUFFERS
            # ---------------------------------

            if key not in self.m:

                self.m[key] = np.zeros_like(param)

                self.v[key] = np.zeros_like(param)

            # ---------------------------------
            # UPDATE MOMENTS
            # ---------------------------------

            self.m[key] = (
                self.beta1 * self.m[key]
                + (1 - self.beta1) * grad
            )

            self.v[key] = (
                self.beta2 * self.v[key]
                + (1 - self.beta2) * (grad ** 2)
            )

            # ---------------------------------
            # BIAS CORRECTION
            # ---------------------------------

            m_hat = (
                self.m[key]
                / (1 - self.beta1 ** self.t)
            )

            v_hat = (
                self.v[key]
                / (1 - self.beta2 ** self.t)
            )

            # ---------------------------------
            # ADAM UPDATE
            # ---------------------------------

            update = m_hat / (
                np.sqrt(v_hat) + self.eps
            )

            # ---------------------------------
            # PARAMETER UPDATE
            # ---------------------------------

            param[:] -= self.lr * update

            # ---------------------------------
            # DECOUPLED WEIGHT DECAY
            # ---------------------------------

            # skip weight decay for:
            # - biases
            # - BatchNorm gamma/beta shaped like (1, C, 1, 1)

            is_batchnorm_param = (
                param.ndim == 4
                and param.shape[0] == 1
                and param.shape[2] == 1
                and param.shape[3] == 1
            )

            if param.ndim > 1 and not is_batchnorm_param:

                param[:] -= (
                    self.lr
                    * self.weight_decay
                    * param
                )
