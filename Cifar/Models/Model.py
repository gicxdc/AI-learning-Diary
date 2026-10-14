import numpy as np


class Model:

    def __init__(self):
        self.training = True
        self.layers = []   # FIX: must exist

    # =====================================
    # MODE
    # =====================================

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    # =====================================
    # PARAMS
    # =====================================

    def parameters(self):
        params = []

        for layer in self.layers:
            if hasattr(layer, "parameters"):
                layer_params = layer.parameters()
                if layer_params:
                    params.extend(layer_params)

        return params

    # =====================================
    # SAVE / LOAD (UNCHANGED BUT SAFE)
    # =====================================

    def save(self, path):
        params = {}
        layer_idx = 0

        for layer in self.layers:
            if hasattr(layer, "parameters"):
                layer_params = layer.parameters()
                for param_idx, (param, _) in enumerate(layer_params):
                    params[f"layer{layer_idx}_param{param_idx}"] = param
                layer_idx += 1

        np.savez(path, **params)
        print(f"Model saved to {path}")

    def load(self, path):
        data = np.load(path)
        layer_idx = 0

        for layer in self.layers:
            if hasattr(layer, "parameters"):
                layer_params = layer.parameters()
                for param_idx, (param, _) in enumerate(layer_params):
                    key = f"layer{layer_idx}_param{param_idx}"
                    param[:] = data[key]
                layer_idx += 1

        print(f"Model loaded from {path}")