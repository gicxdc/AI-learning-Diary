from Backend import xp as np

class CrossEntropyLoss:

    def __init__(self, label_smoothing=0.0):
        self.probs = None
        self.labels = None
        self.label_smoothing = label_smoothing

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x)
        return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-9)

    def forward(self, logits, labels):

        self.labels = labels
        self.probs = self.softmax(logits)

        B = logits.shape[0]
        num_classes = logits.shape[1]

        log_probs = -np.log(self.probs + 1e-9)

        if labels.ndim == 2:
            target = labels.astype(np.float32, copy=False)

            if self.label_smoothing > 0:
                smooth = self.label_smoothing
                target = (1.0 - smooth) * target + smooth / num_classes

            self.labels = target
            return np.mean(np.sum(target * log_probs, axis=1))

        if self.label_smoothing > 0:
            smooth = self.label_smoothing
            target = np.full_like(self.probs, smooth / num_classes)
            target[np.arange(B), labels] += 1.0 - smooth
            self.labels = target
            return np.mean(np.sum(target * log_probs, axis=1))

        correct_log_probs = log_probs[np.arange(B), labels]

        return np.mean(correct_log_probs)

    def backward(self):

        B = self.probs.shape[0]

        dout = self.probs.copy()

        if self.labels.ndim == 2:
            target = self.labels
            dout -= target
        else:
            dout[np.arange(B), self.labels] -= 1

        return dout / B

