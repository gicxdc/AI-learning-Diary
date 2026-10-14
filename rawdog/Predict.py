import numpy as np
import matplotlib.pyplot as plt
from Test import forwardPass,read_idx

data = np.load("model.npz")

filters = data["filters"]
convBias = data["convBias"]
weight = data["weight"]
denseBias = data["denseBias"]

x_test = read_idx(r'D:\TeraBoxDownload\archive\t10k-images.idx3-ubyte')
y_test = read_idx(r'D:\TeraBoxDownload\archive\t10k-labels.idx1-ubyte')

perm = np.random.permutation(len(x_test))
x_shuffled = x_test[perm]
y_shuffled = y_test[perm]
  

# plt.figure(figsize=(10,5))

# for i in range(10):
#     image = x_shuffled[i] / 255.0
#     image_input = image.reshape(1, 28, 28)

#     probs,_  = forwardPass(image_input, filters, weight, convBias, denseBias)
    

#     print("Probabilities:", np.max(probs))

#     pred = np.argmax(probs)
#     print(f"Pred: {pred}, Confidence: {np.max(probs):.3f}")
#     print("")

#     plt.subplot(2, 5, i+1)
#     plt.imshow(image, cmap='gray')
#     plt.title(f"P:{pred} T:{y_shuffled[i]}")
#     plt.axis('off')

# plt.tight_layout()
# plt.show()

# correct = 0

# 

correct = 0

for i in range(len(x_test)):
    image = x_test[i] / 255.0
    image = image.reshape(1, 28, 28)

    probs, _ = forwardPass(image, filters, weight, convBias, denseBias)
    pred = np.argmax(probs)

    if np.max(probs) < 0.6:
        print("Uncertain:", pred, "True:", y_test[i])

    if pred == y_test[i]:
        correct += 1

    if i%100==0:
        print("Current Progress: ",i,"/",len(x_test))

print("Test Accuracy:", correct / len(x_test))