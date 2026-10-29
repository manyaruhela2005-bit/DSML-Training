import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

mnist= fetch_openml("mnist_784", version=1, as_frame=False)

X=mnist.data.astype("float32")
y=mnist.target.astype(int)

print("dataset shape", X.shape)
print("labels shape", y.shape)

fig,axes=plt.subplots(2,5, figsize=(10,4))

for ax, image, label in zip(axes.ravel(), X[:10], y[:10]):
    ax.imshow(image.reshape(28,28), cmap="gray")
    ax.set_title(f"label: {label}")
    ax.axis("off")

plt.tight_layout()
plt.show()

X=X/255.0 #preprocessing the data

print("minimum pixel value", X.min())
print("maximum pixel value", y.min())