import numpy as np

print("starting...")

img_data = np.load("images.npz")
mask_data = np.load("masks.npz")

images = img_data["images"]
masks = mask_data["masks"]

print("images:", images.shape)
print("masks:", masks.shape)



