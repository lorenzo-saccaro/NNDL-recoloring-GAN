# plot an image with matplotlib and PIl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = Image.open('C:\\Users\\loren\\Datasets\\coco2017\\train2017\\000000000009.jpg')
plt.imshow(img)
plt.show()
img_resized = img.resize((512, 512))
plt.imshow(img_resized)
plt.show()