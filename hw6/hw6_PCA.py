import numpy as np
from skimage import io
import sys

print("reading face")
row_img = []
for i in range(415):
    row_img.append(io.imread(str(sys.argv[1])+"/"+str(i)+".jpg").reshape(600*600*3)*1.0)
row_img = np.array(row_img)

print("making average face")
average_img = []
for i in range(600*600*3):
    average_img.append(np.sum(row_img[:, i])*1.0/415.0)
average_img = np.array(average_img)

print("computing SVD")
img = []
for i in range(415):
    row_img[i] -= average_img
    img.append(row_img[i].reshape(600*600*3))
img = np.array(img)
U, S, V = np.linalg.svd(img, full_matrices=False)

print("reconstruct")
test_picture = io.imread(str(sys.argv[1])+"/"+sys.argv[2]).reshape(600*600*3)*1.0-average_img.reshape(600*600*3)
projection = np.dot(test_picture, V.T)
projection_4 = np.zeros(415)
projection_4[0:4] = projection[0:4]
reconstruct_img = np.dot(projection_4, V) + average_img
reconstruct_img -= np.min(reconstruct_img)
reconstruct_img /= np.max(reconstruct_img)
reconstruct_img = (reconstruct_img * 255).astype(np.uint8)
reconstruct_img = reconstruct_img.reshape((600, 600, 3))
io.imsave("reconstruction.jpg", reconstruct_img)
