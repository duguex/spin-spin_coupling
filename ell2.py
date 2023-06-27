import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the principal stresses
s1 = 10 # MPa
s2 = 5 # MPa
s3 = 2 # MPa

# Define the semi-axes of the ellipsoid
a = s1 / 2
b = s2 / 2
c = s3 / 2

# Define the angles for plotting
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# Define the coordinates of the ellipsoid
x = a * np.outer(np.cos(u), np.sin(v))
y = b * np.outer(np.sin(u), np.sin(v))
z = c * np.outer(np.ones_like(u), np.cos(v))

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel('s3')
ax.set_title('Stress Ellipsoid')
plt.show()