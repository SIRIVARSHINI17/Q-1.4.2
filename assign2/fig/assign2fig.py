import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping

# Orthogonal matrix
omat = np.array([[0, 1], [-1, 0]])
A = np.array([1, -1])
B = np.array([-4, 6])
C = np.array([-3, -5])
O = np.array([-53/12,5/12])
D = (B + C) / 2
E = (C + A) / 2
F = (A + B) / 2

def dir_vec(A, B):
    return B - A

def norm_vec(A, B):
    return omat @ dir_vec(A, B) 

def line_gen(A, B):
    len = 10
    dim = A.shape[0]
    x_AB = np.zeros((dim, len))
    lam_1 = np.linspace(0, 1, len)
    for i in range(len):
        temp1 = A + lam_1[i] * (B - A)
        x_AB[:, i] = temp1.T
    return x_AB


def perpendicular_bisector(A, B):
    mid_point = (A + B) / 2
    direction_vector = B - A
    perpendicular_direction = np.array([-direction_vector[1], direction_vector[0]])
   
    # Define a point on the perpendicular bisector line
    C = mid_point + perpendicular_direction/np.linalg.norm(perpendicular_direction)
    D = mid_point - perpendicular_direction

    # Generate points along the perpendicular bisector line
    len = 10  # Number of points
    dim = A.shape[0]
    x_perpendicular = np.zeros((dim, len))
    lam = np.linspace(-1, 1, len)  # Adjust the range if needed

    for i in range(len):
        temp = C + lam[i] * (D-C)
        x_perpendicular[:, i] = temp.T

    return x_perpendicular



pb_AB = perpendicular_bisector(A, B)
pb_AC = perpendicular_bisector(A, C)

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(pb_AB[0,:],pb_AB[1,:])
plt.plot(pb_AC[0,:],pb_AC[1,:])
plt.scatter(-53/12, 5/12, color='red', marker='o', label='circumcenter(O)')

A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D = D.reshape(-1,1)
E = E.reshape(-1,1)
F = F.reshape(-1,1)
tri_coords = np.block([[A,B,C,D,E,F]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')


plt.show()

