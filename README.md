# Subspace

In linear algebra, a subspace is a subset of a vector space that is itself a vector space under the same operations of addition and scalar multiplication as the larger vector space. In other words, a subspace is a collection of vectors that is closed under vector addition and scalar multiplication.

To be more precise, a subset U of a vector space V is a subspace of V if and only if:

- The zero vector of V is in U.
- U is closed under vector addition: for any u, v in U, the sum u + v is also in U.
- U is closed under scalar multiplication: for any scalar c and u in U, the product c*u is also in U.

These conditions ensure that the subset U is itself a vector space, with the same operations of addition and scalar multiplication as the larger vector space V. Note that U may have a different dimension than V, but it must be contained within V.

# Linear Independence

In linear algebra, a set of vectors {v1, v2, ..., vn} is said to be linearly independent if no vector in the set can be expressed as a linear combination of the other vectors in the set. In other words, the only way to obtain the zero vector as a linear combination of the vectors is by setting all the coefficients to zero.

Formally, a set of vectors {v1, v2, ..., vn} in a vector space V over a field F is linearly independent if and only if the equation

```
c1v1 + c2v2 + ... + cn*vn = 0
```

holds only when c1 = c2 = ... = cn = 0, where c1, c2, ..., cn are scalars from the field F.

Intuitively, linearly independent vectors "point in different directions" and cannot be obtained as a linear combination of each other. If a set of vectors is not linearly independent, it is said to be linearly dependent.

It is worth noting that the concept of linear independence applies not only to finite sets of vectors, but also to infinite sets in some cases.

# Linear Transformations

In linear algebra, a linear transformation (also known as a linear map or linear operator) is a function that maps one vector space to another in a way that preserves certain algebraic properties. Specifically, a linear transformation T from a vector space V to a vector space W is a function that satisfies the following properties:

1. T(u + v) = T(u) + T(v) for all u, v in V (additivity)
2. T(cu) = cT(u) for all u in V and all scalars c (homogeneity)

In other words, a linear transformation preserves the properties of vector addition and scalar multiplication. Geometrically, a linear transformation can be thought of as a "stretching", "rotation", or "shearing" of the vector space, but the key property is that the transformation preserves the algebraic structure of the space.

Linear transformations are an important concept in linear algebra because they provide a way to study the properties of vector spaces and to solve systems of linear equations. Many common operations in linear algebra, such as matrix multiplication, can be thought of as linear transformations.

# Linear Transformations Script

In this example, we define a 2x2 matrix A that represents a linear transformation. We also define a 2-dimensional vector v that we want to transform. We then apply the transformation by computing the dot product of A and v using NumPy's dot function, and store the result in the variable Av.

Finally, we plot the original vector v and the transformed vector Av using Matplotlib's arrow function, and display the plot. The resulting plot should show the two vectors pointing in different directions, illustrating the effect of the linear transformation.

Code:
```
import numpy as np
import matplotlib.pyplot as plt

# Define a linear transformation matrix
A = np.array([[2, -1], [1, 3]])

# Define a vector to be transformed
v = np.array([1, 1])

# Apply the linear transformation to the vector
Av = np.dot(A, v)

# Plot the original and transformed vectors
fig, ax = plt.subplots()
ax.arrow(0, 0, v[0], v[1], head_width=0.2, head_length=0.2, fc='blue', ec='blue')
ax.arrow(0, 0, Av[0], Av[1], head_width=0.2, head_length=0.2, fc='red', ec='red')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_aspect('equal')
ax.grid()
plt.show()
```
