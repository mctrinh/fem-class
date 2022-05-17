# =========================================================================
#
#    Graduate Course: Finite Element Analysis - Spring 2022 @JBNU
#
#  ========================================================================
# 
#   Lesson 8.1: Isotropic Bar under Uniform Load
#
#   Last updated: 17/05/2022 by Minh-Chien Trinh (mctrinh@jbnu.ac.kr)
#
#   Copyright 2022 Minh-Chien Trinh. All rights reserved.
# 
# =========================================================================

import numpy as np


def shapeFunction1D(xi):
    
    # xi is 1D natural coordinate within the range (-1, 1)

    # Shape Function
    shapeFunc = np.array([[(1-xi)/2], [(1+xi)/2]])
    
    # Natural derivative of shapeFunc with respect to xi
    natDeriv  = np.array([[1/2], [-1/2]])

    return shapeFunc, natDeriv


# Given paramerers
E = 30e6; A = 1; EA = E*A; L = 90; p = 50

# Number of elements
nElem = 3

# Node coordinates
nCoord = np.linspace(0, L, nElem + 1)

# Number of nodes
nNode = np.size(nCoord)

# Element nodes
eNode = np.zeros((nElem, 2), dtype=int)
eNode[:, 0] = np.arange(1, nNode, dtype=int)
eNode[:, 1] = np.arange(1, nNode, dtype=int) + 1

# Initialization
Umat = np.zeros((nNode,1))              # Displacement vector
Fmat = np.zeros((nNode,1))              # Force vector
Kmat = np.zeros((nNode,nNode))          # Stiffness matrix

# Applied load at node 2
Fmat[1,0] = 10                          # Python index from 0

for i in range(nElem):
    
    # Element degree of freedom
    eDof = eNode[i,:]

    # Number of element Dofs
    nDof = np.size(eDof)

    # Element length
    eLeng = nCoord[eDof[1]-1] - nCoord[eDof[0]-1]

    # Determinant of Jacobian
    detJacob = eLeng / 2

    # Inverse of Jacobian
    invJacob = 1 / detJacob

    # Using one Gauss point at center (xi=0, weight W=2)
    [shapeFunc, natDeriv] = shapeFunction1D(0.0)

    # Derivative of shapeFunc w.r.t global/system coordinate
    Xderiv = natDeriv * invJacob

    # B matrix (dN/dx)
    B = Xderiv

    # Row index
    rIndex = np.zeros((nDof,nDof), dtype=int)
    rIndex[0:2,0] = eDof - 1 
    rIndex[0:2,1] = eDof - 1

    # Column index
    cIndex = np.zeros((nDof,nDof), dtype=int)
    cIndex[0,0:2] = eDof - 1
    cIndex[1,0:2] = eDof - 1

    # The contribution of the element to the stiffness matrix
    Kmat[rIndex,cIndex] = Kmat[rIndex,cIndex] + 2 * detJacob * EA * np.matmul(B, B.T)

    # The force vector
    rInd = np.zeros((nDof,1), dtype=int)
    rInd[0:2,0] = eDof - 1
    cInd = np.zeros((nDof,1), dtype=int)
    cInd[0:2,0] = 0
    Fmat[rInd,cInd] = Fmat[rInd,cInd] + 2 * shapeFunc * p * detJacob

# Apply boundary condition
# Fix/prescribed degree of freedom
fixDof = np.array([[0], [3]])
# Free/active degree of freedom
activeDof = np.setdiff1d(np.arange(0,nNode,1), fixDof)

# Solution
# Indexing of activeDof
iIndex = np.zeros((np.size(activeDof),np.size(activeDof)), dtype=int)
iIndex[0:2,0] = activeDof
iIndex[0:2,1] = activeDof
jIndex = np.zeros((np.size(activeDof),np.size(activeDof)), dtype=int)
jIndex[0,0:2] = activeDof
jIndex[1,0:2] = activeDof
# Evaluate displacement at activeDof
Umat[activeDof,0] = np.dot(np.linalg.inv(Kmat[iIndex,jIndex]),Fmat[activeDof,0])
print(Umat)

# Force vector
Fmat = np.dot(Kmat,Umat)
print(Fmat)

# Reactions
Reaction = np.zeros(np.size(fixDof))
Reaction = Fmat[fixDof,0]
print(Reaction)

# Stresses of elements
sigma = np.zeros((nElem, 1))

for e in range(nElem):

    # Element degree of freedom
    eDof = eNode[e,:]

    # Number of element Dofs
    nDof = np.size(eDof)

    # Element length
    eLeng = nCoord[eDof[1]-1] - nCoord[eDof[0]-1]

    # Stress of element e
    sigma[e, 0] = E / eLeng * np.matmul(np.array([-1, 1]), np.array([Umat[eDof[0]-1,0], Umat[eDof[1]-1,0]]))

print(sigma)


# -------------------------------------------------------------------------
#
# For students:
# 
# 1. Plot the deform shape (x: length, y: displacement)
# 
# 2. Plot the stress distribution along the system (x: length, y: stress)
# 
# Note: Should plot both FEM solution and exact solution
#
# -------------------------------------------------------------------------