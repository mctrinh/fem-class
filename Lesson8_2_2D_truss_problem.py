# =========================================================================
#
#    Graduate Course: Finite Element Analysis - Spring 2022 @JBNU
#
#  ========================================================================
# 
#   Example 8.2: 2D Truss Problem
#
#   Last updated: 24/05/2022 by Minh-Chien Trinh (mctrinh@jbnu.ac.kr)
#
#   Copyright 2022 Minh-Chien Trinh. All rights reserved.
# 
# =========================================================================

import numpy as np


def Stiffness2DTruss(gDof, nElem, eNode, nNode, nCoord, xCoord, yCoord, EA):
    
    # Initialization
    Kmat = np.zeros((gDof,gDof))

    # Computation of the system stiffness matrix
    for i in range(nElem):
        # Element degree of freedom
        indice = eNode[i,:]

        # Element DOFs
        eDof = np.array([indice[0]*2-1, indice[0]*2, indice[1]*2-1, indice[1]*2])
        nDof = np.size(eDof)
        
        # Element length
        xa = xCoord[indice[1]-1] - xCoord[indice[0]-1]
        ya = yCoord[indice[1]-1] - xCoord[indice[0]-1]
        eLeng = np.sqrt(xa**2 + ya**2)
        
        # The contribution of the element to the stiffness matrix
        C = xa / eLeng
        S = ya / eLeng
        k1 = EA / eLeng * np.array([[C*C, C*S, -C*C, -C*S], [C*S, S*S, -C*S, -S*S], 
                                [-C*C, -C*S, C*C, C*S], [-C*S, -S*S, C*S, S*S]])
        
        # Assemble the contribution to the global stiffness matrix
        # Row index
        rIndex = np.zeros((nDof,nDof), dtype=int)
        rIndex[0:4,0] = eDof - 1 
        rIndex[0:4,1] = eDof - 1
        rIndex[0:4,2] = eDof - 1
        rIndex[0:4,3] = eDof - 1
        # Column index
        cIndex = np.zeros((nDof,nDof), dtype=int)
        cIndex[0,0:4] = eDof - 1
        cIndex[1,0:4] = eDof - 1
        cIndex[2,0:4] = eDof - 1
        cIndex[3,0:4] = eDof - 1
        # Assemble
        Kmat[rIndex,cIndex] = Kmat[rIndex,cIndex] + k1
        
    return Kmat


def Stresses2DTruss(nElem, eNode, xCoord, yCoord, Umat, E):

    # Initialization
    sigma = np.zeros((nElem, 1))

    for i in range(nElem):
        # Element degree of freedom
        indice = eNode[i,:]

        # Element DOFs
        eDof = np.array([indice[0]*2-1, indice[0]*2, indice[1]*2-1, indice[1]*2])
        nDof = np.size(eDof)
        
        # Element length
        xa = xCoord[indice[1]-1] - xCoord[indice[0]-1]
        ya = yCoord[indice[1]-1] - xCoord[indice[0]-1]
        eLeng = np.sqrt(xa**2 + ya**2)
        
        C = xa / eLeng
        S = ya / eLeng
        
        # Element stress
        sigma[i,0] = E / eLeng * np.matmul(np.array([-C, -S, C, S]), 
                        np.array([Umat[eDof[0]-1,0], Umat[eDof[1]-1,0], Umat[eDof[2]-1,0], Umat[eDof[3]-1,0]]))

    return sigma


# Given paramerers
E = 30e6; A = 2; EA = E*A

# Number of elements
nElem = 3

# Number of nodes
nNode = 4

# Element nodes
eNode = np.array([[1,2], [1,3], [1,4]], dtype=int)

# Node coordinates
nCoord = np.array([[0,0], [0,120], [120,120], [120,0]])
xCoord = nCoord[:,0]
yCoord = nCoord[:,1]

# Total number of degree of freedom
gDof  = 2 * nNode

# Initialization
Umat = np.zeros((gDof,1))              # Displacement vector
Fmat = np.zeros((gDof,1))              # Force vector
Kmat = np.zeros((gDof,gDof))           # Stiffness matrix

# Applied load at node 2
Fmat[1,0] = -10000

# Computation of the system stiffness matrix
Kmat = Stiffness2DTruss(gDof, nElem, eNode, nNode, nCoord, xCoord, yCoord, EA)
print("---------------------------------------------------------------------")
print("Stiffness matrix (Kmat): \n\n", Kmat)
print("---------------------------------------------------------------------")

# Apply boundary condition
# Fix/prescribed degree of freedom
fixDof = np.array([[2],[3],[4],[5],[6],[7]])
# Free/active degree of freedom
activeDof = np.setdiff1d(np.arange(0,gDof,1), fixDof)       # 02 active Dofs

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
print("Displacement vector (Umat): \n\n", Umat)
print("-------------------------------------------------")

# Force vector
Fmat = np.dot(Kmat,Umat)
print("Force vector (Fmat): \n\n", Fmat)
print("-------------------------------------------------")

# Reactions
Reaction = np.zeros(np.size(fixDof))
Reaction = Fmat[fixDof,0]
print("Reactions: \n\n", Reaction)
print("-------------------------------------------------")

# Stresses of elements
sigma = np.zeros((nElem, 1))
sigma = Stresses2DTruss(nElem, eNode, xCoord, yCoord, Umat, E)
print("Stresses: \n\n", sigma)
print("-------------------------------------------------")


# -------------------------------------------------------------------------
#
# For students:
# 
# 1. Plot the intact and deformed shapes of the system
# 
# -------------------------------------------------------------------------

# Add your code from there ...
