# =========================================================================
#
#    Graduate Course: Finite Element Analysis - Spring 2022 @JBNU
#
#  ========================================================================
# 
#   Example 9: Euler Bernoulli Beam
#
#   Last updated: 07/06/2022 by Minh-Chien Trinh (mctrinh@jbnu.ac.kr)
#
#   Copyright 2022 Minh-Chien Trinh. All rights reserved.
# 
# =========================================================================

import numpy as np


def StiffnessBernoulli(gDof, nElem, eNode, nNode, nCoord, EI, P):
    
    # Initialization
    Fmat = np.zeros((gDof,1))
    Kmat = np.zeros((gDof,gDof))

    # Computation of the system stiffness matrix and force vector
    for i in range(nElem):
        # Element degree of freedom
        indice = eNode[i,:]

        # Element DOFs
        eDof = np.array([2*(indice[0]-1)+1, 2*(indice[1]-1), 2*(indice[1]-1)+1, 2*(indice[1]-1)+2])
        nDof = np.size(eDof)
        
        # Element length
        eLeng = nCoord[indice[1]-1] - nCoord[indice[0]-1]
        
        # Stiffness matrix of the element
        k1 = EI / eLeng**3 * np.array([[12, 6*eLeng, -12, -6*eLeng],
                                        [6*eLeng, 4*eLeng**2, -6*eLeng, 2*eLeng**2], 
                                        [-12, -6*eLeng, 12, -6*eLeng],
                                        [6*eLeng, 2*eLeng**2, -6*eLeng, 4*eLeng**2]])
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
        # Assemble the element stiffness matrix to the global stiffness matrix
        Kmat[rIndex,cIndex] = Kmat[rIndex,cIndex] + k1

        # Force vector of the element
        f1 = np.array([[P*eLeng/2],
                        [P*eLeng**2/12],
                        [P*eLeng/2],
                        [-P*eLeng**2/12]])
        # Row index
        rInd = np.zeros((nDof,1), dtype=int)
        rInd[0:4,0] = eDof - 1
        # Column index
        cInd = np.zeros((nDof,1), dtype=int)
        cInd[0:4,0] = 0
        # Assemble the element force vector to the global force vector
        Fmat[rInd,cInd] = Fmat[rInd,cInd] + f1

    return Kmat, Fmat


# Given paramerers
E = 1; I = 1; EI = E*I; L = 1

# Number of elements
nElem = 2

# Number of nodes
nNode = nElem + 1

# Element nodes
eNode = np.array([[1,2], [2,3]], dtype=int)

# Node coordinates
nCoord = np.linspace(0, L, nNode)
print(nCoord)

# Distributed load
P = -1

# Total number of degree of freedom
gDof  = 2 * nNode

# Initialization
Umat = np.zeros((gDof,1))              # Displacement vector
Fmat = np.zeros((gDof,1))              # Force vector
Kmat = np.zeros((gDof,gDof))           # Stiffness matrix

# Computation of the system stiffness matrix
Kmat, Fmat = StiffnessBernoulli(gDof, nElem, eNode, nNode, nCoord, EI, P)
print("---------------------------------------------------------------------")
print("Stiffness matrix (Kmat): \n\n", Kmat)
print("---------------------------------------------------------------------")
print("Force matrix (Fmat): \n\n", Fmat)
print("---------------------------------------------------------------------")

# Apply boundary condition

# Case 1: Clamped at x=0 & x=L
# ------------------------
# Fix/prescribed degree of freedom
fixDof = np.array([[0],[1], [2*nElem], [2*nElem+1]])
# Free/active degree of freedom
activeDof = np.setdiff1d(np.arange(0,gDof,1), fixDof)
nActive = np.size(activeDof)

# Case 2: Clamped at x = 0, free at x=L
# ------------------------
# .... practice for students ....

# Case 3: Simply supported
# ----------------------------
# .... practice for students ....

# Solution
# Indexing of activeDof
iIndex = np.zeros((nActive, nActive), dtype=int)
jIndex = np.zeros((nActive, nActive), dtype=int)
for i in range(nActive):
    iIndex[0:nActive,i] = activeDof
    jIndex[i,0:nActive] = activeDof

# Evaluate displacement at activeDof
Umat[activeDof,0] = np.dot(np.linalg.inv(Kmat[iIndex,jIndex]),Fmat[activeDof,0])
print("Displacement vector (Umat): \n\n", Umat)
print("-------------------------------------------------")

# -------------------------------------------------------------------------
#
# For students:
# 
# 1. Modify the code for different number of elements
# 2. Modify the code for different boundary conditions
# 3. Plot the deformed shape of the beam
# 
# -------------------------------------------------------------------------