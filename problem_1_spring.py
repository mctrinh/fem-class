# =========================================================================
#
#    Graduate Course: Finite Element Analysis - Spring 2022 @JBNU
#
#  ========================================================================
# 
#   Problem 1: A spring system
#
#   Last updated: 14/03/2022 by Minh-Chien Trinh (mctrinh@jbnu.ac.kr)
#
#   Copyright 2022 Minh-Chien Trinh. All rights reserved.
# 
# =========================================================================

import numpy as np

# Element nodes
eNode = np.array([[1,2], [2,3], [2,4]])

# Number of elements
nElem = np.size(eNode,0)                # axis=0 returns the number of row

# Number of nodes
nNode = 4

# Initialization
Umat = np.zeros((nNode,1))              # Displacement vector
Fmat = np.zeros((nNode,1))              # Force vector
Kmat = np.zeros((nNode,nNode))          # Stiffness matrix
# Applied load at node 2
Fmat[1,0] = 10                          # Python index from 0

for i in range(nElem):
    # Element degree of freedom
    eDof = eNode[i,:]

    # Row index
    rIndex = np.zeros((np.size(eDof),np.size(eDof)), dtype=int)
    rIndex[0:2,0] = eDof - 1 
    rIndex[0:2,1] = eDof - 1

    # Column index
    cIndex = np.zeros((np.size(eDof),np.size(eDof)), dtype=int)
    cIndex[0,0:2] = eDof - 1
    cIndex[1,0:2] = eDof - 1

    # The contribution of the element to the stiffness matrix
    Kmat[rIndex,cIndex] = Kmat[rIndex,cIndex] + np.array([[1,-1], [-1, 1]])

# Apply boundary condition
# Fix/prescribed degree of freedom
fixDof = np.array([[0], [2], [3]])
# Free/active degree of freedom
activeDof = np.setdiff1d(np.arange(0,nNode,1), fixDof)

# Solution
# Indexing of activeDof
iIndex = np.zeros((1,1), dtype=int)
iIndex[0,0] = activeDof
jIndex = np.zeros((1,1), dtype=int)
jIndex[0,0] = activeDof
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
