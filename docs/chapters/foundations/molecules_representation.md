# Molecular Representations for Machine Learning

Representing molecules in a format suitable for machine learning algorithms is a critical step in computational drug discovery. In this chapter, we'll explore different ways to encode molecular structures into numerical or vector representations that can be used in Bayesian optimization and active learning approaches.

## Why Molecular Representations Matter

The choice of molecular representation can significantly impact:

- The performance of machine learning models
- The ability to capture relevant chemical information
- The computational efficiency of the optimization process
- The interpretability of the results

## Common Molecular Representations

### 1. Strings

Molecules can be represneted as strings. The **Simplified Molecular-Input Line-Entry System (SMILES)** is a linear text notation that is commonly used to represent molecular structures. There are other alternative string-based representations, such as **InChI**,  **InChIKey**, and **Self-Referencing Embedded Strings (SELFIES)**.

Example: Aspirin is represented as `CC(=O)OC1=CC=CC=C1C(=O)O`

```python
from rdkit import Chem
from rdkit.Chem import Draw

# Create an RDKit molecule from SMILES
aspirin = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')

# Display the molecule
Draw.MolToImage(aspirin)
```

**Advantages:**
- Compact and human-readable
- Widely used in chemical databases
- Easy to store and share

**Limitations:**
- Multiple valid SMILES can represent the same molecule
- Doesn't explicitly encode 3D information

### 2. Molecular Fingerprints

Fingerprints are bit vectors or count vectors that encode the presence or frequency of specific structural features in a molecule.

#### a. Morgan/Circular Fingerprints (ECFP)

Based on the Morgan algorithm, these fingerprints encode circular environments around each atom.

```python
from rdkit.Chem import AllChem

# Generate Morgan fingerprint (radius 2, 1024 bits)
morgan_fp = AllChem.GetMorganFingerprintAsBitVect(aspirin, 2, nBits=1024)
```

#### b. MACCS Keys

A set of 166 predefined structural keys.

```python
from rdkit.Chem import MACCSkeys

# Generate MACCS fingerprint
maccs_fp = MACCSkeys.GenMACCSKeys(aspirin)
```

#### c. Atom Pair Fingerprints

Encode pairs of atoms and the shortest path between them.

```python
from rdkit.Chem.AtomPairs import Pairs

# Generate atom pair fingerprint
ap_fp = Pairs.GetAtomPairFingerprint(aspirin)
```

**Advantages:**
- Fixed length, suitable for many ML algorithms
- Capture important chemical substructures
- Computationally efficient

**Limitations:**
- Loss of structural information
- May not capture complex 3D relationships
- Different fingerprints capture different aspects of chemistry

### 3. Molecular Descriptors

Numerical values that describe various physical, chemical, or topological properties of molecules.

```python
from rdkit.Chem import Descriptors

# Calculate some common descriptors
# Molecular Weight - Mass of the molecule in g/mol
mw = Descriptors.MolWt(aspirin) 

# LogP - Octanol-water partition coefficient, measure of lipophilicity
logp = Descriptors.MolLogP(aspirin)

# TPSA - Topological Polar Surface Area, sum of surface contributions of polar atoms
tpsa = Descriptors.TPSA(aspirin)

# Number of H-bond acceptors (typically O and N atoms)
hba = Descriptors.NumHAcceptors(aspirin)

# Number of H-bond donors (typically OH and NH groups) 
hbd = Descriptors.NumHDonors(aspirin)

# Print all calculated descriptors
print(f"Molecular Weight: {mw}")
print(f"LogP: {logp}")
print(f"TPSA: {tpsa}")
print(f"H-Bond Acceptors: {hba}")
print(f"H-Bond Donors: {hbd}")
```

**Advantages:**
- Interpretable physicochemical properties
- Directly related to drug-like properties
- Can incorporate expert knowledge

**Limitations:**
- May require feature selection
- Different descriptors may be correlated
- May not capture all relevant structural information

### 4. Graph-Based Representations

Molecules are naturally represented as graphs, with atoms as nodes and bonds as edges.

```python
import networkx as nx

# Convert RDKit molecule to NetworkX graph
def mol_to_nx(mol):
    # Initialize an empty undirected graph
    G = nx.Graph()
    
    # Add nodes (atoms) with their properties
    for atom in mol.GetAtoms():
        # Each atom becomes a node with properties like:
        # - atomic number (element type)
        # - formal charge (+1, -1, etc)
        # - chirality (R/S configuration) 
        # - hybridization (sp, sp2, sp3)
        # - aromaticity (is the atom part of an aromatic ring)
        G.add_node(atom.GetIdx(),
                  atomic_num=atom.GetAtomicNum(),
                  formal_charge=atom.GetFormalCharge(),
                  chiral_tag=atom.GetChiralTag(),
                  hybridization=atom.GetHybridization(),
                  is_aromatic=atom.GetIsAromatic())
    
    # Add edges (bonds) between atoms
    for bond in mol.GetBonds():
        # Each bond connects two atoms and has a type
        # (single, double, triple, aromatic)
        G.add_edge(bond.GetBeginAtomIdx(),
                  bond.GetEndAtomIdx(),
                  bond_type=bond.GetBondType())
    
    return G

# Create a graph representation of aspirin
# This converts the RDKit molecule into a NetworkX graph object
# where nodes are atoms and edges are bonds
aspirin_graph = mol_to_nx(aspirin)
```

**Advantages:**
- Preserves molecular topology
- No information loss from the 2D structure
- Suitable for graph neural networks

**Limitations:**
- More complex to process
- Variable size, requiring special handling
- May not explicitly encode 3D information

### 5. 3D Representations

Capture the three-dimensional structure of molecules, which is often crucial for biological activity.

#### a. Conformers

```python
# Generate a 3D conformer
mol = Chem.AddHs(aspirin)
AllChem.EmbedMolecule(mol)
AllChem.MMFFOptimizeMolecule(mol)  # Energy minimization
```

#### b. Pharmacophore Features

Represent the spatial arrangement of key interaction features (H-bond donors/acceptors, charged groups, etc.).

#### c. 3D Grid Representations

Place the molecule in a 3D grid and encode properties at each grid point.

**Advantages:**
- Capture spatial relationships critical for binding
- More accurate representation of molecular shape
- Better for modeling protein-ligand interactions

**Limitations:**
- Computationally expensive
- Molecules may have multiple conformations
- Requires 3D structure generation or experimental data

## Choosing the Right Representation

The optimal molecular representation depends on:

1. **The task at hand**: Different tasks may require different representations
2. **Available data**: Size and quality of your dataset
3. **Computational resources**: Some representations are more computationally intensive
4. **Model type**: Different models work better with different representations
5. **Interpretability needs**: Some representations are more interpretable than others

## Kernel Functions for Molecular Similarity

In Bayesian optimization and active learning, kernel functions that measure similarity between molecules are often used. Common kernels include:

### Tanimoto Kernel

Based on the Tanimoto coefficient, widely used for fingerprint similarity:

```python
def tanimoto_kernel(fp1, fp2):
    """Tanimoto kernel for bit vectors"""
    intersection = np.sum(np.logical_and(fp1, fp2))
    union = np.sum(np.logical_or(fp1, fp2))
    return intersection / union if union > 0 else 0.0
```

### Gaussian Kernel with Molecular Descriptors

```python
def gaussian_kernel(x1, x2, sigma=1.0):
    """Gaussian kernel for descriptor vectors"""
    return np.exp(-np.sum((x1 - x2)**2) / (2 * sigma**2))
```

### Graph Kernels

For comparing molecular graphs directly:

- Shortest-path kernels
- Random walk kernels
- Weisfeiler-Lehman subtree kernels

## Feature Engineering for Molecular Data

Some useful approaches for feature engineering:

1. **Dimensionality reduction**: PCA, t-SNE, or UMAP to reduce high-dimensional fingerprints
2. **Feature selection**: Choose the most relevant descriptors using statistical methods
3. **Normalization**: Scale features to have similar ranges
4. **Combining representations**: Hybrid approaches that combine different representation types

## Implementation Example: Molecular Similarity Search

Here's a simple example of how to use molecular representations to find similar compounds:

```python
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

# Define a set of molecules
smiles_list = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',         # Aspirin
    'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',     # Ibuprofen
    'CC(=O)NC1=CC=C(C=C1)O',             # Acetaminophen
    'OC1=C(C=CC=C1)C(=O)O',              # Salicylic acid
    'C1=CC=C2C(=C1)C(=O)OCC2',           # Coumarin
]

# Convert to RDKit molecules
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
names = ['Aspirin', 'Ibuprofen', 'Acetaminophen', 'Salicylic acid', 'Coumarin']

# Generate Morgan fingerprints
fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]

# Calculate similarity matrix
similarity_matrix = np.zeros((len(fps), len(fps)))
for i in range(len(fps)):
    for j in range(len(fps)):
        similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])

# Print similarity matrix
print("Molecular Similarity Matrix (Tanimoto):")
print("                ", end="")
for name in names:
    print(f"{name:15}", end="")
print()

for i, name in enumerate(names):
    print(f"{name:15}", end="")
    for j in range(len(names)):
        print(f"{similarity_matrix[i, j]:.3f}        ", end="")
    print()
```

## Conclusion

The choice of molecular representation is a critical factor in the success of machine learning approaches in drug discovery. Understanding the strengths and limitations of different representations helps in selecting the most appropriate one for your specific task.

In the context of Bayesian optimization and active learning, molecular representations form the foundation upon which acquisition functions and selection strategies operate. A well-chosen representation can significantly enhance the efficiency of the optimization process and lead to more rapid discovery of promising drug candidates.

## References

1. Mario Krenn, Florian Häse, AkshatKumar Nigam, Pascal Friederich, and Alan Aspuru-Guzik.
Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation (2020). *Machine Learning: Science and Technology*.

2. Cereto-Massagué, A., Ojeda, M. J., Valls, C., Mulero, M., Garcia-Vallvé, S., & Pujadas, G. (2015). Molecular fingerprint similarity search in virtual screening. *Methods*, 71, 58-63.

3. Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742-754.

4. Todeschini, R., & Consonni, V. (2009). *Molecular descriptors for chemoinformatics*. John Wiley & Sons.

5. Duvenaud, D. K., Maclaurin, D., Iparraguirre, J., Bombarell, R., Hirzel, T., Aspuru-Guzik, A., & Adams, R. P. (2015). Convolutional networks on graphs for learning molecular fingerprints. *Advances in Neural Information Processing Systems*, 28.