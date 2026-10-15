# Reaction Representations for Machine Learning

Representing chemical reactions in a format suitable for machine learning algorithms is crucial for computational reaction optimization and discovery. In this chapter, we'll explore different ways to encode chemical reactions into numerical or vector representations that can be used in Bayesian optimization and active learning approaches.

## Why Reaction Representations Matter

The choice of reaction representation can significantly impact:

- The performance of machine learning models in predicting reaction outcomes
- The ability to capture relevant chemical transformation information
- The computational efficiency of the optimization process
- The interpretability of the results

## Common Reaction Representations

### 1. String-Based Representations

Reactions can be represented as strings using various notations. The most common is the **Reaction SMILES** format, which represents reactions as a combination of reactant and product SMILES strings.

Example: The esterification of acetic acid with ethanol is represented as:
`CC(=O)O.CCO>>CC(=O)OCC.O`

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# Create an RDKit reaction from SMILES
rxn = AllChem.ReactionFromSmarts('CC(=O)O.CCO>>CC(=O)OCC.O')

# Display the reaction
Draw.ReactionToImage(rxn)
```

**Advantages:**
- Compact and human-readable
- Widely used in reaction databases
- Easy to store and share

**Limitations:**
- Multiple valid representations for the same reaction
- Doesn't explicitly encode reaction conditions or mechanisms

### 2. Reaction Fingerprints

Fingerprints that encode the structural changes occurring in a reaction.

#### a. Difference Fingerprints

Capture the difference between products and reactants.

```python
from rdkit.Chem import AllChem

# Generate difference fingerprint
def get_reaction_fingerprint(rxn, radius=2, nBits=1024):
    # Get reactant and product fingerprints
    reactant_fp = AllChem.GetMorganFingerprintAsBitVect(rxn.GetReactantTemplate(0), radius, nBits)
    product_fp = AllChem.GetMorganFingerprintAsBitVect(rxn.GetProductTemplate(0), radius, nBits)
    
    # XOR operation to get difference
    return reactant_fp ^ product_fp
```

#### b. Reaction Class Fingerprints

Encode the type of chemical transformation.

```python
from rdkit.Chem import rdChemReactions

# Generate reaction class fingerprint
def get_reaction_class_fingerprint(rxn):
    return rdChemReactions.GetReactionFingerprint(rxn)
```

**Advantages:**
- Fixed length, suitable for ML algorithms
- Capture important reaction features
- Computationally efficient

**Limitations:**
- May miss subtle reaction details
- Different fingerprints capture different aspects of reactions

### 3. Reaction Descriptors

Numerical values that describe various properties of reactions.

```python
from rdkit.Chem import Descriptors

def calculate_reaction_descriptors(rxn):
    # Calculate descriptors for reactants and products
    reactant_mol = rxn.GetReactantTemplate(0)
    product_mol = rxn.GetProductTemplate(0)
    
    # Calculate property changes
    delta_mw = Descriptors.MolWt(product_mol) - Descriptors.MolWt(reactant_mol)
    delta_logp = Descriptors.MolLogP(product_mol) - Descriptors.MolLogP(reactant_mol)
    delta_tpsa = Descriptors.TPSA(product_mol) - Descriptors.TPSA(reactant_mol)
    
    return {
        'delta_mw': delta_mw,
        'delta_logp': delta_logp,
        'delta_tpsa': delta_tpsa
    }
```

**Advantages:**
- Interpretable physicochemical changes
- Directly related to reaction properties
- Can incorporate expert knowledge

**Limitations:**
- May require feature selection
- Different descriptors may be correlated
- May not capture all relevant reaction information

### 4. Graph-Based Representations

Reactions can be represented as graphs showing the transformation of molecular graphs.

```python
import networkx as nx

def reaction_to_graph(rxn):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add reactant and product nodes
    for i, mol in enumerate(rxn.GetReactants()):
        G.add_node(f'reactant_{i}', type='reactant', mol=mol)
    
    for i, mol in enumerate(rxn.GetProducts()):
        G.add_node(f'product_{i}', type='product', mol=mol)
    
    # Add edges representing atom mapping
    for mapping in rxn.GetAtomMapping():
        G.add_edge(f'reactant_{mapping[0]}', f'product_{mapping[1]}')
    
    return G
```

**Advantages:**
- Preserves reaction topology
- Captures atom mapping information
- Suitable for graph neural networks

**Limitations:**
- More complex to process
- Variable size, requiring special handling
- May not explicitly encode reaction conditions

### 5. 3D Reaction Representations

Capture the three-dimensional aspects of reactions, including transition states and conformations.

#### a. Transition State Representations

```python
def generate_transition_state(rxn):
    # Generate 3D structures
    reactant_3d = Chem.AddHs(rxn.GetReactantTemplate(0))
    product_3d = Chem.AddHs(rxn.GetProductTemplate(0))
    
    # Generate conformers
    AllChem.EmbedMolecule(reactant_3d)
    AllChem.EmbedMolecule(product_3d)
    
    # Energy minimization
    AllChem.MMFFOptimizeMolecule(reactant_3d)
    AllChem.MMFFOptimizeMolecule(product_3d)
    
    return reactant_3d, product_3d
```

#### b. Reaction Coordinate Representations

Encode the progress of the reaction along a reaction coordinate.

**Advantages:**
- Capture spatial relationships critical for reactivity
- More accurate representation of reaction mechanisms
- Better for modeling reaction kinetics

**Limitations:**
- Computationally expensive
- Requires quantum chemical calculations
- May be difficult to generate for complex reactions

## Choosing the Right Representation

The optimal reaction representation depends on:

1. **The task at hand**: Different tasks may require different representations
2. **Available data**: Size and quality of your reaction dataset
3. **Computational resources**: Some representations are more computationally intensive
4. **Model type**: Different models work better with different representations
5. **Interpretability needs**: Some representations are more interpretable than others

## Kernel Functions for Reaction Similarity

In Bayesian optimization and active learning, kernel functions that measure similarity between reactions are often used. Common kernels include:

### Tanimoto Kernel for Reaction Fingerprints

```python
def reaction_tanimoto_kernel(rxn1, rxn2):
    """Tanimoto kernel for reaction fingerprints"""
    fp1 = get_reaction_fingerprint(rxn1)
    fp2 = get_reaction_fingerprint(rxn2)
    intersection = np.sum(np.logical_and(fp1, fp2))
    union = np.sum(np.logical_or(fp1, fp2))
    return intersection / union if union > 0 else 0.0
```

### Gaussian Kernel with Reaction Descriptors

```python
def reaction_gaussian_kernel(rxn1, rxn2, sigma=1.0):
    """Gaussian kernel for reaction descriptor vectors"""
    desc1 = calculate_reaction_descriptors(rxn1)
    desc2 = calculate_reaction_descriptors(rxn2)
    x1 = np.array(list(desc1.values()))
    x2 = np.array(list(desc2.values()))
    return np.exp(-np.sum((x1 - x2)**2) / (2 * sigma**2))
```

### Graph Kernels for Reactions

For comparing reaction graphs directly:
```python
def reaction_graph_kernel(G1, G2):
    """Graph kernel for reaction graphs"""
    # Implement graph kernel method (e.g., Weisfeiler-Lehman)
    pass
```


