## Dataset Overview
This dataset contains graph-based representations of Active Directory (AD) environments, structured to support statistical learning of attack paths within networks. Each graph represents an AD environment, where nodes correspond to various entities (e.g., users, computers, domains), and edges encode relationships and privileges. The dataset is designed for machine learning tasks such as attack path prediction and node classification.

## Purpose
The dataset facilitates research in cybersecurity, particularly in:
- Learning the joint distribution of attack paths and network environments.
- Identifying high-risk attack paths based on graph structures.
- Enhancing automated red teaming and security assessments.

Unlike existing datasets that either focus on attack sequences or environmental details, this dataset combines both, addressing a gap in the cybersecurity research community.

## Dataset Structure
Each sample in the dataset is stored as a `.pt` file and consists of:
- **Adjacency Tensor**: Represents different edge types in the graph.
- **Feature Matrix**: Encodes node attributes, including entity types and properties.
- **Target Matrix**: Defines the attack paths as binary adjacency matrices.

Formally, the dataset is structured as:
- $X_i = (A_i, F_i)$, where:
  - $A_i \in \mathbb{R}^{|V| \times |V| \times d}$ is the adjacency tensor with $d = 16$ edge types.
  - $F_i \in \mathbb{R}^{|V| \times p}$ is the feature matrix with $p = 20$ node features.
- $Y_i \in \mathbb{R}^{|V| \times |V|}$, where $Y_{ij} = 1$ if an edge between nodes $i\rightarrow j$ is part of an attack path.

## Node Features
Each node in the graph is represented by a feature vector, capturing:
- **Entity Type**: One-hot encoding for User, Computer, OU, Group, GPO, Domain.
- **Operating System**: One-hot encoding of OS types (e.g., Windows Server 2008, Windows 10).
- **Boolean Properties**: Indicators for security-relevant attributes such as:
  - `enabled`
  - `hasspn`
  - `highvalue`
  - `is_vulnerable`
  - `target` (indicates the end node)
  - `owned` (indicates the start node)

## Edge Types
Each graph includes 16 directed edge types representing different AD relationships:
- `AdminTo`
- `AllowedToDelegate`
- `CanRDP`
- `Contains`
- `DCSync`
- `ExecuteDCOM`
- `GenericAll`
- `GetChanges`
- `GetChangesAll`
- `GpLink`
- `HasSession`
- `MemberOf`
- `Open` (for remote code execution if CVE is present)
- `Owns`
- `WriteDacl`
- `WriteOwner`

## Preprocessing Steps
The dataset is preprocessed with the following steps:
1. **Feature Matrix Construction**: Node attributes are extracted and encoded as feature vectors.
2. **Adjacency Tensor Construction**: A multi-dimensional tensor is generated for different edge types.
3. **Target Matrix Generation**: Attack paths are identified and stored as binary adjacency matrices.
4. **Graph Filtering**: Only graphs with exactly 361 nodes are retained.
5. **Path Sampling**: A random valid attack path is selected from each graph.

## Dataset Statistics
- **Number of Samples**: 1,033 graphs
- **Average Nodes per Graph**: 361
- **Average Attack Path Length**: ~8 hops
- **Total Edge Types**: 16
- **Feature Dimensions**: 20 per node

## Intended Use
This dataset is intended for research in:
- Graph-based cybersecurity analytics
- Attack path prediction
- Adversarial machine learning
- Automated red teaming

## Citation
If you use this dataset, please cite the following works:


