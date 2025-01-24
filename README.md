### README for Incremental Deployment Method of SRv6 Network

---

# Incremental Deployment Method of Segment Routing over an IPv6 (SRv6) Network Based on Graph Neural Network (GNN) and Multi-Agent Reinforcement Learning (MARL)

## Introduction

This project tackles the challenge of network load imbalance by utilizing a novel Incremental Deployment Method for Segment Routing over IPv6 (SRv6) networks, leveraging the strengths of Graph Neural Networks (GNN) and Multi-Agent Reinforcement Learning (MARL).

## Motivation

The fast-evolving network technology necessitates effective management of load balancing. SRv6 represents the cutting edge of this technological frontier, yet existing algorithms often fail to fully perceive the network's dynamic state, leading to load imbalances. Our project aims to enhance the perception of network status information and bolster the incremental deployment capability of SRv6 networks.

## Project Components

- **Python**: Primary programming language used.
- **TensorFlow**: Utilized for implementing GNNs and MARL algorithms.
- **PyCharm**: Chosen as the integrated development environment (IDE) for coding, compiling, and debugging.

## Algorithm Logic

1. **Data Preprocessing Stage**:
   - Collect raw network status information from the SRv6 network, such as link load data.

2. **Feature Engineering**:
   - Compute features for each node and link within the network, such as node degree, betweenness centrality, and the most load link.

3. **Application of Graph Neural Networks (GNN)**:
   - Employ Graph Neural Networks to process the extracted network features to understand the topology and dynamics within the network.
   - GNNs capture the complex interactions between nodes through their structure and effectively encode the network state.

4. **Multi-Agent Reinforcement Learning (MARL)**:
   - On the basis of the network state encoded by GNNs, use Multi-Agent Reinforcement Learning to train agents.
   - Agents learn to make decisions based on the current network state to optimize the segment routing deployment in the SRv6 network.

5. **Policy Iteration**:
   - Iteratively optimize the perception of network state information and the decision-making strategy for incremental deployment.

6. **Load Balancing**:
   - The ultimate goal is to achieve a balanced network load, addressing the issue of load imbalance.


## Implementation

Through the implementation of our algorithm, we have successfully devised a method for deploying SRv6 networks, ensuring harmonious link load balancing and thus offering a more resilient and efficient network infrastructure.

## Results

_The outcome of this research is a patented method for the incremental deployment of segment routing networks using Graph Neural Networks and Reinforcement Learning._

## Equal Contributions
This project and all associated work were equally contributed by Yuqing Xia and Hongyu Yan.

