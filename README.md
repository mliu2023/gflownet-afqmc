# GFlowNets for sampling auxiliary fields in quantum systems

This project aims to use reinforcement learning to learn a distribution of discrete auxiliary fields from a relatively small number of samples. We view these fields as discrete trajectories from an initial state to a terminal state. The initial state is a blank slate where every spin is unspecified, and the terminal state is a fully specified spin configuration. From one state to the next, an unspecified spin is set to +1 or -1. 

We use generative flow networks (GFlowNets) to sample these trajectories because we believe they are better suited to learn high-dimensional distributions. We concurrently train an energy model to learn the energy of each terminal state, and the reward of each terminal state is computed as the exponential of minus energy.

The GFlowNet is trained using a flow-matching objective to sample paths proportional to the rewards of the terminal states; this allows it to find reward modes (low-energy states) more efficiently than other RL or MCMC methods.

## Running the code
```
python gflownet_af_void.py
```
