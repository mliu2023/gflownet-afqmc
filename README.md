# GFlowNets for sampling trajectories in quantum systems

This project aims to accelerate Quantum Monte Carlo algorithms by using a generative flow network (GFlowNet). As a proof-of-concept, we generate trajectories on the Ising model. Each trajectory consists of a sequence of states, each differing from the previous state by one spin. The GFlowNet is trained to sample paths proportional to the energies of their terminal states; this allows the network to more efficiently find reward modes (low energy states) than other RL or MCMC methods.
