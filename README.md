# GFlowNets for sampling paths on the Ising model

This project aims to accelerate the computation of quantum path integrals by using a generative flow network (GFlowNet) to generate paths on the Ising model. Each path consists of a sequence of states, each differing from the previous state by one spin. The GFlowNet is trained to sample paths proportional to the energies of their terminal states; this allows the network to more efficiently find reward modes (low energy states) than other RL or MCMC methods.
