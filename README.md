# Neural Architecture Search for semantic segmentation in medical imaging

This is an implementation of NAS method that searches for the best architecture from the given search space.

For the search part, Dispersive Flies Optimisation algorithm is used which belongs to a group of Swarm Intelligence algorithms.

The base architecture is U-Net that is consisted of 7 blocks in total. Four blocks is devoted to the encoder part, and three blocks are for the decoder part.
Each block is represented as a Directed Acyclic Graph that consists of up to 5 nodes, and the connections between them are enocoded.

This is the coding part of my dissertation.