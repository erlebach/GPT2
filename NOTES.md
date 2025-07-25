# List of things to do
- Use causal attention and transformer layers from nn.activate and nn.Module
- Find out whether we could process characters rather than tokens. 
- Implement a wide path: two transformers in parallel with merged (weighted, 
  or gated) at the end of a block, and duplicate this for two blocks. 
- Compile this parallel version and see if loss decreases. 
- Run code on the H100. 

## 2025-07-25
### `train_gp2_with_nn.py`
- Create a superblock that considers two parallel blocks, that are 
  concatenated before entering a second superblock. 

