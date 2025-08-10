model:
  _target_: gpt2_standalone.lightning_module_adapter.GPTLMAdapter
  # Everything under here is passed to the class constructor
  # For this LightningModule, the constructor expects a GPTConfig.
  # We'll inject block_size and vocab_size from the spec automatically in code.
  n_layer: 4
  n_head: 8
  n_embd: 1024
  n_blocks_per_super: 2

optimizer:
  _target_: torch.optim.AdamW
  lr: 3e-4
  weight_decay: 0.01
