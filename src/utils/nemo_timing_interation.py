# src/utils/nemo_timing_integration.py
from utils.metrics_extensions import measure_performance
from utils.memory_utils import benchmark_memory_usage

def add_timing_decorators(model):
    """Add your timing and memory decorators to NeMo-wrapped models."""
    
    # Decorate the core forward pass
    original_forward = model.forward
    
    @measure_performance(memory_enabled=True, timing_enabled=True)
    def timed_forward(*args, **kwargs):
        return original_forward(*args, **kwargs)
    
    model.forward = timed_forward
    
    # Decorate SuperBlock forward passes if they exist
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        for i, superblock in enumerate(model.transformer.h):
            original_superblock_forward = superblock.forward
            
            @measure_performance(memory_enabled=True, timing_enabled=True)
            def timed_superblock_forward(x, superblock=superblock):
                return original_superblock_forward(x)
            
            superblock.forward = timed_superblock_forward
    
    return model
