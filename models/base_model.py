from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any


class BaseModelConfig(ABC):
    """
    Abstract base class for model configurations.
    
    This class defines the interface that all model configurations must implement
    to work with the ModelProfiler system. Subclasses should implement model-specific
    parameter extraction and layer definitions.
    """

    def __init__(self, model_params):
        """
        Initialize the model configuration.
        
        Args:
            model_params: Model parameters from transformers.AutoConfig or similar
        """
        self.model_params = model_params

    @abstractmethod
    def get_num_attention_heads(self) -> int:
        """Get the number of attention heads."""
        pass

    @abstractmethod
    def get_hidden_size(self) -> int:
        """Get the hidden size/dimension."""
        pass

    @abstractmethod
    def get_num_key_value_heads(self) -> int:
        """Get the number of key-value heads (for GQA/MQA)."""
        pass

    @abstractmethod
    def get_norm_layers(self) -> List[str]:
        """Get the list of normalization layer names."""
        pass

    @abstractmethod
    def get_num_hidden_layers(self) -> int:
        """Get the number of transformer layers."""
        pass

    @abstractmethod
    def get_intermediate_size(self) -> int:
        """Get the intermediate/feed-forward dimension."""
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        pass

    @abstractmethod
    def get_linear_layers(self, tp_size: int = 1) -> Dict[str, Tuple[int, int]]:
        """
        Get the linear layer specifications.
        
        Args:
            tp_size: Tensor parallelism size
            
        Returns:
            Dict mapping layer names to (input_size, output_size) tuples
        """
        pass

    @abstractmethod
    def post_process(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get post-processing layer specifications (e.g., language modeling head).
        
        Args:
            args: Dictionary containing batch_size, w_byte, a_byte, etc.
            
        Returns:
            List of layer specifications with name, stage, ops, load_weight, etc.
        """
        pass

    @property
    @abstractmethod
    def transformer_layer_graph(self) -> Dict[str, List[str]]:
        """Get the transformer layer computation graph."""
        pass

    @property
    @abstractmethod
    def flashattention_transformer_layer_graph(self) -> Dict[str, List[str]]:
        """Get the flash attention transformer layer computation graph."""
        pass

    def validate_tp_size(self, tp_size: int) -> None:
        """
        Validate tensor parallelism size for this model configuration.
        
        Args:
            tp_size: Tensor parallelism size to validate
            
        Raises:
            AssertionError: If tp_size is invalid for this model
        """
        if tp_size <= 1:
            return
            
        hidden_size = self.get_hidden_size()
        intermediate_size = self.get_intermediate_size()
        key_value_heads = self.get_num_key_value_heads()
        
        assert hidden_size % tp_size == 0, f"Hidden size {hidden_size} not divisible by tp_size {tp_size}"
        assert intermediate_size % tp_size == 0, f"Intermediate size {intermediate_size} not divisible by tp_size {tp_size}"
        assert key_value_heads % tp_size == 0, f"Key-value heads {key_value_heads} not divisible by tp_size {tp_size}"