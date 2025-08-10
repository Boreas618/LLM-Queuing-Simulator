from .base_model import BaseModelConfig
from typing import Dict, List, Tuple, Any


class GPT2Config(BaseModelConfig):
    """GPT-2 model configuration implementing the BaseModelConfig interface."""

    def get_num_attention_heads(self) -> int:
        return getattr(self.model_params, "n_head")

    def get_hidden_size(self) -> int:
        return getattr(self.model_params, "n_embd")

    def get_num_key_value_heads(self) -> int:
        # GPT-2 uses multi-head attention (no GQA)
        return self.get_num_attention_heads()

    def get_norm_layers(self) -> List[str]:
        return ["ln_1", "ln_2"]

    def get_num_hidden_layers(self) -> int:
        return getattr(self.model_params, "n_layer")

    def get_intermediate_size(self) -> int:
        # GPT-2 uses 4x hidden size for MLP
        return 4 * self.get_hidden_size()

    def get_vocab_size(self) -> int:
        return getattr(self.model_params, "vocab_size")

    def get_linear_layers(self, tp_size: int = 1) -> Dict[str, Tuple[int, int]]:
        hidden_size = self.get_hidden_size()
        intermediate_size = self.get_intermediate_size()

        self.validate_tp_size(tp_size)

        return {
            "c_attn": (hidden_size, 3 * hidden_size // tp_size),  # Combined Q, K, V
            "c_proj": (hidden_size // tp_size, hidden_size),      # Output projection
            "c_fc": (hidden_size, intermediate_size // tp_size),   # MLP up
            "c_proj_mlp": (intermediate_size // tp_size, hidden_size),  # MLP down
        }

    def post_process(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        hidden_size = self.get_hidden_size()
        vocab_size = self.get_vocab_size()
        layers = []
        for stage in ["prefill", "decode"]:
            layers.append({
                'name': 'lm_head',
                'stage': stage,
                'ops': args['batch_size'] * hidden_size * vocab_size * 1,
                'load_weight': hidden_size * vocab_size * args['w_byte'],
                'load_act': hidden_size * args['a_byte'],
                'store_act': vocab_size * args['a_byte'],
            })
        return layers

    @property
    def transformer_layer_graph(self) -> Dict[str, List[str]]:
        return {
            "input": [],
            "ln_1": ["input"],
            "c_attn": ["ln_1"],
            "attention": ["c_attn"],
            "c_proj": ["attention"],
            "attn_add": ["input", "c_proj"],
            "ln_2": ["attn_add"],
            "c_fc": ["ln_2"],
            "gelu": ["c_fc"],
            "c_proj_mlp": ["gelu"],
            "mlp_add": ["attn_add", "c_proj_mlp"],
            "output": ["mlp_add"]
        }

    @property
    def flashattention_transformer_layer_graph(self) -> Dict[str, List[str]]:
        # GPT-2 style attention doesn't typically use flash attention,
        # but we provide a similar structure for compatibility
        return {
            "input": [],
            "ln_1": ["input"],
            "c_attn": ["ln_1"],
            "fused_attention": ["c_attn"],
            "c_proj": ["fused_attention"],
            "attn_add": ["input", "c_proj"],
            "ln_2": ["attn_add"],
            "c_fc": ["ln_2"],
            "gelu": ["c_fc"],
            "c_proj_mlp": ["gelu"],
            "mlp_add": ["attn_add", "c_proj_mlp"],
            "output": ["mlp_add"]
        }