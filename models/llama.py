
from .base_model import BaseModelConfig
from typing import Dict, List, Tuple, Any


class LlamaConfig(BaseModelConfig):
    """Llama model configuration implementing the BaseModelConfig interface."""

    def get_num_attention_heads(self) -> int:
        return getattr(self.model_params, "num_attention_heads")

    def get_hidden_size(self) -> int:
        return getattr(self.model_params, "hidden_size")

    def get_num_key_value_heads(self) -> int:
        return getattr(self.model_params, "num_key_value_heads")

    def get_norm_layers(self) -> List[str]:
        return ["attn_norm", "mlp_norm"]

    def get_num_hidden_layers(self) -> int:
        return getattr(self.model_params, "num_hidden_layers")

    def get_intermediate_size(self) -> int:
        return getattr(self.model_params, "intermediate_size")

    def get_vocab_size(self) -> int:
        return getattr(self.model_params, "vocab_size")

    def get_linear_layers(self, tp_size: int = 1) -> Dict[str, Tuple[int, int]]:
        hidden_size = self.get_hidden_size()
        intermediate_size = self.get_intermediate_size()
        key_value_heads = self.get_num_key_value_heads()
        attention_heads = self.get_num_attention_heads()

        self.validate_tp_size(tp_size)

        return {
            "q_proj": (hidden_size, hidden_size // tp_size),
            "k_proj": (hidden_size, hidden_size * key_value_heads // attention_heads // tp_size),
            "v_proj": (hidden_size, hidden_size * key_value_heads // attention_heads // tp_size),
            "out_proj": (hidden_size // tp_size, hidden_size),
            "gate_proj": (hidden_size, intermediate_size // tp_size),
            "up_proj": (hidden_size, intermediate_size // tp_size),
            "down_proj": (intermediate_size // tp_size, hidden_size),
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
            "attn_norm": ["input"],
            "q_proj": ["attn_norm"],
            "k_proj": ["attn_norm"],
            "v_proj": ["attn_norm"],
            "qk_matmul": ["q_proj", "k_proj"],
            "softmax": ["qk_matmul"],
            "sv_matmul": ["softmax", "v_proj"],
            "out_proj": ["sv_matmul"],
            "attn_add": ["input", "out_proj"],
            "mlp_norm": ["attn_add"],
            "gate_proj": ["mlp_norm"],
            "up_proj": ["mlp_norm"],
            "mlp_act": ["up_proj", "gate_proj"],
            "down_proj": ["mlp_act"],
            "mlp_add": ["attn_add", "down_proj"],
            "output": ["mlp_add"]
        }

    @property
    def flashattention_transformer_layer_graph(self) -> Dict[str, List[str]]:
        return {
            "input": [],
            "attn_norm": ["input"],
            "q_proj": ["attn_norm"],
            "k_proj": ["attn_norm"],
            "v_proj": ["attn_norm"],
            "fused_attention": ["q_proj", "k_proj", "v_proj"],
            "out_proj": ["fused_attention"],
            "attn_add": ["input", "out_proj"],
            "mlp_norm": ["attn_add"],
            "gate_proj": ["mlp_norm"],
            "up_proj": ["mlp_norm"],
            "mlp_act": ["up_proj", "gate_proj"],
            "down_proj": ["mlp_act"],
            "mlp_add": ["attn_add", "down_proj"],
            "output": ["mlp_add"]
        }


# Backward compatibility - keep function-based interface
def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")


def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")


def get_num_key_value_heads(model_params):
    return getattr(model_params, "num_key_value_heads")


def get_norm_layers(model_params):
    config = LlamaConfig(model_params)
    return config.get_norm_layers()


def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")


def get_intermediate_size(model_params):
    return getattr(model_params, "intermediate_size")


def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size")


def post_process(model_params, args):
    config = LlamaConfig(model_params)
    return config.post_process(args)


def get_linear_layers(model_params, tp_size: int):
    config = LlamaConfig(model_params)
    return config.get_linear_layers(tp_size)


# name, input_names
transformer_layer_graph = {
    "input": [],
    "attn_norm": ["input"],
    "q_proj": ["attn_norm"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "qk_matmul": ["q_proj", "k_proj"],
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "v_proj"],
    "out_proj": ["sv_matmul"],
    "attn_add": ["input", "out_proj"],
    "mlp_norm": ["attn_add"],
    "gate_proj": ["mlp_norm"],
    "up_proj": ["mlp_norm"],
    "mlp_act": ["up_proj", "gate_proj"],
    "down_proj": ["mlp_act"],
    "mlp_add": ["attn_add", "down_proj"],
    "output": ["mlp_add"]
}

flashattention_transformer_layer_graph = {
    "input": [],
    "attn_norm": ["input"],
    "q_proj": ["attn_norm"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "fused_attention": ["q_proj", "k_proj", "v_proj"],
    "out_proj": ["fused_attention"],
    "attn_add": ["input", "out_proj"],
    "mlp_norm": ["attn_add"],
    "gate_proj": ["mlp_norm"],
    "up_proj": ["mlp_norm"],
    "mlp_act": ["up_proj", "gate_proj"],
    "down_proj": ["mlp_act"],
    "mlp_add": ["attn_add", "down_proj"],
    "output": ["mlp_add"]
}
