import os
import importlib
from configs.hardware_params import hardware_params
from transformers import AutoConfig
import math
from extension import ModelConfigLoader


def str_number(num):
    """
    Converts a large number into a compact human-readable string using suffixes:
    T (trillion), G (billion), M (million), K (thousand).
    The formatting precision changes based on the number's magnitude.
    """
    for factor, suffix, precision in [
        (1e14, 'T', 0), (1e12, 'T', 1),
        (1e11, 'G', 0), (1e9, 'G', 1),
        (1e8, 'M', 0), (1e6, 'M', 1),
        (1e5, 'K', 0), (1e3, 'K', 1)
    ]:
        if num > factor:
            return f"{num / (10 ** (3 * ('TGMK'.index(suffix) + 1))):.{precision}f}{suffix}"
    return f"{num:.1f}" if num >= 1 else f"{num:.2f}"


def str_number_time(num):
    """
    Converts a small number representing time into a compact string
    using sub-second units:
    m (milliseconds), u (microseconds), n (nanoseconds).
    Falls back to showing as-is if ≥ 1 or as an integer if very small.
    """
    for factor, suffix in [(1e-3, 'm'), (1e-6, 'u'), (1e-9, 'n')]:
        if num > factor:
            return f"{num / factor:.1f}{suffix}"
    return f"{num:.1f}" if num >= 1 else f"{num:.0f}"


ALL_DATA_NAMES = [
    "ops",
    "memory_access",
    "load_weight",
    "load_act",
    "store_act",
    "load_kv_cache",
    "store_kv_cache",
    "inference_time",
]


def roofline_analyze(bandwidth, max_ops, ops, memory_access):
    """
    Analyzes performance based on the Roofline model.

    Parameters:
        bandwidth (float): Memory bandwidth in bytes/second.
        max_ops (float): Maximum achievable operations per second (ops).
        ops (float): Total number of operations performed.
        memory_access (float): Total memory accessed in bytes.

    Returns:
        tuple:
            - arithmetic_intensity (float): Operations per byte.
            - performance (float): Estimated performance in ops.
            - bound (str): Performance bound ('memory' or 'compute').
    """
    arithmetic_intensity = ops / memory_access
    turning_point = max_ops / bandwidth
    if arithmetic_intensity < turning_point:
        bound = "memory"
        performance = arithmetic_intensity * bandwidth
    else:
        bound = "compute"
        performance = max_ops
    return arithmetic_intensity, performance, bound


class ModelProfiler:
    def __init__(self, model_id, config_file=None):
        """
        Initializes the model configuration.

        Args:
            model_id (str): The identifier of the model.
            config_file (str, optional): Path to the model configuration file.
        """
        self.model_id = model_id

        # Load model parameters
        self.model_params = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True)

        # Use unified extension loader to find model configuration
        config_loader = ModelConfigLoader()

        if config_file is None:
            # Auto-discover config using the unified loader
            config_class = config_loader.find_model_config(model_id)
            if config_class is None:
                raise FileNotFoundError(
                    f"No model configuration found for '{model_id}'. "
                    f"Please specify config_file manually or add a config to the models/ directory.")
            self.config = config_class(self.model_params)
            print(
                f"Auto-discovered config: {config_class.__name__} for model: {model_id}")
        else:
            # Load specific config file (legacy support)
            config_module_path = config_file.replace(
                "/", ".").replace(".py", "")
            config_module = importlib.import_module(config_module_path)

            # Check if module has a class-based config (preferred)
            self.config = None
            for attr_name in dir(config_module):
                attr = getattr(config_module, attr_name)
                if (isinstance(attr, type) and
                    attr_name.endswith('Config') and
                    not attr_name.startswith('Base') and
                        hasattr(attr, 'get_num_attention_heads')):
                    self.config = attr(self.model_params)
                    break

            # Fallback to function-based config for backward compatibility
            if self.config is None:
                self.config = config_module

            print(
                f"Using specified config file: {config_file} for model: {model_id}")

        # Temporary attributes
        self.results = None
        self.w_bit = None
        self.a_bit = None
        self.kv_bit = None
        self.batch_size = None
        self.seqlen = None

    def _get_config_value(self, method_name, *args):
        """Helper method to call configuration methods with both class and function-based configs."""
        method = getattr(self.config, method_name)
        if hasattr(self.config, 'model_params'):  # Class-based config
            return method(*args)
        else:  # Function-based config
            return method(self.model_params, *args)

    def _profile_to_results(
        self,
        hardware,
        stage,
        name,
        ops=0,
        load_weight=0,
        load_act=0,
        store_act=0,
        load_kv_cache=0,
        store_kv_cache=0,
    ):

        bandwidth, max_ops, _ = self._get_hardware_info(hardware)
        memory_access = load_weight + load_act + \
            store_act + load_kv_cache + store_kv_cache
        arithmetic_intensity, performance, bound = roofline_analyze(
            bandwidth, max_ops, ops, memory_access)
        inference_time = ops / performance
        self.results[stage][name] = {
            "ops": ops,
            "memory_access": memory_access,
            "arithmetic_intensity": arithmetic_intensity,
            "performance": performance,
            "bound": bound,
            "load_weight": load_weight,
            "load_act": load_act,
            "store_act": store_act,
            "load_kv_cache": load_kv_cache,
            "store_kv_cache": store_kv_cache,
            "inference_time": inference_time,
        }

    def save_csv(self, save_path=None):
        # Construct default save_path if not provided
        if save_path is None:
            model_base = self.model_id[:self.model_id.rfind('/')]
            model_suffix = self.model_id[self.model_id.rfind('/'):]
            save_dir = os.path.join("output", model_base)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, model_suffix)

        # File paths for decode and prefill stages
        file_paths = {
            "decode": f"{save_path}_decode.csv",
            "prefill": f"{save_path}_prefill.csv"
        }

        print(f"Saving to {file_paths['decode']} and {file_paths['prefill']}")

        # Header information
        header_info = (
            f"\n\n=== {self.model_id} {self.hardware} "
            f"w_bit={self.w_bit} a_bit={self.a_bit} "
            f"kv_bit={self.kv_bit} batch_size={self.batch_size} "
            f"seqlen={self.seqlen} tp_size={self.tp_size} ===\n"
        )
        legend = (
            "layer_name,ops,Access,arithmetic_intensity,performance,bound,"
            "load_weight,load_act,store_act,load_kv_cache,store_kv_cache,inference_time\n"
        )

        for stage, file_name in file_paths.items():
            with open(file_name, "a+") as f:
                f.write(header_info)
                f.write(legend)
                for layer_name, result in self.results[stage].items():
                    row = (
                        f"{layer_name},{str_number(result['ops'])},"
                        f"{str_number(result['memory_access'])}B,"
                        f"{str_number(result['arithmetic_intensity'])},"
                        f"{str_number(result['performance'])},{result['bound']},"
                        f"{str_number(result['load_weight'])}B,"
                        f"{str_number(result['load_act'])}B,"
                        f"{str_number(result['store_act'])}B,"
                        f"{str_number(result['load_kv_cache'])}B,"
                        f"{str_number(result['store_kv_cache'])}B,"
                        f"{str_number_time(result['inference_time'])}s\n"
                    )
                    f.write(row)

    def _profile_linear_layers(self, config, model_params, w_byte, a_byte, kv_byte, hardware):
        for name, (ic, oc) in self._get_config_value('get_linear_layers', self.tp_size).items():
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj

            self._profile_to_results(hardware, "decode", name,
                                     ops=ic * oc * self.batch_size * 2,
                                     load_weight=ic * oc * w_byte,
                                     load_act=ic * self.batch_size * a_byte,
                                     store_act=0 if is_kv_proj else oc * self.batch_size * a_byte,
                                     load_kv_cache=0,
                                     store_kv_cache=0 if is_normal_proj else oc * self.batch_size * kv_byte,
                                     )

            self._profile_to_results(hardware, "prefill", name,
                                     ops=ic * oc * self.batch_size * self.seqlen * 2,
                                     load_weight=ic * oc * w_byte,
                                     load_act=ic * self.batch_size * self.seqlen * a_byte,
                                     store_act=0 if is_kv_proj else oc * self.batch_size * self.seqlen * a_byte,
                                     load_kv_cache=0,
                                     store_kv_cache=0 if is_normal_proj else oc *
                                     self.batch_size * self.seqlen * kv_byte,
                                     )

    def _profile_attention(self, head_size, num_heads, kv_heads, w_byte, a_byte, kv_byte, use_flash, hardware):
        seqlen, bs = self.seqlen, self.batch_size

        for stage, is_prefill in [("decode", False), ("prefill", True)]:
            seq1 = seqlen if is_prefill else 1
            seq2 = seqlen if is_prefill else 1

            qk_ops = seq1 * seq2 * head_size * num_heads * bs * 2
            sv_ops = seq2 * head_size * seq1 * num_heads * bs * 2
            softmax_ops = bs * num_heads * seq1 * seq2 * 5

            if use_flash:
                name = "fused_attention"
                _, _, buffer = self._get_hardware_info(hardware)
                block_size = min(
                    math.ceil(buffer / (kv_byte * head_size)), head_size)
                n_blocks = math.ceil(seq2 / block_size)

                q_bytes = seq1 * head_size * bs * num_heads * a_byte
                o_bytes = seq1 * seq2 * bs * num_heads * a_byte

                self._profile_to_results(hardware, stage, name,
                                         ops=qk_ops + sv_ops + softmax_ops,
                                         load_weight=0,
                                         load_act=q_bytes,
                                         store_act=o_bytes * 2,
                                         load_kv_cache=n_blocks * seq2 * head_size * bs * kv_heads * kv_byte * 2,
                                         store_kv_cache=0
                                         )
            else:
                for name, ops in [("qk_matmul", qk_ops), ("sv_matmul", sv_ops), ("softmax", softmax_ops)]:
                    load_act = bs * num_heads * seq1 * seq2 * a_byte
                    store_act = load_act
                    load_kv = seq2 * head_size * bs * kv_heads * kv_byte if "matmul" in name else 0

                    self._profile_to_results(hardware, stage, name,
                                             ops=ops,
                                             load_weight=0,
                                             load_act=load_act,
                                             store_act=store_act,
                                             load_kv_cache=load_kv,
                                             store_kv_cache=0
                                             )

    def _profile_norm_and_misc(self, hidden_size, a_byte, hardware):
        seqlen, bs = self.seqlen, self.batch_size
        for stage, seq in [("decode", 1), ("prefill", seqlen)]:
            for name in self._get_config_value('get_norm_layers'):
                self._profile_to_results(hardware, stage, name, ops=bs * hidden_size * seq * 7, load_weight=0, load_act=bs *
                                         hidden_size * seq * a_byte, store_act=bs * hidden_size * seq * a_byte, load_kv_cache=0, store_kv_cache=0)

            for name in ["attn_add", "mlp_add"]:
                self._profile_to_results(hardware, stage, name, ops=bs * hidden_size * seq, load_weight=0, load_act=bs *
                                         hidden_size * seq * a_byte, store_act=bs * hidden_size * seq * a_byte, load_kv_cache=0, store_kv_cache=0)

            self._profile_to_results(hardware, stage, "mlp_act", ops=bs * hidden_size * seq * 2, load_weight=0, load_act=bs *
                                     hidden_size * seq * a_byte * 2, store_act=bs * hidden_size * seq * a_byte, load_kv_cache=0, store_kv_cache=0)

    def _compute_totals(self, num_layers):
        total = {"decode": {}, "prefill": {}}
        for key in ALL_DATA_NAMES:
            total["decode"][key] = 0
            total["prefill"][key] = 0

        for stage in ["decode", "prefill"]:
            for _, result in self.results[stage].items():
                for key in ALL_DATA_NAMES:
                    total[stage][key] += result[key] * num_layers

        for stage in ["decode", "prefill"]:
            tmp_act = sum(result["store_act"]
                          for result in self.results[stage].values())
            weight_kv = total["prefill"]["load_weight"] + \
                total["prefill"]["store_kv_cache"]

            total[stage]["memory_consumption"] = tmp_act + weight_kv
            total[stage]["memory_consumption_tmp_act"] = tmp_act
            total[stage]["memory_consumption_weight"] = total["prefill"]["load_weight"]
            total[stage]["memory_consumption_kv_cache"] = total["prefill"]["store_kv_cache"]

        self.results["total_results"] = total

    def _profile_lm_head(self, w_byte, a_byte, hardware):
        args = {"batch_size": self.batch_size,
                "a_byte": a_byte, "w_byte": w_byte}
        for info in self._get_config_value('post_process', args):
            self._profile_to_results(
                hardware,
                info['stage'],
                info['name'],
                info['ops'],
                info['load_weight'],
                info['load_act'],
                info['store_act']
            )
            for key in ALL_DATA_NAMES:
                self.results["total_results"][info["stage"]
                                              ][key] += self.results[info["stage"]][info["name"]][key]

    def profile(
        self,
        hardware,
        seqlen,
        batch_size,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention=False,
        tp_size: int = 1,
    ):
        """
        Analyze memory, compute, and performance characteristics for both decode and prefill phases.

        Args:
            seqlen (int): Sequence length.
            batch_size (int): Batch size.
            w_bit (int): Bit width of weights.
            a_bit (int): Bit width of activations.
            kv_bit (int or None): Bit width for key-value cache (defaults to `a_bit`).
            use_flashattention (bool): Whether to use flash attention.
            tp_size (int): Tensor parallelism size.

        Returns:
            dict: Structured analysis result for decode and prefill, including per-layer stats and total summaries.
        """
        assert seqlen > 0 and batch_size > 0

        self.results = {"decode": {}, "prefill": {}}
        kv_bit = kv_bit if kv_bit is not None else a_bit

        # Cache commonly used fields
        self.w_bit, self.a_bit, self.kv_bit = w_bit, a_bit, kv_bit
        self.batch_size, self.seqlen, self.tp_size = batch_size, seqlen, tp_size

        w_byte, a_byte, kv_byte = w_bit / 8, a_bit / 8, kv_bit / 8
        config, model_params = self.config, self.model_params

        # TODO: support MLA
        num_heads = self._get_config_value('get_num_attention_heads')
        kv_heads = self._get_config_value('get_num_key_value_heads')
        hidden_size = self._get_config_value('get_hidden_size')
        num_layers = self._get_config_value('get_num_hidden_layers')
        head_size = hidden_size // num_heads

        # TODO: support MoE
        self._profile_linear_layers(
            config, model_params, w_byte, a_byte, kv_byte, hardware)
        self._profile_attention(
            head_size, num_heads, kv_heads, w_byte, a_byte, kv_byte, use_flashattention, hardware
        )
        self._profile_norm_and_misc(hidden_size, a_byte, hardware)
        self._compute_totals(num_layers)
        self._profile_lm_head(w_byte, a_byte, hardware)

        return self.results

    def profile_prefill_iteration(self, hardware, sequence_length, batch_size,
                                  w_bit=16,
                                  a_bit=16,
                                  kv_bit=None,
                                  use_flashattention=False,
                                  tp_size: int = 1):
        result = self.profile(
            hardware,
            sequence_length,
            batch_size,
            w_bit,
            a_bit,
            kv_bit,
            use_flashattention=use_flashattention,
            tp_size=tp_size
        )
        prefill_time = result["total_results"]["prefill"]["inference_time"]
        return prefill_time

    def profile_decode_iteration(self, hardware, sequence_length, batch_size,
                                 w_bit=16,
                                 a_bit=16,
                                 kv_bit=None,
                                 use_flashattention=False,
                                 tp_size: int = 1):
        result = self.profile(hardware, sequence_length, batch_size, w_bit, a_bit, kv_bit,
                              use_flashattention=use_flashattention, tp_size=tp_size)
        decode_time = result["total_results"]["decode"]["inference_time"]
        return decode_time

    def _get_hardware_info(self, hardware):
        bandwidth = hardware_params[hardware]["bandwidth"]
        if self.w_bit <= 8 and self.a_bit <= 8 and self.kv_bit <= 8:
            max_ops = hardware_params[hardware]["INT8"]
        else:
            max_ops = hardware_params[hardware]["FP16"]
        onchip_buffer = hardware_params[hardware]["onchip_buffer"]
        return bandwidth, max_ops, onchip_buffer

    def _get_model_info(self):
        if self._get_config_value('get_num_attention_heads') != self._get_config_value('get_num_key_value_heads'):
            GQA = True
        else:
            GQA = False

        info = {"GQA": GQA}  # group query attention
        return info


if __name__ == "__main__":
    model_id = 'unsloth/Llama-3.3-70B-Instruct'
    hardware = 'NVIDIA_A100_40G'
    result = ModelProfiler(model_id).analyze_generate_task(
        hardware, 2048, 1024, 8)
    print(result)
