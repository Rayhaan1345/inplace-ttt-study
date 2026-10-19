"""OpenCompass model wrapper that loads checkpoints with the In-Place TTT inference classes.

OpenCompass runs each inference task in its own process, so the TTT classes must be registered
with ``AutoModelForCausalLM`` inside that process. Importing ``inference_model`` (from
third_party/In-Place-TTT, on PYTHONPATH) does that; stock Qwen3 checkpoints load unchanged.
"""
from opencompass.models import HuggingFaceBaseModel


class TTTHuggingFaceModel(HuggingFaceBaseModel):
    def _load_model(self, path, kwargs, peft_path=None, peft_kwargs=dict()):
        import inference_model  # noqa: F401  registers the TTT-aware Qwen3/Llama classes

        kwargs = dict(attn_implementation="flash_attention_2", torch_dtype="bfloat16", **kwargs)
        super()._load_model(path, kwargs, peft_path, peft_kwargs)
        cfg = self.model.config
        self.logger.info(f"loaded {type(self.model).__name__} ttt_mode={getattr(cfg, 'ttt_mode', None)} "
                         f"ttt_lr={getattr(cfg, 'ttt_lr', None)} ttt_chunk={getattr(cfg, 'ttt_chunk', None)} "
                         f"rope_scaling={getattr(cfg, 'rope_scaling', None)}")
