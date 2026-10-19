"""Create evaluation variants of a checkpoint by editing config.json only.

A variant directory symlinks every file of the checkpoint except config.json, which is rewritten
with the given overrides. This is how the lesion arm (ttt_lr=0), the ttt_lr sweep, ttt_mode=false
and YaRN rope scaling are produced without copying weights.

Spec syntax: ``name=path[:key=value[,key=value...]]``; values are parsed as JSON when possible, and
``yarn=<factor>`` expands to the YaRN rope_scaling block for a 32k-native model.
    base=runs/models/Qwen3-4B-Base
    ttt_s1_frozen=runs/checkpoints/ttt_on_seed1/checkpoints/global_step_1431/hf_ckpt:ttt_lr=0
    base_yarn2=runs/models/Qwen3-4B-Base:yarn=2
"""
import json
import os


def parse_spec(spec):
    name, rest = spec.split("=", 1)
    path, _, overrides = rest.partition(":")
    config = {}
    for item in filter(None, overrides.split(",")):
        key, value = item.split("=", 1)
        if key == "yarn":
            factor = float(value)
            config["rope_scaling"] = {"rope_type": "yarn", "factor": factor, "original_max_position_embeddings": 32768}
            config["max_position_embeddings"] = int(32768 * factor)
            continue
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            config[key] = value
    return name, path, config


def make_variant(name, path, overrides, variants_dir):
    """Return a model directory for `name`: `path` itself if no overrides, else a symlink dir."""
    if not overrides:
        return os.path.abspath(path)
    out = os.path.join(os.path.abspath(variants_dir), name)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(path):
        link = os.path.join(out, f)
        if f != "config.json" and not os.path.lexists(link):
            os.symlink(os.path.join(os.path.abspath(path), f), link)
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    config.update(overrides)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return out
