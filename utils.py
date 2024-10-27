import os, re, json
import torch, numpy
from collections import defaultdict
from easyeditor.util import nethook
from easyeditor.models.rome.globals import DATA_DIR
from easyeditor.models.rome.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap, calculate_hidden_flow, plot_hidden_flow,
)
from easyeditor.models.rome.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from easyeditor.models.rome.dsets import KnownsDataset
import yaml

torch.set_grad_enabled(False)

class RomeTest:
    def __init__(self, hparams_path):
        self.noise_level = None
        with open(hparams_path, "r") as stream:
            config = yaml.safe_load(stream)
            self.model_name = config["model_name"]
            try:
                self.noise_level = config["noise_level"]
            except KeyError:
                pass

        self.mt = ModelAndTokenizer(
            self.model_name,
            torch_dtype=(torch.float16 if "20b" in self.model_name else None),
        )

        if not self.noise_level:
            knowns = KnownsDataset(DATA_DIR)  # Dataset of known facts
            self.noise_level = 3 * collect_embedding_std(self.mt, [k["subject"] for k in knowns])
            print(f"noise level: {self.noise_level}")
            with open(hparams_path, "w") as stream:
                config["noise_level"] = self.noise_level
                yaml.dump(config, stream)

    def get_hidden_flow(self, prompt, subject, kind="mlp"):
        hidden_flow = calculate_hidden_flow(self.mt, prompt, subject, kind=kind, noise=self.noise_level)
        max_effect_tensor = (hidden_flow["scores"]==torch.max(hidden_flow["scores"][hidden_flow["subject_range"][1]-1])).nonzero()[0]
        min_effect_tensor = (hidden_flow["scores"] == torch.min(hidden_flow["scores"][hidden_flow["subject_range"][1] - 1])).nonzero()[0]
        max_effect_layer = int(max_effect_tensor[1])
        min_effect_layer = int(min_effect_tensor[1])
        return max_effect_layer, min_effect_layer, hidden_flow

    def plot_causal_tracing_effect(self, prompt, kind="mlp", hidden_flow=None):
        plot_hidden_flow(self.mt, prompt, kind=kind, result=hidden_flow, savepdf=f"./benchmark/results/tracing/{self.model_name.replace('/', '-')}-{prompt}.png")

if __name__ == '__main__':
    rome_test = RomeTest("google/gemma-2-2b", noise_level=0.15380)
    rome_test.plot_causal_tracing_effect("Steve Jobs was the founder of", kind="mlp")
