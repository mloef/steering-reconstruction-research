import torch
from typing import Optional, Callable

import nnsight
from nnsight.intervention import InterventionProxy

InterventionInterface = Callable[[InterventionProxy], InterventionProxy]

class ObservableLanguageModel:
    def __init__(
        self,
        model: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.dtype = dtype
        self.device = device
        self._original_model = model

        self._model = nnsight.LanguageModel(
            self._original_model,
            device_map=device,
            torch_dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype
        )

        # Quickly run a trace to force model to download due to nnsight lazy download
        input_tokens = self._model.tokenizer.apply_chat_template([{"role": "user", "content": "hello"}])
        with self._model.trace(input_tokens):
          pass

        self.tokenizer = self._model.tokenizer

        self.d_model = self._attempt_to_infer_hidden_layer_dimensions()

        self.safe_mode = False  # Nnsight validation is disabled by default, slows down inference a lot. Turn on to debug.

    def _attempt_to_infer_hidden_layer_dimensions(self):
        config = self._model.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)

        raise Exception(
            "Could not infer hidden number of layer dimensions from model config"
        )

    def _find_module(self, hook_point: str):
        submodules = hook_point.split(".")
        module = self._model
        while submodules:
            module = getattr(module, submodules.pop(0))
        return module

    def forward(
        self,
        inputs: torch.Tensor,
        cache_activations_at: Optional[list[str]] = None,
        interventions: Optional[dict[str, InterventionInterface]] = None,
        use_cache: bool = False, #fixes unknown bug
        past_key_values: Optional[tuple[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor], dict[str, torch.Tensor]]:
        cache: dict[str, torch.Tensor] = {}
        with self._model.trace(
            inputs,
            scan=self.safe_mode,
            validate=self.safe_mode,
            use_cache=use_cache,
            past_key_values=past_key_values,
        ):
            # If we input an intervention
            if interventions:
                for hook_site in interventions.keys():
                    if interventions[hook_site] is None:
                        continue

                    module = self._find_module(hook_site)

                    intervened_acts = interventions[
                        hook_site
                    ](module.output[0])
                    # We only modify module.output[0]
                    if use_cache:
                        module.output = (
                            intervened_acts,
                            module.output[1],
                        )
                    else:
                        module.output = (intervened_acts,)

            if cache_activations_at is not None:
                for hook_point in cache_activations_at:
                    module = self._find_module(hook_point)
                    cache[hook_point] = module.output.save()

            if not past_key_values:
                logits = self._model.output[0][:, -1, :].save()
            else:
                logits = self._model.output[0].squeeze(1).save()

            kv_cache = self._model.output.past_key_values.save()

        return (
            logits.value.detach(),
            kv_cache.value,
            {k: v[0].detach() for k, v in cache.items()},
        )