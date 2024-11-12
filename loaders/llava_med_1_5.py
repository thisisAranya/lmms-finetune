from typing import Tuple

from transformers import AutoProcessor, LlavaMistralForCausalLM, PreTrainedTokenizer

from . import register_loader
from .base import BaseModelLoader


@register_loader("llava-med-1.5")
class LLaVA15ModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[LlavaMistralForCausalLM, PreTrainedTokenizer, AutoProcessor]:
        if load_model:
            model = LlavaMistralForCausalLM.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            )
            model.config.hidden_size = model.language_model.config.hidden_size # useful for deepspeed
        else:
            model = None

        processor = AutoProcessor.from_pretrained(self.model_hf_path)
        tokenizer = processor.tokenizer
        return model, tokenizer, processor



    
    # if 'llava' in model_name.lower():
    #     # Load LLaVA model
    #         if 'mistral' in model_name.lower():
    #             tokenizer = AutoTokenizer.from_pretrained(model_path)
    #             model = LlavaMistralForCausalLM.from_pretrained(
    #                 model_path,
    #                 low_cpu_mem_usage=False,
    #                 use_flash_attention_2=False,
    #                 **kwargs
    #             )
