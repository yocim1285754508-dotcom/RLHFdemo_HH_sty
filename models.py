# models.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from hf_utils import resolve_model_path
from weave_support import weave_op

class LanguageModelPolicy:
    def __init__(self):
        print(f"Loading models: {Config.MODEL_NAME}...")
        model_path = resolve_model_path(Config.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Active Model (我们要训练的) - 对应表格中的 Policy
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
        ).to(Config.DEVICE)
        self.model.train()
        
        # Reference Model (冻结的，用于计算 KL) - 对应表格中的 Constraints
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
        ).to(Config.DEVICE)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
    @weave_op
    def generate(self, prompt_text):
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(Config.DEVICE)
        with torch.no_grad():
            # 简单的生成采样
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=Config.MAX_GEN_LEN,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response_ids = outputs[0][inputs.input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response_ids, response_text, inputs.input_ids
