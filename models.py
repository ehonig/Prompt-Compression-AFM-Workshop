import torch
import torch.nn as nn
from peft import get_peft_model
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_lora_parameters(model, lora_params_path):
    nloaded = 0
    with safe_open(lora_params_path, framework="pt", device=next(model.parameters()).device.index) as f:
        with torch.no_grad():
            for nparams, (name, param) in enumerate(model.named_parameters()):
                if name in f.keys():
                    param.copy_(f.get_tensor(name))
                    print(f"Loaded parameter {name}")
                    nloaded += 1
                else:
                    print(f"No saved parameter for {name}")
    print(f"loaded {nloaded}/{nparams+1} named parameters")

class BaseModel(nn.Module):
    def __init__(self, llm_path, cache_dir, max_length, num_mem, device, lora_path=None):
        super().__init__()
        self.eos_2_id = 128001
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
        for param in self.llm.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.num_mem = num_mem
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, self.llm.config.hidden_size, dtype=torch.bfloat16).to(device), requires_grad=True)
        if lora_path:
            load_lora_parameters(self, lora_path)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def encode(self, text, text_tokens=None, output_hidden_states=False):
        if text_tokens is None:
            text_tokens = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        else:
            text_tokens = text_tokens.to(self.device)
        
        text_tok_embeddings = self.llm.get_input_embeddings()(text_tokens)
        memory_tok_embeddings = self.memory_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1).to(self.device)
        encoder_input_embeddings = torch.cat((text_tok_embeddings, memory_tok_embeddings), dim=1)
        encoder_output = self.encoder(inputs_embeds=encoder_input_embeddings, output_hidden_states=output_hidden_states)
        return encoder_output

    def compress(self, text, text_tokens=None, output_path=None):
        encoder_output = self.encode(text, text_tokens)
        past_key_values = encoder_output.past_key_values
        trimmed_past_key_values = tuple((layer_key[:, :, -self.num_mem:, :], layer_value[:, :, -self.num_mem:, :]) for layer_key, layer_value in past_key_values)

        if output_path:
            torch.save(trimmed_past_key_values, output_path)
            print(f"Saved compressed past_key_values to {output_path}")

        return trimmed_past_key_values

    def decode(self, decoder_input_embeddings, past_key_values=None, attention_mask=None):
        if hasattr(self.llm, 'disable_adapter'):
            with self.llm.disable_adapter():
                decoder_output = self.llm(inputs_embeds=decoder_input_embeddings, past_key_values=past_key_values, attention_mask=attention_mask)
        else:
            decoder_output = self.llm(inputs_embeds=decoder_input_embeddings, past_key_values=past_key_values, attention_mask=attention_mask)
        
        return decoder_output

    def forward(self, input_ids, labels):
        trimmed_past_key_values = self.compress(None, input_ids)
        prompt_tokens = torch.tensor([self.tokenizer.bos_token_id], device=self.device)
        prompt_tok_embeddings = self.llm.get_input_embeddings()(prompt_tokens).repeat(input_ids.shape[0], 1, 1)
        decoder_input_embeddings = torch.cat((prompt_tok_embeddings, self.llm.get_input_embeddings()(input_ids)), dim=1)
        decoder_output = self.decode(decoder_input_embeddings, trimmed_past_key_values)
        loss = self.criterion(decoder_output.logits.view(-1, decoder_output.logits.size(-1)), labels.view(-1))
        return {'loss': loss, 'logits': decoder_output.logits}

    def predict(self, past_key_values, max_new_tokens, prompt=None):
        B = past_key_values[0][0].size(0)
        end = torch.zeros((B,), dtype=torch.long, device=self.device)
        if prompt is None or prompt == "bos":
            input_tokens = torch.tensor([[self.tokenizer.bos_token_id]]*B, device=self.device)
            attention_mask = None
        else:
            input_tokens, temp_attention_mask = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False, padding=True).to(self.device).values()
            attention_mask = torch.ones((B, self.num_mem+temp_attention_mask.shape[-1]), dtype=torch.long, device=self.device)
            attention_mask[:, self.num_mem:] = temp_attention_mask
        assert len(input_tokens) == B

        generated_text = []
        for i in range(max_new_tokens):
            decoder_input_embeddings = self.llm.get_input_embeddings()(input_tokens)
            decoder_output = self.decode(decoder_input_embeddings, past_key_values, attention_mask)
            logits, past_key_values = decoder_output.logits, decoder_output.past_key_values
            if attention_mask is not None:
                attention_mask = torch.concat((attention_mask, torch.ones(attention_mask.shape[0], 1, dtype=attention_mask.dtype, device=self.device)), dim=1)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            end += i*(end == 0).logical_and((next_token.squeeze() == self.eos_2_id).logical_or(next_token.squeeze() == self.tokenizer.eos_token_id))
            if end.all().item():
                break
            generated_text.append(next_token)
            input_tokens = next_token

        generated_token_length = end
        num_auto_stop = (sum(end > 0) - sum(end == max_new_tokens)).item()
        generated_text = torch.hstack(generated_text)
        generated_text = [self.tokenizer.decode(gt, skip_special_tokens=True, clean_up_tokenization_spaces=False) for gt in generated_text]

        return generated_text, num_auto_stop, generated_token_length

class LoRA_500xCompressor(BaseModel):
    def __init__(self, llm_path, cache_dir, max_length, lora_path, lora_config, num_mem, device):
        super().__init__(llm_path, cache_dir, max_length, num_mem, device, lora_path)
        self.llm = get_peft_model(self.llm, lora_config)
        for name, param in self.llm.named_parameters():
            param.requires_grad = 'lora' in name
        self.encoder = self.llm
        if lora_path:
            del self.llm
            load_lora_parameters(self, lora_path)
            self.llm = self.encoder

class LoRA_AOC(BaseModel):
    def __init__(self, llm_path, cache_dir, max_length, lora_path, lora_config, num_mem, device):
        super().__init__(llm_path, cache_dir, max_length, num_mem, device, lora_path)
        encoder = AutoModelForCausalLM.from_pretrained(llm_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
        for layer in next(encoder.children()).layers:
            layer.mlp = nn.Identity()
        self.encoder = get_peft_model(encoder, peft_config=lora_config)
        for name, param in self.encoder.named_parameters():
            param.requires_grad = 'lora' in name
        if lora_path:
            load_lora_parameters(self, lora_path)
        torch.cuda.empty_cache()

class AOC(BaseModel):
    def __init__(self, llm_path, cache_dir, max_length, lora_path, num_mem, device):
        super().__init__(llm_path, cache_dir, max_length, num_mem, device, lora_path)
        self.encoder = AutoModelForCausalLM.from_pretrained(llm_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
        for layer in next(self.encoder.children()).layers:
            layer.mlp = nn.Identity()
        for param in self.encoder.parameters():
            param.requires_grad = True
        if lora_path:
            load_lora_parameters(self, lora_path)
        torch.cuda.empty_cache()

class ICAE(BaseModel):
    def __init__(self, llm_path, cache_dir, max_length, lora_path, lora_config, num_mem, device):
        super().__init__(llm_path, cache_dir, max_length, num_mem, device, lora_path)
        self.llm = get_peft_model(self.llm, lora_config)
        for name, param in self.llm.named_parameters():
            param.requires_grad = 'lora' in name
        self.encoder = self.llm
        self.ae_embedding = nn.Parameter(torch.randn(1, 1, self.llm.config.hidden_size, dtype=torch.bfloat16).to(device), requires_grad=True)
        if lora_path:
            del self.llm
            load_lora_parameters(self, lora_path)
            self.llm = self.encoder

    def compress(self, text, text_tokens=None, output_path=None):
        encoder_output = self.encode(text, text_tokens, output_hidden_states=True).hidden_states[-1]
        mem_vec = encoder_output[:, -self.num_mem:, :]

        if output_path:
            torch.save(mem_vec, output_path)
            print(f"Saved compressed past_key_values to {output_path}")

        return mem_vec

    def forward(self, input_ids, labels):
        mem_vec = self.compress(None, input_ids)
        prompt_tok_embeddings = self.ae_embedding.repeat(input_ids.shape[0], 1, 1).to(self.device)
        decoder_input_embeddings = torch.cat((mem_vec, prompt_tok_embeddings, self.llm.get_input_embeddings()(input_ids)), dim=1)
        decoder_output = self.decode(decoder_input_embeddings, None)
        loss = self.criterion(decoder_output.logits.view(-1, decoder_output.logits.size(-1)), labels.view(-1))
        return {'loss': loss, 'logits': decoder_output.logits}

    def predict(self, mem_vec, max_new_tokens, prompt=None):
        B, M, h = mem_vec.shape
        end = torch.zeros((B,), dtype=torch.long, device=self.device)
        prompt_embedding = self.ae_embedding if (prompt is None or prompt == "ae") else self.llm.get_input_embeddings()(self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(self.device))
        prompt_embedding = prompt_embedding.repeat(B, 1, 1)
        decoder_input_embeddings = torch.cat((mem_vec, prompt_embedding), dim=1)
        past_key_values = None

        generated_text = []
        for i in range(max_new_tokens):
            decoder_output = self.decode(decoder_input_embeddings, past_key_values)
            logits, past_key_values = decoder_output.logits, decoder_output.past_key_values
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            end += i*(end == 0).logical_and((next_token.squeeze() == self.eos_2_id).logical_or(next_token.squeeze() == self.tokenizer.eos_token_id))
            if end.all().item():
                break
            generated_text.append(next_token)
            input_tokens = next_token
            decoder_input_embeddings = self.llm.get_input_embeddings()(input_tokens)

        generated_token_length = end
        num_auto_stop = (sum(end > 0) - sum(end == max_new_tokens)).item()
        generated_text = torch.hstack(generated_text)
        generated_text = [self.tokenizer.decode(gt, skip_special_tokens=True, clean_up_tokenization_spaces=False) for gt in generated_text]

        return generated_text, num_auto_stop, generated_token_length

class ICAE_AOC(ICAE):
    def __init__(self, llm_path, cache_dir, max_length, lora_path, lora_config, num_mem, device):
        super().__init__(llm_path, cache_dir, max_length, lora_path, lora_config, num_mem, device)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
        for param in self.llm.parameters():
            param.requires_grad = False
        self.encoder = AutoModelForCausalLM.from_pretrained(llm_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
        for layer in next(self.encoder.children()).layers:
            layer.mlp = nn.Identity()
        for param in self.encoder.parameters():
            param.requires_grad = True
        if lora_path:
            load_lora_parameters(self, lora_path)
        torch.cuda.empty_cache()
