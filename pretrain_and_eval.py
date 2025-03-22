import json
import os
import torch
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from tqdm.auto import tqdm
from inspect import getfullargspec

from models import *
from util_funs import *

# -----------------------------------------------------------------------------
# I/O
# ====================
# training configurations
# ====================
max_length = 96
num_mem = 16
LLM = "AOC"
# this may change depending on script arguments, so we will reinitialize it later
project_name = "TokenCompression" 
# do not change this when calling the script, just manually edit
root_dir = os.path.dirname(__file__)
train_text_path = f"{root_dir}/data/train.txt"
valid_text_path = f"{root_dir}/data/valid.txt"
# do not change this when calling the script, just manually edit
# this may change depending on script arguments, so we will reinitialize it later
output_dir = f"{root_dir}/results/{project_name}/max{max_length}-mem{num_mem}-{LLM}"
cache_dir = f"{root_dir}/.tokencompression_cache"
llm_path = "meta-llama/Llama-3.2-1B-Instruct" # this should only be llama
deepspeed_config = ""
logging_dir = "results/logs/"
num_train_epochs = 1
per_device_train_batch_size = 16
per_device_eval_batch_size = 64
save_strategy = "steps"
save_steps = 300
evaluation_strategy = "steps"
eval_steps = 100
eval_accumulation_steps = 4
logging_steps = 1
learning_rate = 5e-4
save_total_limit = 1
lr_scheduler_type = "constant_with_warmup"
warmup_steps = 300
device = torch.device(f"cuda")
use_wandb = False
debug = False
# ====================
# prediction configurations
# ====================
overwrite_previous_files = True
prompt = None
context_len = max_length
max_new_tokens = max_length
batch_size = per_device_eval_batch_size
source_text_path = f"{root_dir}/data/test.txt"
regen_path = f"{output_dir}/output.txt"
target_path = f"{output_dir}/target.txt"
status_path = f"{output_dir}/status.txt"
start_line = 0
# ====================
# evaluation configurations
# ====================
eval_output_path = f"{output_dir}/evaluate_metrics.csv"
eval_avg_path = f"{output_dir}/averages.csv"
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open("configurator.py").read())  # overrides from command line or config file
output_dir = f"{root_dir}/results/{project_name}/max{max_length}-mem{num_mem}-{LLM}"
regen_path = f"{output_dir}/output.txt"
target_path = f"{output_dir}/target.txt"
status_path = f"{output_dir}/status.txt"
eval_output_path = f"{output_dir}/evaluate_metrics.csv"
eval_avg_path = f"{output_dir}/averages.csv"
context_len = max_length
max_new_tokens = max_length
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------
assert LLM in ["LoRA_500xCompressor", "LoRA_AOC", "AOC", "ICAE", "ICAE_AOC"], "Expect LLM to be one of the classes in `models.py`"

# create the training and validation datasets
train_dataset = TrainTextDataset(train_text_path, llm_path, max_length, device=device, num_mem=num_mem if 'ICAE' in LLM else 0)
valid_dataset = TrainTextDataset(valid_text_path, llm_path, max_length, device=device, num_mem=num_mem if 'ICAE' in LLM else 0)
print("Dataset created.")

if debug:
    train_dataset.text = train_dataset.text[:(per_device_train_batch_size * 3)]
    save_steps = 2
    with open(source_text_path, "r") as f:
        start_line = len(f.readlines()) - 3
    project_name += '-debug'
    output_dir = f"{root_dir}/results/{project_name}/max{max_length}-mem{num_mem}-{LLM}"
    regen_path = f"{output_dir}/output.txt"
    target_path = f"{output_dir}/target.txt"
    status_path = f"{output_dir}/status.txt"
    eval_output_path = f"{output_dir}/evaluate_metrics.csv"
    eval_avg_path = f"{output_dir}/averages.csv"

if use_wandb:
    import wandb
    wandb.init(project=project_name, name=output_dir.split('/')[-1], group=f"max{max_length}-mem{num_mem}", config=config)

# ====================
# compression model
# ====================

# lora configurations
full_llama_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",]
lora_config = LoraConfig(
    r=64,
    target_modules=full_llama_target_modules,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("Loading enc + llm ...")
model_args = {
    "llm_path":llm_path,
    "cache_dir":cache_dir,
    "max_length":max_length,
    "num_mem":num_mem,
    "device":device,
    "lora_path":None,
}
if "lora_config" in getfullargspec(eval(LLM)).args: model_args["lora_config"] = lora_config
model = eval(LLM)(**model_args)
print(f"Number of trainable parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print("Model is on CUDA device:", torch.cuda.current_device())
model.config = model.llm.config
print("model.llm.config: ", model.llm.config)
print("enc + llm loaded successfully.")

# ====================
# Training
# ====================
## give the detailed information for the error
# torch.autograd.set_detect_anomaly(True)

# training parameters
training_args = TrainingArguments(
    output_dir=output_dir,          
    overwrite_output_dir=False,
    num_train_epochs=num_train_epochs,              
    per_device_train_batch_size=per_device_train_batch_size,   
    per_device_eval_batch_size=per_device_eval_batch_size, 
    save_strategy=save_strategy,
    save_steps=save_steps,      
    evaluation_strategy=evaluation_strategy,    
    eval_steps=eval_steps, 
    eval_accumulation_steps=eval_accumulation_steps,
    logging_dir=logging_dir,    
    logging_steps=logging_steps,
    deepspeed=deepspeed_config,
    learning_rate=learning_rate,
    save_total_limit=save_total_limit,
    lr_scheduler_type=lr_scheduler_type,
    warmup_steps=warmup_steps,
    bf16=True,
    report_to="wandb" if use_wandb else "none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()

with open(f"{output_dir}/config.json", "w") as f:
    json.dump(config, f)

evaluation_results = trainer.evaluate()
print("evaluation_results: ", evaluation_results)

##-- Begin Prediction Code --------
dataset = PredictionTextDataset(source_text_path, start_line=start_line)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model.to(device).eval()
try:
    print(f"{sum(p.numel() for p in model.encoder.parameters()):,} parameters in encoder")
except:
    print(f"model.encoder doesn't exist for {LLM}")
print(f"{sum(p.numel() for p in model.llm.parameters()):,} parameters in decoder llm")

# whether to overwrite_previous_files the output file before prediction
if overwrite_previous_files:
    with open(regen_path, 'w') as file:
        pass

    with open(target_path, 'w') as file:
        pass

    with open(status_path, 'w') as file:
        pass

# all_inputs = []
# all_outputs = []
# number of data records that stop automatically
n_auto_stop = 0
with torch.no_grad():
    for i,batch_texts in enumerate(pbar:=tqdm(data_loader)):
        # to store context tokens to be compressed
        back_tokens = torch.full(
            (len(batch_texts), context_len,), 
            model.tokenizer.eos_token_id, 
            dtype=torch.long
        )

        # context tokens to be compressed
        text_tokens = model.tokenizer(
            batch_texts, 
            truncation=True, 
            padding="max_length",
            max_length=max_length, 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids

        # store the context tokens to be compressed
        back_tokens[:,:max_length] = text_tokens

        # target text (original context)
        target_text = [model.tokenizer.decode(
            tt, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False,
        ) for tt in text_tokens]

        # record the target text (original context)
        target_text = [t.replace("\n", " ").strip() for t in target_text]
        # all_inputs.extend(target_text)

        # compress the context to compressed tokens (K V values)
        compressed_information = model.compress(
            text=batch_texts, 
            text_tokens=back_tokens, 
            output_path=None
        )

        # regenerate the context based on the compressed tokens (K V values)
        predicted_text, end, generated_token_length = model.predict(
            compressed_information, 
            max_new_tokens=max_new_tokens, 
            prompt=prompt
        )
        
        # whether the regeneration stops automatically
        n_auto_stop += end

        # record the regenerated text
        predicted_text = [t.replace("\n", " ").strip() for t in predicted_text]
        # all_outputs.extend(predicted_text)

        with open(target_path, 'a', encoding='utf-8') as file:
            file.write('\n'.join(target_text))

        with open(regen_path, 'a', encoding='utf-8') as file:
            file.write('\n'.join(predicted_text))

        pbar.set_description(f"{n_auto_stop} automatically stopped/{(i+1)*batch_size} total processed regenerations.")

# record the number of data records that stop automatically
with open(status_path, "a") as file:
    file.write(f"{n_auto_stop} auto stopped/{len(dataset)} examples")

print(f"---\nFinished Prediction.\nProject name: {project_name}.\nOutput will be found at {regen_path}\n---\n")

##-- Begin Evaluation Code --------
main_evaluate(target_path, regen_path, eval_output_path)
averages = calculate_metrics_averages(eval_output_path)
print("Average metrics:", averages, sep="\n")
averages.to_csv(eval_avg_path, header=True)
print(f"---\nFinished Evaluation.\nProject name: {project_name}.\nOutput will be found at {regen_path}\n---\n")
