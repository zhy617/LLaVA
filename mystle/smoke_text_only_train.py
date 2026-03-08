import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model_name = os.environ.get("SMOKE_MODEL", "sshleifer/tiny-gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.use_cache = False

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=["c_attn"],
)
model = get_peft_model(model, peft_config).to(device)
model.train()

text = "User: hello\nAssistant: hi\n"
batch = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
batch = {k: v.to(device) for k, v in batch.items()}
batch["labels"] = batch["input_ids"].clone()

optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss = model(**batch).loss
loss.backward()
optim.step()

print({"ok": True, "device": device, "loss": float(loss.detach().cpu())})