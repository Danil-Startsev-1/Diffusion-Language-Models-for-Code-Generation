
import json
import math
import os
import sys
import types
import importlib.machinery
from pathlib import Path

import importlib
from transformers import Trainer
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
sys.modules['bitsandbytes'] = None

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model

try:
    import peft.tuners.tuners_utils as _peft_tuners_utils
    if hasattr(_peft_tuners_utils, "_torch_supports_distributed"):
        _peft_tuners_utils._torch_supports_distributed = False
except Exception:
    pass
MODEL_PATH = "/home/dstartsev/models/stable-diffcoder-8b"

CLEANDIR = Path("/home/poalivanova/cleaning_miem_hpc")
D2_PATH = CLEANDIR / "D2" / "data.parquet"

OUTPUT_ROOT = "/home/dstartsev/checkpoint/D2"
RUN_NAME = "stable-diffcoder_D2_1epoch"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, RUN_NAME)

MAX_LENGTH = 256
SEED = 42
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 1
GRAD_ACCUM = 1
SAVE_STEPS = 1000
EVAL_STEPS = 1000
LOGGING_STEPS = 50
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
DATALOADER_NUM_WORKERS = 2

set_seed(SEED)

def build_text(docstring, code, tokenizer):
    doc = (docstring or "").strip()
    code_text = (code or "").strip()

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": doc},
            {"role": "assistant", "content": code_text},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        text = f"""### Task
{doc}

### Solution
{code_text}"""

    return text


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss



class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = build_text(row["docstring"], row["code"], self.tokenizer)

        out = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80, flush=True)
    print("Starting Stable-DiffCoder LoRA fine-tuning on D2", flush=True)
    print(f"Model path: {MODEL_PATH}", flush=True)
    print(f"Dataset path: {D2_PATH}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print("=" * 80, flush=True)

    print("Loading parquet with pandas...", flush=True)
    df = pd.read_parquet(D2_PATH)
    print(f"Total rows before filtering: {len(df)}", flush=True)

    required_cols = ["split", "docstring", "code"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in parquet: {missing_cols}")

    df = df.copy()
    df["docstring"] = df["docstring"].fillna("").astype(str).str.strip()
    df["code"] = df["code"].fillna("").astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip()

    df = df[(df["docstring"] != "") & (df["code"] != "")].copy()

    print(f"Total rows after filtering: {len(df)}", flush=True)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    print(f"Train rows: {len(train_df)}", flush=True)
    print(f"Val rows:   {len(val_df)}", flush=True)
    print(f"Epochs: {NUM_EPOCHS}", flush=True)
    print(f"Max length: {MAX_LENGTH}", flush=True)
    print(f"Train batch size per device: {TRAIN_BATCH_SIZE}", flush=True)
    print(f"Gradient accumulation: {GRAD_ACCUM}", flush=True)
    print(f"Effective train batch size: {TRAIN_BATCH_SIZE * GRAD_ACCUM}", flush=True)

    approx_steps_per_epoch = math.ceil(len(train_df) / (TRAIN_BATCH_SIZE * GRAD_ACCUM))
    print(f"Approx. optimizer steps per epoch: {approx_steps_per_epoch}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model (no 4-bit quantization)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    print("Applying LoRA adapters...", flush=True)
    lora_config = LoraConfig(
    	r=LORA_R,
    	lora_alpha=LORA_ALPHA,
    	lora_dropout=LORA_DROPOUT,
    	bias="none",
    	task_type="CAUSAL_LM",
    	target_modules=[
        	"q_proj",
        	"k_proj",
        	"v_proj",
        	"o_proj",
        	"up_proj",
        	"down_proj",
        	"gate_proj",
    	],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    print("Building torch datasets...", flush=True)
    train_ds = CodeDataset(train_df, tokenizer, MAX_LENGTH)
    val_ds = CodeDataset(val_df, tokenizer, MAX_LENGTH)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
 	output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        run_name=RUN_NAME,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        eval_strategy="no",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=False,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        disable_tqdm=False,
        log_level="info",
        log_on_each_node=False,
    )

    config_to_save = {
        "model_path": MODEL_PATH,
        "dataset_path": str(D2_PATH),
        "output_dir": OUTPUT_DIR,
        "run_name": RUN_NAME,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "effective_train_batch_size": TRAIN_BATCH_SIZE * GRAD_ACCUM,
        "save_steps": SAVE_STEPS,
        "eval_steps": EVAL_STEPS,
        "logging_steps": LOGGING_STEPS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "approx_steps_per_epoch": approx_steps_per_epoch,
    }

    with open(os.path.join(OUTPUT_DIR, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )
    print("Starting training... Progress bar should appear below.", flush=True)
    train_result = trainer.train()

    print("Training finished.", flush=True)
    print(train_result, flush=True)

    final_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    print(f"Saving final adapter to: {final_dir}", flush=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
