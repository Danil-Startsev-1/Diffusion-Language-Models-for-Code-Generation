from pathlib import Path
import json
import textwrap
import gzip
import ast
import time
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import pandas as pd
import signal

import peft.tuners.tuners_utils as _peft_tuners_utils
if hasattr(_peft_tuners_utils, "_torch_supports_distributed"):
    _peft_tuners_utils._torch_supports_distributed = False


DATA_PATH = Path("/home/dstartsev/dataset/HumanEval/human-eval/data") / "HumanEval.jsonl.gz"

BASE_MODEL_PATH = "/home/dstartsev/models/stable-diffcoder-8b"
ADAPTER_FINAL_DIR = "/home/dstartsev/checkpoint/D3/stable-diffcoder_D3_1epoch/final_adapter"

SAVE_DIR = Path("/home/dstartsev/checkpoint/D3/test")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURE = 0.1
TOP_P = 0.95
ALG_TEMP = 0.0
EOS_PENALTY = 3.0

steps_list = [16, 32, 64, 128, 256, 512]
max_new_list = [256, 512]

checkpoint_path_stable = SAVE_DIR / "humaneval_stable_diffcoder_grid_checkpoint.csv"
per_sample_path_stable = SAVE_DIR / "humaneval_stable_diffcoder_samples.jsonl"
final_path_stable = SAVE_DIR / "humaneval_stable_diffcoder_grid_final.csv"


def load_humaneval_tasks(path: Path):
    tasks = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    print("Total tasks:", len(tasks))
    if tasks:
        print("Example of fields:", tasks[0].keys())
        print("The first task_id:", tasks[0]["task_id"])
    return tasks

def run_tests_for_task(task: Dict[str, Any], gen_code: str) -> bool:
    prompt = task["prompt"]
    tests = task["test"]
    entry_point = task["entry_point"]
    full_src = prompt + "\n" + gen_code + "\n" + tests

    ns: Dict[str, Any] = {}
    try:
        exec(textwrap.dedent(full_src), ns)
    except Exception as e:
        print(f"[EXEC ERROR] {task['task_id']}: {e}")
        return False

    if "check" not in ns or entry_point not in ns:
        print(f"[NS MISSING] {task['task_id']}: 'check' or '{entry_point}' not in namespace")
        return False

    try:
        signal.alarm(3)
        ns["check"](ns[entry_point])
        signal.alarm(0)
        return True
    except Exception as e:
        signal.alarm(0)
        print(f"[CHECK ERROR] {task['task_id']}: {e}")
        return False


def clean_and_check(code: str):
    code = textwrap.dedent(code).strip()
    lines = code.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            start = i
            break
    if start is not None:
        code = "\n".join(lines[start:])
    try:
        ast.parse(code)
        return code, True
    except SyntaxError:
        return code, False


def extract_first_function(text: str) -> str:
    lines = text.splitlines()

    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            start = i
            break
    if start is None:
        return text

    func_lines = [lines[start]]
    base_indent = len(lines[start]) - len(lines[start].lstrip())

    for line in lines[start + 1:]:
        if line.strip() == "":
            func_lines.append(line)
            continue

        indent = len(line) - len(line.lstrip())
        if indent > base_indent:
            func_lines.append(line)
        else:
            break

    return "\n".join(func_lines)


def load_stable_model_with_adapter():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = AutoModelForCausalLM.from_pretrained(
        ADAPTER_FINAL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_FINAL_DIR,
        trust_remote_code=True,
    )

    model.eval()
    print("LoRA-adapted Stable-DiffCoder D3 loaded from", ADAPTER_FINAL_DIR)
    return model, tokenizer


def make_prompt_stable(tokenizer_stable, prompt_text: str):
    messages = [{"role": "user", "content": prompt_text.strip()}]
    return tokenizer_stable.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )


def generate_stable_params(model_stable, tokenizer_stable, prompt: str, steps: int, max_new: int):
    block_length = 4
    gen_length = max_new - (max_new % block_length)
    if gen_length == 0:
        gen_length = block_length

    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        steps = num_blocks * max(1, steps // num_blocks)

    inputs = make_prompt_stable(tokenizer_stable, prompt)
    input_ids = inputs["input_ids"].to(model_stable.device)

    with torch.no_grad():
        out_ids = model_stable.generate(
            input_ids=input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=TEMPERATURE,
            remasking="low_confidence",
            tokenizer=tokenizer_stable,
            shift=False,
            threshold=None,
            eos_id=tokenizer_stable.eos_token_id,
        )

    input_len = input_ids.shape[1]
    gen_ids = out_ids[0, input_len:]
    gen_text = tokenizer_stable.decode(gen_ids, skip_special_tokens=True)

    if tokenizer_stable.eos_token is not None:
        gen_text = gen_text.split(tokenizer_stable.eos_token)[0]

    gen_text = extract_first_function(gen_text)
    code, _ = clean_and_check(gen_text)
    return code, gen_ids.shape[-1]


def main():
    tasks = load_humaneval_tasks(DATA_PATH)
    model_stable, tokenizer_stable = load_stable_model_with_adapter()

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    all_rows_stable = []

    open(per_sample_path_stable, "w", encoding="utf-8").close()

    CHUNK_SIZE = 40

    for steps in steps_list:
        for max_new in max_new_list:
            print(f"\n=== [Stable-DiffCoder] STEPS = {steps}, MAX_NEW = {max_new} ===")

            n_tasks = 0
            n_pass = 0
            n_ast_ok = 0
            total_len_tokens = 0
            n_trunc = 0

            t_start = time.time()

            for chunk_idx, task_chunk in enumerate(chunks(tasks, CHUNK_SIZE)):
                print(f"\n--- Chunk {chunk_idx} ({len(task_chunk)} tasks) ---")

                for task in tqdm(task_chunk,
                                 desc=f"steps={steps}, max_new={max_new}, chunk={chunk_idx}"):
                    n_tasks += 1
                    task_id = task["task_id"]
                    prompt = task["prompt"]

                    gen_code, gen_len_tokens = generate_stable_params(
                        model_stable, tokenizer_stable, prompt, steps, max_new
                    )

                    try:
                        ast.parse(gen_code)
                        ast_ok = True
                    except SyntaxError as e:
                        print(f"[AST ERROR] {task_id} @ steps={steps}, max_new={max_new}: {e}")
                        ast_ok = False

                    if ast_ok:
                        n_ast_ok += 1
                        if gen_len_tokens >= max_new:
                            n_trunc += 1

                    total_len_tokens += gen_len_tokens

                    passed = run_tests_for_task(task, gen_code) if ast_ok else False
                    if passed:
                        n_pass += 1

                    print(f"{task_id}: {'Correct' if passed else 'Incorrect'}")

                    with open(per_sample_path_stable, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "model": "stable-diffcoder-8b",
                            "steps": steps,
                            "max_new_tokens": max_new,
                            "task_id": task_id,
                            "ast_ok": ast_ok,
                            "passed": passed,
                            "gen_len_tokens": gen_len_tokens,
                        }, ensure_ascii=False) + "\n")

            
                wall_time_so_far = time.time() - t_start
                pass_at_1_so_far = n_pass / n_tasks if n_tasks > 0 else 0.0
                ast_validity_so_far = n_ast_ok / n_tasks if n_tasks > 0 else 0.0
                trunc_rate_so_far = n_trunc / n_tasks if n_tasks > 0 else 0.0
                avg_len_tokens_so_far = total_len_tokens / n_tasks if n_tasks > 0 else 0.0
                avg_time_per_task_so_far = wall_time_so_far / n_tasks if n_tasks > 0 else 0.0

                row_partial = {
                    "model": "stable-diffcoder-8b",
                    "steps": steps,
                    "max_new_tokens": max_new,
                    "num_tasks": n_tasks,
                    "pass_at_1": pass_at_1_so_far,
                    "ast_validity": ast_validity_so_far,
                    "truncation_rate": trunc_rate_so_far,
                    "avg_len_tokens": avg_len_tokens_so_far,
                    "avg_time_per_task_sec": avg_time_per_task_so_far,
                    "total_time_sec": wall_time_so_far,
                    "chunk_idx": chunk_idx,
                    "finished": False,
                }
                all_rows_stable.append(row_partial)
                pd.DataFrame(all_rows_stable).to_csv(checkpoint_path_stable, index=False)
                print(f"Partial checkpoint saved after chunk {chunk_idx} to {checkpoint_path_stable}")
        

            t_end = time.time()
            wall_time = t_end - t_start

            pass_at_1 = n_pass / n_tasks if n_tasks > 0 else 0.0
            ast_validity = n_ast_ok / n_tasks if n_tasks > 0 else 0.0
            trunc_rate = n_trunc / n_tasks if n_tasks > 0 else 0.0
            avg_len_tokens = total_len_tokens / n_tasks if n_tasks > 0 else 0.0
            avg_time_per_task = wall_time / n_tasks if n_tasks > 0 else 0.0

            print(
                f"RESULTS steps={steps}, max_new={max_new}: "
                f"Pass@1={pass_at_1:.3f}, AST-validity={ast_validity:.3f}, "
                f"Trunc={trunc_rate:.3f}, avg_len={avg_len_tokens:.1f}, "
                f"time/task={avg_time_per_task:.2f}s"
            )

            row_final = {
                "model": "stable-diffcoder-8b",
                "steps": steps,
                "max_new_tokens": max_new,
                "num_tasks": n_tasks,
                "pass_at_1": pass_at_1,
                "ast_validity": ast_validity,
                "truncation_rate": trunc_rate,
                "avg_len_tokens": avg_len_tokens,
                "avg_time_per_task_sec": avg_time_per_task,
                "total_time_sec": wall_time,
                "chunk_idx": chunk_idx,
                "finished": True,
            }
            all_rows_stable.append(row_final)
            pd.DataFrame(all_rows_stable).to_csv(checkpoint_path_stable, index=False)
            print(f"Checkpoint saved to {checkpoint_path_stable}")

    pd.DataFrame(all_rows_stable).to_csv(final_path_stable, index=False)
    print(f"Final saved to {final_path_stable}")

if __name__ == "__main__":
    main()