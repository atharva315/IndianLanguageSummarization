
# ============================================================
# MR-FT-001 — DDP TRAINING WORKER
# ============================================================

import os
import gc
import json
import math
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import bitsandbytes as bnb

from kaggle_secrets import UserSecretsClient

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from peft import PeftModel


# ============================================================
# 1. DISTRIBUTED ENVIRONMENT
# ============================================================

LOCAL_RANK = int(
    os.environ["LOCAL_RANK"]
)

RANK = int(
    os.environ["RANK"]
)

WORLD_SIZE = int(
    os.environ["WORLD_SIZE"]
)

if WORLD_SIZE != 2:

    raise RuntimeError(
        f"Expected WORLD_SIZE=2, "
        f"got {WORLD_SIZE}"
    )

if LOCAL_RANK not in [0, 1]:

    raise RuntimeError(
        f"Unexpected LOCAL_RANK={LOCAL_RANK}"
    )

torch.cuda.set_device(
    LOCAL_RANK
)

DEVICE = torch.device(
    f"cuda:{LOCAL_RANK}"
)


# ============================================================
# 2. INITIALIZE NCCL
# ============================================================

dist.init_process_group(
    backend="nccl",
    init_method="env://",
)

dist.barrier()


# ============================================================
# 3. EXPERIMENT CONSTANTS
# ============================================================

MODEL_ID = (
    "google/gemma-4-E2B-it"
)

MODEL_REVISION = (
    "b515064b63ff28985d549455f7709f112e8a5e39"
)

OUTPUT_DIR = Path(
    "/kaggle/working/results/"
    "qlora_MR-FT-001"
)

TRAIN_FILE = Path(
    "/kaggle/working/data/"
    "marathi_v1/04_train_sft.csv"
)

VAL_FILE = Path(
    "/kaggle/working/data/"
    "marathi_v1/05_validation_sft.csv"
)

MAX_SEQ_LENGTH = 768

PER_DEVICE_TRAIN_BATCH_SIZE = 1

PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 1e-4

WARMUP_STEPS = 263

TOTAL_STEPS = 2631

NUM_EPOCHS = 3

MAX_GRAD_NORM = 1.0

SEED = 42


# ============================================================
# 4. SELECT LATEST CHECKPOINT
# ============================================================

def checkpoint_number(path):

    try:

        return int(
            path.name.split("-")[-1]
        )

    except Exception:

        return -1


checkpoints = sorted(
    [
        p
        for p in OUTPUT_DIR.glob(
            "checkpoint-*"
        )
        if (
            p.is_dir()
            and p.name.split("-")[-1].isdigit()
        )
    ],
    key=checkpoint_number,
)

if not checkpoints:

    raise RuntimeError(
        "No MR-FT-001 checkpoints found."
    )

CHECKPOINT_DIR = checkpoints[-1]


# ============================================================
# 5. READ CHECKPOINT STATE
# ============================================================

with open(
    CHECKPOINT_DIR / "trainer_state.json",
    "r",
    encoding="utf-8",
) as f:

    saved_state = json.load(f)

START_STEP = int(
    saved_state["global_step"]
)

START_EPOCH = float(
    saved_state["epoch"]
)

SAVED_MAX_STEPS = int(
    saved_state["max_steps"]
)

BEST_EVAL_LOSS = (
    saved_state.get(
        "best_metric"
    )
)

BEST_CHECKPOINT = (
    saved_state.get(
        "best_model_checkpoint"
    )
)

LOG_HISTORY = list(
    saved_state.get(
        "log_history",
        []
    )
)

if START_STEP < 750:

    raise RuntimeError(
        f"Checkpoint step {START_STEP} "
        "is older than checkpoint-750."
    )

if SAVED_MAX_STEPS != TOTAL_STEPS:

    raise RuntimeError(
        f"Checkpoint max_steps="
        f"{SAVED_MAX_STEPS}; "
        f"expected {TOTAL_STEPS}"
    )


# ============================================================
# 6. DISPLAY WORKER
# ============================================================

print("\n" + "=" * 80)

print(
    f"MR-FT-001 WORKER "
    f"RANK={RANK} "
    f"LOCAL_RANK={LOCAL_RANK}"
)

print("=" * 80)

print("\nWorld size:")
print(WORLD_SIZE)

print("\nGPU:")
print(
    torch.cuda.get_device_name(
        LOCAL_RANK
    )
)

print("\nDevice:")
print(DEVICE)

if RANK == 0:

    print("\n" + "-" * 80)
    print("CHECKPOINT")
    print("-" * 80)

    print("\nSelected checkpoint:")
    print(CHECKPOINT_DIR)

    print("\nSaved step:")
    print(START_STEP)

    print("\nSaved epoch:")
    print(START_EPOCH)

    print("\nMax steps:")
    print(SAVED_MAX_STEPS)

    print("\nBest eval loss:")
    print(BEST_EVAL_LOSS)


# ============================================================
# 7. LOAD DATA
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE
)

val_df = pd.read_csv(
    VAL_FILE
)

if len(train_df) != 14020:

    raise RuntimeError(
        "Training size is not 14020."
    )

if len(val_df) != 796:

    raise RuntimeError(
        "Validation size is not 796."
    )

if RANK == 0:

    print("\nTrain rows:")
    print(len(train_df))

    print("\nValidation rows:")
    print(len(val_df))


# ============================================================
# 8. HF TOKEN
# ============================================================

hf_token = (
    UserSecretsClient()
    .get_secret(
        "HF_TOKEN"
    )
)

if not hf_token:

    raise RuntimeError(
        "HF_TOKEN unavailable."
    )


# ============================================================
# 9. LOAD PROCESSOR
# ============================================================

processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=hf_token,
    )
)

tokenizer = processor.tokenizer

if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

tokenizer.padding_side = "right"


# ============================================================
# 10. EXACT COMPLETION-ONLY COLLATOR
# ============================================================

class GemmaCompletionOnlyCollator:

    def __init__(
        self,
        processor,
        max_length=768,
    ):

        self.processor = processor

        self.tokenizer = (
            processor.tokenizer
        )

        self.max_length = max_length

    def build_messages(
        self,
        prompt,
        completion=None,
    ):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": str(
                            prompt
                        ),
                    }
                ],
            }
        ]

        if completion is not None:

            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": str(
                                completion
                            ),
                        }
                    ],
                }
            )

        return messages

    def __call__(
        self,
        examples,
    ):

        inputs = []
        labels_list = []

        for example in examples:

            prompt = str(
                example["prompt"]
            )

            completion = str(
                example["completion"]
            )

            full = (
                self.processor
                .apply_chat_template(
                    self.build_messages(
                        prompt,
                        completion,
                    ),
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    return_dict=True,
                )
            )

            prompt_only = (
                self.processor
                .apply_chat_template(
                    self.build_messages(
                        prompt
                    ),
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            )

            full_ids = (
                full["input_ids"][0]
            )

            prompt_ids = (
                prompt_only[
                    "input_ids"
                ][0]
            )

            prompt_length = (
                prompt_ids.shape[0]
            )

            full_length = (
                full_ids.shape[0]
            )

            if not torch.equal(
                full_ids[:prompt_length],
                prompt_ids,
            ):

                raise RuntimeError(
                    "Prompt prefix mismatch."
                )

            if full_length > (
                self.max_length
            ):

                raise RuntimeError(
                    f"Sequence length "
                    f"{full_length} exceeds "
                    f"{self.max_length}"
                )

            ids = (
                full_ids.tolist()
            )

            labels = (
                ids.copy()
            )

            for i in range(
                prompt_length
            ):

                labels[i] = -100

            if not any(
                x != -100
                for x in labels
            ):

                raise RuntimeError(
                    "No active loss tokens."
                )

            inputs.append(
                {
                    "input_ids":
                        ids
                }
            )

            labels_list.append(
                {
                    "input_ids":
                        labels
                }
            )

        batch = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
        )

        label_batch = (
            self.tokenizer.pad(
                labels_list,
                padding=True,
                return_tensors="pt",
            )
        )

        batch["labels"] = (
            label_batch[
                "input_ids"
            ]
        )

        batch["labels"][
            batch["attention_mask"] == 0
        ] = -100

        return batch


data_collator = (
    GemmaCompletionOnlyCollator(
        processor,
        MAX_SEQ_LENGTH,
    )
)


# ============================================================
# 11. LOAD EXACT 4-BIT GEMMA
# ============================================================

gc.collect()

torch.cuda.empty_cache()

bnb_config = (
    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(
            torch.float16
        ),
    )
)

if RANK == 0:

    print("\n" + "-" * 80)
    print("LOADING BASE GEMMA 4")
    print("-" * 80)

base_model = (
    AutoModelForMultimodalLM
    .from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=hf_token,
        quantization_config=bnb_config,
        device_map={
            "": LOCAL_RANK
        },
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
)

if hasattr(
    base_model.config,
    "use_cache",
):

    base_model.config.use_cache = False


# ============================================================
# 12. LOAD SAVED LORA
# ============================================================

model = (
    PeftModel
    .from_pretrained(
        base_model,
        str(CHECKPOINT_DIR),
        is_trainable=True,
    )
)

model.train()

if hasattr(
    model,
    "enable_input_require_grads",
):

    model.enable_input_require_grads()

if hasattr(
    model,
    "gradient_checkpointing_enable",
):

    try:

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            }
        )

    except TypeError:

        model.gradient_checkpointing_enable()


# ============================================================
# 13. MODEL REPLICA IDENTICALITY CHECK
# ============================================================
#
# DDP init_sync=False is safe only because both workers
# independently loaded the same pinned base + same adapter.
#
# We verify a small deterministic checksum before wrapping.
# ============================================================

checksum_sum = torch.zeros(
    1,
    dtype=torch.float64,
    device=DEVICE,
)

checksum_sq = torch.zeros(
    1,
    dtype=torch.float64,
    device=DEVICE,
)

for name, param in (
    model.named_parameters()
):

    if (
        param.requires_grad
        and "lora_" in name.lower()
    ):

        values = (
            param.detach()
            .float()
        )

        checksum_sum += (
            values.sum(
                dtype=torch.float64
            )
        )

        checksum_sq += (
            (
                values * values
            ).sum(
                dtype=torch.float64
            )
        )

dist.all_reduce(
    checksum_sum,
    op=dist.ReduceOp.SUM,
)

dist.all_reduce(
    checksum_sq,
    op=dist.ReduceOp.SUM,
)

expected_sum = (
    checksum_sum.item()
    / WORLD_SIZE
)

expected_sq = (
    checksum_sq.item()
    / WORLD_SIZE
)

# The reduced mean checksum must match the local checksum.
local_sum = 0.0
local_sq = 0.0

for name, param in (
    model.named_parameters()
):

    if (
        param.requires_grad
        and "lora_" in name.lower()
    ):

        values = (
            param.detach()
            .float()
        )

        local_sum += (
            values.sum(
                dtype=torch.float64
            ).item()
        )

        local_sq += (
            (
                values * values
            ).sum(
                dtype=torch.float64
            ).item()
        )

if not np.isclose(
    local_sum,
    expected_sum,
    rtol=1e-7,
    atol=1e-3,
):

    raise RuntimeError(
        "LoRA replicas are not identical "
        "across workers."
    )

if not np.isclose(
    local_sq,
    expected_sq,
    rtol=1e-7,
    atol=1e-3,
):

    raise RuntimeError(
        "LoRA replicas are not identical "
        "across workers."
    )


# ============================================================
# 14. TRAINABLE PARAMETER GATE
# ============================================================

trainable_params = []

trainable_count = 0

trainable_tensor_count = 0

bad_modules = []

for name, param in (
    model.named_parameters()
):

    if param.requires_grad:

        trainable_params.append(
            param
        )

        trainable_count += (
            param.numel()
        )

        trainable_tensor_count += 1

        if (
            "vision_tower" in name
            or "audio_tower" in name
        ):

            bad_modules.append(
                name
            )

if trainable_count != 24_158_208:

    raise RuntimeError(
        f"Expected 24,158,208 trainable "
        f"parameters, got "
        f"{trainable_count}"
    )

if trainable_tensor_count != 410:

    raise RuntimeError(
        f"Expected 410 trainable tensors, "
        f"got {trainable_tensor_count}"
    )

if bad_modules:

    raise RuntimeError(
        "Vision/audio parameters "
        "are trainable."
    )


# ============================================================
# 15. DDP CONSTRUCTOR CAPABILITY CHECK
# ============================================================

import inspect

ddp_signature = inspect.signature(
    DDP
)

if "init_sync" not in (
    ddp_signature.parameters
):

    raise RuntimeError(
        "This PyTorch build does not "
        "support DDP(init_sync=False)."
    )


# ============================================================
# 16. WRAP WITH DDP WITHOUT INITIAL PARAMETER BROADCAST
# ============================================================

if RANK == 0:

    print("\n" + "-" * 80)
    print("CREATING DDP WITHOUT INITIAL BROADCAST")
    print("-" * 80)

model = DDP(
    model,
    device_ids=[
        LOCAL_RANK
    ],
    output_device=LOCAL_RANK,
    broadcast_buffers=False,
    init_sync=False,
    find_unused_parameters=False,
)


# ============================================================
# 17. OPTIMIZER
# ============================================================

optimizer = bnb.optim.AdamW(
    [
        {
            "params":
                trainable_params,
            "weight_decay":
                0.01,
        },
        {
            "params":
                [],
            "weight_decay":
                0.0,
        },
    ],
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-8,
    optim_bits=8,
    is_paged=True,
)


# ============================================================
# 18. RESTORE OPTIMIZER STATE
# ============================================================

optimizer_state = torch.load(
    CHECKPOINT_DIR
    / "optimizer.pt",
    map_location="cpu",
    weights_only=False,
)

optimizer.load_state_dict(
    optimizer_state
)

del optimizer_state

gc.collect()


# ============================================================
# 19. RESTORE COSINE SCHEDULER
# ============================================================

scheduler = (
    get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TOTAL_STEPS,
    )
)

scheduler_state = torch.load(
    CHECKPOINT_DIR
    / "scheduler.pt",
    map_location="cpu",
    weights_only=False,
)

scheduler.load_state_dict(
    scheduler_state
)

del scheduler_state


# ============================================================
# 20. RESTORE RNG
# ============================================================

def restore_rng():

    rank_specific = (
        CHECKPOINT_DIR
        / f"rng_state_rank{RANK}.pth"
    )

    generic = (
        CHECKPOINT_DIR
        / "rng_state.pth"
    )

    path = (
        rank_specific
        if rank_specific.exists()
        else generic
    )

    if not path.exists():

        print(
            f"WARNING RANK {RANK}: "
            "RNG checkpoint not found."
        )

        return

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "python" in state:

        random.setstate(
            state["python"]
        )

    if "numpy" in state:

        np.random.set_state(
            state["numpy"]
        )

    if "cpu" in state:

        torch.random.set_rng_state(
            state["cpu"]
        )

    if "cuda" in state:

        cuda_state = state["cuda"]

        if isinstance(
            cuda_state,
            list,
        ):

            if RANK < len(
                cuda_state
            ):

                torch.cuda.set_rng_state(
                    cuda_state[RANK],
                    device=LOCAL_RANK,
                )

        else:

            torch.cuda.set_rng_state(
                cuda_state,
                device=LOCAL_RANK,
            )


restore_rng()


# ============================================================
# 21. VERIFY OPTIMIZER
# ============================================================

if RANK == 0:

    print("\n" + "-" * 80)
    print("OPTIMIZER")
    print("-" * 80)

    print(
        "\nClass:"
    )

    print(
        type(optimizer)
    )

    print(
        "\nModule:"
    )

    print(
        type(optimizer).__module__
    )

    print(
        "\nPaged:"
    )

    print(
        getattr(
            optimizer,
            "is_paged",
            None,
        )
    )

    print(
        "\noptim_bits:"
    )

    print(
        getattr(
            getattr(
                optimizer,
                "args",
                None,
            ),
            "optim_bits",
            None,
        )
    )

assert (
    type(optimizer).__module__
    == "bitsandbytes.optim.adamw"
)

assert (
    getattr(
        optimizer,
        "is_paged",
        False,
    )
    is True
)


# ============================================================
# 22. DATASET / DISTRIBUTED SAMPLER
# ============================================================

train_records = (
    train_df.to_dict(
        orient="records"
    )
)

val_records = (
    val_df.to_dict(
        orient="records"
    )
)

train_sampler = (
    DistributedSampler(
        train_records,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=True,
        seed=SEED,
        drop_last=False,
    )
)

val_sampler = (
    DistributedSampler(
        val_records,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=False,
        drop_last=False,
    )
)

train_loader = DataLoader(
    train_records,
    batch_size=1,
    sampler=train_sampler,
    collate_fn=data_collator,
    num_workers=0,
    pin_memory=False,
)

val_loader = DataLoader(
    val_records,
    batch_size=1,
    sampler=val_sampler,
    collate_fn=data_collator,
    num_workers=0,
    pin_memory=False,
)


# ============================================================
# 23. DEFINITIVE GEOMETRY CHECK
# ============================================================

local_batches = len(
    train_loader
)

if local_batches != 7010:

    raise RuntimeError(
        f"Expected 7010 train batches "
        f"per rank; got {local_batches}"
    )

steps_per_epoch = math.ceil(
    local_batches
    / GRADIENT_ACCUMULATION_STEPS
)

if steps_per_epoch != 877:

    raise RuntimeError(
        f"Expected 877 steps/epoch; "
        f"got {steps_per_epoch}"
    )

if RANK == 0:

    print("\n" + "=" * 80)
    print("MR-FT-001 — DEFINITIVE GEOMETRY")
    print("=" * 80)

    print("\nWorld size:")
    print(WORLD_SIZE)

    print("\nPer-device train batch:")
    print(
        PER_DEVICE_TRAIN_BATCH_SIZE
    )

    print("\nPer-device eval batch:")
    print(
        PER_DEVICE_EVAL_BATCH_SIZE
    )

    print("\nGradient accumulation:")
    print(
        GRADIENT_ACCUMULATION_STEPS
    )

    print("\nGlobal effective batch:")
    print(
        WORLD_SIZE
        * PER_DEVICE_TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
    )

    print("\nTrain batches / rank:")
    print(local_batches)

    print("\nSteps / epoch:")
    print(steps_per_epoch)

    print("\nTotal steps:")
    print(TOTAL_STEPS)

    print("\nResume checkpoint:")
    print(CHECKPOINT_DIR)

    print("\nResume step:")
    print(START_STEP)


# ============================================================
# 24. FIND EXACT RESUME POSITION
# ============================================================

# At step 750:
#
# 750 * 8 = 6000 batches consumed per rank.
#
# 6000 / 7010 =
# 0.8559201141226819
#
# exactly matching checkpoint epoch.

resume_epoch = int(
    math.floor(
        START_EPOCH
    )
)

resume_fraction = (
    START_EPOCH
    - resume_epoch
)

resume_batch = round(
    resume_fraction
    * local_batches
)

if START_STEP == 750:

    if resume_epoch != 0:

        raise RuntimeError(
            "Expected checkpoint-750 "
            "to be in epoch 0."
        )

    if resume_batch != 6000:

        raise RuntimeError(
            f"Expected 6000 consumed "
            f"batches; got {resume_batch}"
        )


# ============================================================
# 25. RESUME SHUFFLED DATA EXACTLY
# ============================================================

train_sampler.set_epoch(
    resume_epoch
)

train_iterator = iter(
    train_loader
)

print_rank0 = (
    RANK == 0
)

if print_rank0:

    print("\n" + "-" * 80)
    print("FAST-FORWARDING TO RESUME POSITION")
    print("-" * 80)

    print(
        "\nConsumed batches:",
        resume_batch
    )

# Skip batches already processed at checkpoint.
for _ in range(
    resume_batch
):

    next(
        train_iterator
    )

if print_rank0:

    print(
        "\nFast-forward complete."
    )


# ============================================================
# 26. DDP BACKWARD SMOKE TEST
# ============================================================
#
# We test two microbatches:
#   first = no_sync()
#   second = synchronized backward
#
# NO optimizer update is performed.
# ============================================================

if print_rank0:

    print("\n" + "-" * 80)
    print("DDP BACKWARD SMOKE TEST")
    print("-" * 80)

model.train()

model.zero_grad(
    set_to_none=True
)

smoke_batch_1 = next(
    train_iterator
)

for key, value in (
    smoke_batch_1.items()
):

    if torch.is_tensor(value):

        smoke_batch_1[key] = (
            value.to(DEVICE)
        )

with model.no_sync():

    smoke_output_1 = model(
        **smoke_batch_1
    )

    smoke_loss_1 = (
        smoke_output_1.loss
        / 2.0
    )

    smoke_loss_1.backward()

del smoke_output_1
del smoke_loss_1
del smoke_batch_1

smoke_batch_2 = next(
    train_iterator
)

for key, value in (
    smoke_batch_2.items()
):

    if torch.is_tensor(value):

        smoke_batch_2[key] = (
            value.to(DEVICE)
        )

smoke_output_2 = model(
    **smoke_batch_2
)

smoke_loss_2 = (
    smoke_output_2.loss
    / 2.0
)

if not torch.isfinite(
    smoke_loss_2
):

    raise RuntimeError(
        "Non-finite DDP smoke loss."
    )

smoke_loss_2.backward()

print(
    f"[rank {RANK}] "
    "DDP synchronized backward PASS"
)

model.zero_grad(
    set_to_none=True
)

del smoke_output_2
del smoke_loss_2
del smoke_batch_2

gc.collect()

torch.cuda.empty_cache()

dist.barrier()

if print_rank0:

    print(
        "\nDDP BACKWARD SMOKE TEST: PASS"
    )


# IMPORTANT:
# The train iterator has advanced by 2 batches during smoke test.
# Recreate it and fast-forward again so training starts at the
# EXACT checkpoint position.

train_sampler.set_epoch(
    resume_epoch
)

train_iterator = iter(
    train_loader
)

for _ in range(
    resume_batch
):

    next(
        train_iterator
    )


# ============================================================
# 27. VALIDATION FUNCTION
# ============================================================

def evaluate():

    model.eval()

    local_loss_sum = 0.0

    local_token_count = 0

    for batch in val_loader:

        for key, value in batch.items():

            if torch.is_tensor(value):

                batch[key] = (
                    value.to(DEVICE)
                )

        with torch.no_grad():

            output = model(
                **batch
            )

        loss = output.loss

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite validation loss."
            )

        active_tokens = (
            batch["labels"]
            != -100
        ).sum().item()

        local_loss_sum += (
            float(loss)
            * active_tokens
        )

        local_token_count += (
            active_tokens
        )

        del output
        del loss
        del batch

    loss_tensor = torch.tensor(
        local_loss_sum,
        dtype=torch.float64,
        device=DEVICE,
    )

    token_tensor = torch.tensor(
        local_token_count,
        dtype=torch.int64,
        device=DEVICE,
    )

    dist.all_reduce(
        loss_tensor,
        op=dist.ReduceOp.SUM,
    )

    dist.all_reduce(
        token_tensor,
        op=dist.ReduceOp.SUM,
    )

    model.train()

    if token_tensor.item() <= 0:

        raise RuntimeError(
            "Validation produced zero "
            "active tokens."
        )

    return (
        loss_tensor.item()
        / token_tensor.item()
    )


# ============================================================
# 28. CHECKPOINT SAVER
# ============================================================

def save_checkpoint(
    step,
    epoch,
    batch_position,
    eval_loss,
):

    checkpoint_path = (
        OUTPUT_DIR
        / f"checkpoint-{step}"
    )

    if RANK == 0:

        checkpoint_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save adapter only.
        model.module.save_pretrained(
            checkpoint_path,
            safe_serialization=True,
        )

        processor.save_pretrained(
            checkpoint_path
        )

    # Make sure rank 0 finished writing adapter
    # before any rank proceeds.
    dist.barrier()

    # Every rank saves its own optimizer state.
    #
    # The optimizer states should be identical because DDP
    # averages gradients and every rank performs the same update.
    torch.save(
        optimizer.state_dict(),
        checkpoint_path
        / f"optimizer_rank{RANK}.pt",
    )

    # Rank 0 also creates a generic optimizer.pt for
    # compatibility/convenience.
    if RANK == 0:

        shutil_copy_source = (
            checkpoint_path
            / "optimizer_rank0.pt"
        )

        shutil_copy_dest = (
            checkpoint_path
            / "optimizer.pt"
        )

        import shutil

        shutil.copy2(
            shutil_copy_source,
            shutil_copy_dest,
        )

        torch.save(
            scheduler.state_dict(),
            checkpoint_path
            / "scheduler.pt",
        )

        # rank 0 generic RNG file
        rank0_rng = {
            "python":
                random.getstate(),

            "numpy":
                np.random.get_state(),

            "cpu":
                torch.random.get_rng_state(),

            "cuda":
                [
                    torch.cuda.get_rng_state(
                        0
                    )
                ],
        }

        torch.save(
            rank0_rng,
            checkpoint_path
            / "rng_state.pth",
        )

    # Every rank saves rank-specific RNG.
    rank_rng = {

        "python":
            random.getstate(),

        "numpy":
            np.random.get_state(),

        "cpu":
            torch.random.get_rng_state(),

        "cuda":
            torch.cuda.get_rng_state(
                LOCAL_RANK
            ),
    }

    torch.save(
        rank_rng,
        checkpoint_path
        / f"rng_state_rank{RANK}.pth",
    )

    # Ensure every rank completed checkpoint writes.
    dist.barrier()

    if RANK == 0:

        state = {

            "global_step":
                int(step),

            "epoch":
                (
                    float(epoch)
                    + (
                        float(batch_position)
                        / float(
                            local_batches
                        )
                    )
                ),

            "max_steps":
                TOTAL_STEPS,

            "resume_epoch":
                int(epoch),

            "resume_batch":
                int(batch_position),

            "best_metric":
                (
                    float(
                        eval_loss
                    )
                    if eval_loss
                    is not None
                    else BEST_EVAL_LOSS
                ),

            "best_model_checkpoint":
                BEST_CHECKPOINT,

            "log_history":
                LOG_HISTORY,
        }

        with open(
            checkpoint_path
            / "trainer_state.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                state,
                f,
                ensure_ascii=False,
                indent=2,
            )

    dist.barrier()

    return checkpoint_path


# ============================================================
# 29. INITIAL TRAINING STATE
# ============================================================

global_step = START_STEP

current_epoch = resume_epoch

batch_position = resume_batch

running_start = (
    __import__("time")
    .time()
)


# ============================================================
# 30. MASTER PRE-TRAINING REPORT
# ============================================================

if RANK == 0:

    print("\n" + "=" * 80)

    print(
        "✅ MR-FT-001 — FULL RESUME PREFLIGHT: PASS"
    )

    print("=" * 80)

    print("\nResume:")
    print(
        f"{global_step} → {TOTAL_STEPS}"
    )

    print("\nPer-device batch:")
    print(
        PER_DEVICE_TRAIN_BATCH_SIZE
    )

    print("\nGradient accumulation:")
    print(
        GRADIENT_ACCUMULATION_STEPS
    )

    print("\nGlobal effective batch:")
    print(
        WORLD_SIZE
        * PER_DEVICE_TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
    )

    print("\nSteps / epoch:")
    print(
        steps_per_epoch
    )

    print("\nTrainable parameters:")
    print(
        f"{trainable_count:,}"
    )

    print("\nOptimizer:")
    print(
        "bitsandbytes AdamW "
        "8-bit paged"
    )

    print("\nAMP:")
    print(
        "disabled"
    )

    print("\nCheckpoint:")
    print(
        CHECKPOINT_DIR
    )

    print("\n🔥 STARTING ACTUAL RESUME")
    print(
        f"from optimizer step {global_step}"
    )


# ============================================================
# 31. TRAINING LOOP
# ============================================================

while (
    global_step
    < TOTAL_STEPS
):

    accumulation = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    loss_values = []

    # --------------------------------------------------------
    # One optimizer update
    # --------------------------------------------------------

    while (
        accumulation
        < GRADIENT_ACCUMULATION_STEPS
        and
        batch_position
        < local_batches
    ):

        try:

            batch = next(
                train_iterator
            )

        except StopIteration:

            raise RuntimeError(
                "Unexpected train iterator "
                "exhaustion."
            )

        for key, value in (
            batch.items()
        ):

            if torch.is_tensor(value):

                batch[key] = (
                    value.to(DEVICE)
                )

        is_last_microbatch = (
            accumulation
            == (
                GRADIENT_ACCUMULATION_STEPS
                - 1
            )
        )

        # Avoid gradient all-reduce on the
        # first 7 microbatches.
        if is_last_microbatch:

            output = model(
                **batch
            )

            loss = output.loss

            scaled_loss = (
                loss
                / GRADIENT_ACCUMULATION_STEPS
            )

            scaled_loss.backward()

        else:

            with model.no_sync():

                output = model(
                    **batch
                )

                loss = output.loss

                scaled_loss = (
                    loss
                    / GRADIENT_ACCUMULATION_STEPS
                )

                scaled_loss.backward()

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Non-finite training loss "
                f"at step "
                f"{global_step + 1}"
            )

        loss_values.append(
            float(
                loss.detach()
            )
        )

        del output
        del loss
        del scaled_loss
        del batch

        batch_position += 1
        accumulation += 1

    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    grad_norm = (
        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            MAX_GRAD_NORM,
        )
    )

    if not torch.isfinite(
        grad_norm
    ):

        raise RuntimeError(
            "Non-finite gradient norm."
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer.step()

    scheduler.step()

    optimizer.zero_grad(
        set_to_none=True
    )

    global_step += 1

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    if (
        global_step % 10
        == 0
    ):

        mean_loss = (
            sum(loss_values)
            / max(
                len(loss_values),
                1,
            )
        )

        elapsed = (
            __import__("time")
            .time()
            - running_start
        )

        completed = (
            global_step
            - START_STEP
        )

        remaining = (
            TOTAL_STEPS
            - global_step
        )

        if completed > 0:

            sec_per_step = (
                elapsed
                / completed
            )

            eta = (
                sec_per_step
                * remaining
            )

        else:

            eta = 0

        if RANK == 0:

            print(
                f"\nStep "
                f"{global_step}/"
                f"{TOTAL_STEPS}"
            )

            print(
                "Training loss:",
                f"{mean_loss:.6f}"
            )

            print(
                "Learning rate:",
                f"{optimizer.param_groups[0]['lr']:.10e}"
            )

            print(
                "Gradient norm:",
                f"{float(grad_norm):.6f}"
            )

            print(
                "ETA hours:",
                f"{eta / 3600:.2f}"
            )

        LOG_HISTORY.append(
            {
                "step":
                    int(global_step),

                "loss":
                    mean_loss,

                "learning_rate":
                    optimizer.param_groups[0]["lr"],

                "grad_norm":
                    float(
                        grad_norm
                    ),
            }
        )

    # --------------------------------------------------------
    # VALIDATION / CHECKPOINT
    # --------------------------------------------------------

    if (
        global_step % 250
        == 0
    ):

        dist.barrier()

        if RANK == 0:

            print(
                "\n" + "=" * 70
            )

            print(
                f"VALIDATION @ STEP "
                f"{global_step}"
            )

            print(
                "=" * 70
            )

        eval_loss = evaluate()

        if RANK == 0:

            print(
                "\nValidation loss:"
            )

            print(
                f"{eval_loss:.10f}"
            )

        LOG_HISTORY.append(
            {
                "step":
                    int(global_step),

                "eval_loss":
                    float(eval_loss),
            }
        )

        # ----------------------------------------------------
        # Best adapter
        # ----------------------------------------------------

        if (
            BEST_EVAL_LOSS
            is None
            or
            eval_loss
            < BEST_EVAL_LOSS
        ):

            BEST_EVAL_LOSS = (
                float(eval_loss)
            )

            BEST_CHECKPOINT = (
                str(
                    OUTPUT_DIR
                    / f"checkpoint-{global_step}"
                )
            )

            if RANK == 0:

                best_dir = (
                    OUTPUT_DIR
                    / "best_adapter"
                )

                if best_dir.exists():

                    import shutil

                    shutil.rmtree(
                        best_dir
                    )

                best_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                model.module.save_pretrained(
                    best_dir,
                    safe_serialization=True,
                )

                processor.save_pretrained(
                    best_dir
                )

                print(
                    "\nNEW BEST MODEL"
                )

                print(
                    "Best validation loss:",
                    BEST_EVAL_LOSS
                )

        dist.barrier()

        # ----------------------------------------------------
        # Save recovery checkpoint
        # ----------------------------------------------------

        saved_path = (
            save_checkpoint(
                global_step,
                current_epoch,
                batch_position,
                eval_loss,
            )
        )

        if RANK == 0:

            print(
                "\nCheckpoint saved:"
            )

            print(
                saved_path
            )

        gc.collect()

        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Epoch boundary
    # --------------------------------------------------------

    if (
        batch_position
        >= local_batches
    ):

        current_epoch += 1

        batch_position = 0

        if (
            current_epoch
            < NUM_EPOCHS
            and global_step
            < TOTAL_STEPS
        ):

            train_sampler.set_epoch(
                current_epoch
            )

            train_iterator = iter(
                train_loader
            )

        if RANK == 0:

            print(
                f"\nEpoch "
                f"{current_epoch} "
                "completed."
            )


# ============================================================
# 32. FINAL ADAPTER
# ============================================================

dist.barrier()

if RANK == 0:

    final_dir = (
        OUTPUT_DIR
        / "final_adapter"
    )

    if final_dir.exists():

        import shutil

        shutil.rmtree(
            final_dir
        )

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.module.save_pretrained(
        final_dir,
        safe_serialization=True,
    )

    processor.save_pretrained(
        final_dir
    )

    final_manifest = {

        "experiment_id":
            "MR-FT-001",

        "model_id":
            MODEL_ID,

        "model_revision":
            MODEL_REVISION,

        "resume_checkpoint":
            str(CHECKPOINT_DIR),

        "starting_step":
            START_STEP,

        "final_step":
            global_step,

        "world_size":
            WORLD_SIZE,

        "per_device_train_batch_size":
            PER_DEVICE_TRAIN_BATCH_SIZE,

        "per_device_eval_batch_size":
            PER_DEVICE_EVAL_BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "effective_global_batch_size":
            16,

        "train_batches_per_rank":
            7010,

        "steps_per_epoch":
            877,

        "total_steps":
            TOTAL_STEPS,

        "epochs":
            NUM_EPOCHS,

        "max_seq_length":
            MAX_SEQ_LENGTH,

        "learning_rate":
            LEARNING_RATE,

        "warmup_steps":
            WARMUP_STEPS,

        "optimizer":
            "bitsandbytes AdamW 8-bit paged",

        "fp16_amp":
            False,

        "bf16_amp":
            False,

        "trainable_parameters":
            24_158_208,

        "train_size":
            14020,

        "validation_size":
            796,

        "frozen_test_size":
            794,

        "best_eval_loss":
            BEST_EVAL_LOSS,

        "best_model_checkpoint":
            BEST_CHECKPOINT,

        "final_adapter":
            str(final_dir),

        "resume_method":
            "DDP init_sync=False",
    }

    manifest_path = (
        OUTPUT_DIR
        / "MR-FT-001_final_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 80)

    print(
        "🎉 MR-FT-001 TRAINING COMPLETE"
    )

    print("=" * 80)

    print(
        "\nFinal global step:",
        global_step
    )

    print(
        "\nBest validation loss:",
        BEST_EVAL_LOSS
    )

    print(
        "\nBest checkpoint:",
        BEST_CHECKPOINT
    )

    print(
        "\nFinal adapter:",
        final_dir
    )

    print(
        "\nManifest:",
        manifest_path
    )


# ============================================================
# 33. CLEAN DISTRIBUTED PROCESS
# ============================================================

dist.barrier()

dist.destroy_process_group()

