
# ============================================================
# MR-FT-001 — DISTRIBUTED TRAINING WORKER
# ============================================================

import os
import gc
import json
import math
from pathlib import Path

import torch
import pandas as pd

from kaggle_secrets import UserSecretsClient

from datasets import Dataset

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel

from trl import (
    SFTConfig,
    SFTTrainer,
)


# ============================================================
# 1. DISTRIBUTED WORKER INFORMATION
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
        f"MR-FT-001 requires exactly 2 workers, "
        f"but WORLD_SIZE={WORLD_SIZE}"
    )

if LOCAL_RANK not in (0, 1):
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
# 2. EXACT EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENT_ID = "MR-FT-001"

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

CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoint-750"
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

PER_DEVICE_BATCH_SIZE = 1

PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 1e-4

WARMUP_STEPS = 263

TOTAL_STEPS = 2631

NUM_EPOCHS = 3

MAX_GRAD_NORM = 1.0

SEED = 42


# ============================================================
# 3. WORKER HEADER
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

print("\nPyTorch:")
print(torch.__version__)

print("\nCUDA:")
print(torch.version.cuda)


# ============================================================
# 4. VERIFY CHECKPOINT
# ============================================================

required_checkpoint_files = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
]

if not CHECKPOINT_DIR.exists():

    raise RuntimeError(
        "checkpoint-750 does not exist."
    )

for filename in required_checkpoint_files:

    path = (
        CHECKPOINT_DIR
        / filename
    )

    if not path.exists():

        raise RuntimeError(
            f"Missing checkpoint file: "
            f"{filename}"
        )


# ============================================================
# 5. VERIFY SAVED TRAINER STATE
# ============================================================

with open(
    CHECKPOINT_DIR
    / "trainer_state.json",
    "r",
    encoding="utf-8",
) as f:

    saved_state = json.load(f)

saved_global_step = int(
    saved_state["global_step"]
)

saved_max_steps = int(
    saved_state["max_steps"]
)

saved_epoch = float(
    saved_state["epoch"]
)

saved_best_metric = (
    saved_state.get(
        "best_metric"
    )
)

saved_best_checkpoint = (
    saved_state.get(
        "best_model_checkpoint"
    )
)

if saved_global_step != 750:

    raise RuntimeError(
        f"Expected global_step=750, "
        f"found {saved_global_step}"
    )

if saved_max_steps != 2631:

    raise RuntimeError(
        f"Expected max_steps=2631, "
        f"found {saved_max_steps}"
    )

if RANK == 0:

    print("\n" + "-" * 80)
    print("CHECKPOINT STATE")
    print("-" * 80)

    print("\nSaved global step:")
    print(saved_global_step)

    print("\nSaved epoch:")
    print(saved_epoch)

    print("\nSaved max steps:")
    print(saved_max_steps)

    print("\nBest eval loss:")
    print(saved_best_metric)

    print("\nBest checkpoint:")
    print(saved_best_checkpoint)


# ============================================================
# 6. LOAD EXACT DATA
# ============================================================

if not TRAIN_FILE.exists():

    raise FileNotFoundError(
        str(TRAIN_FILE)
    )

if not VAL_FILE.exists():

    raise FileNotFoundError(
        str(VAL_FILE)
    )

train_df = pd.read_csv(
    TRAIN_FILE
)

val_df = pd.read_csv(
    VAL_FILE
)

if len(train_df) != 14020:

    raise RuntimeError(
        f"Expected 14020 training rows, "
        f"found {len(train_df)}"
    )

if len(val_df) != 796:

    raise RuntimeError(
        f"Expected 796 validation rows, "
        f"found {len(val_df)}"
    )

if RANK == 0:

    print("\n" + "-" * 80)
    print("DATA")
    print("-" * 80)

    print("\nTrain rows:")
    print(len(train_df))

    print("\nValidation rows:")
    print(len(val_df))


# ============================================================
# 7. HUGGING FACE TOKEN
# ============================================================

secrets = (
    UserSecretsClient()
)

hf_token = secrets.get_secret(
    "HF_TOKEN"
)

if not hf_token:

    raise RuntimeError(
        "HF_TOKEN not available."
    )


# ============================================================
# 8. LOAD EXACT PROCESSOR
# ============================================================

processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=hf_token,
    )
)

tokenizer = (
    processor.tokenizer
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

tokenizer.padding_side = "right"

# Explicitly align token IDs.
# This removes the harmless tokenizer/model mismatch warning.

if tokenizer.pad_token_id is not None:

    processor.tokenizer.pad_token_id = (
        tokenizer.pad_token_id
    )


# ============================================================
# 9. EXACT COMPLETION-ONLY COLLATOR
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

        all_input_ids = []
        all_labels = []

        for example in examples:

            prompt = str(
                example["prompt"]
            )

            completion = str(
                example["completion"]
            )

            full_messages = (
                self.build_messages(
                    prompt,
                    completion,
                )
            )

            prompt_messages = (
                self.build_messages(
                    prompt,
                    None,
                )
            )

            full = (
                self.processor
                .apply_chat_template(
                    full_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    return_dict=True,
                )
            )

            prompt_only = (
                self.processor
                .apply_chat_template(
                    prompt_messages,
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
                prompt_only["input_ids"][0]
            )

            prompt_length = (
                prompt_ids.shape[0]
            )

            full_length = (
                full_ids.shape[0]
            )

            if not torch.equal(
                full_ids[
                    :prompt_length
                ],
                prompt_ids,
            ):

                raise RuntimeError(
                    "Prompt prefix mismatch."
                )

            if full_length > (
                self.max_length
            ):

                raise RuntimeError(
                    f"Full sequence length "
                    f"{full_length} exceeds "
                    f"MAX_SEQ_LENGTH="
                    f"{self.max_length}"
                )

            input_ids = (
                full_ids.tolist()
            )

            labels = (
                input_ids.copy()
            )

            # Mask prompt/source
            for i in range(
                prompt_length
            ):

                labels[i] = -100

            if not any(
                token != -100
                for token in labels
            ):

                raise RuntimeError(
                    "No active completion tokens."
                )

            all_input_ids.append(
                {
                    "input_ids":
                        input_ids
                }
            )

            all_labels.append(
                {
                    "input_ids":
                        labels
                }
            )

        batch = (
            self.tokenizer.pad(
                all_input_ids,
                padding=True,
                return_tensors="pt",
            )
        )

        label_batch = (
            self.tokenizer.pad(
                all_labels,
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
# 10. COLLATOR SANITY CHECK
# ============================================================

collator_test = data_collator(
    [
        train_df.iloc[0].to_dict(),
        train_df.iloc[1].to_dict(),
    ]
)

active_tokens = (
    collator_test["labels"]
    != -100
).sum().item()

if active_tokens <= 0:

    raise RuntimeError(
        "Collator produced zero active loss tokens."
    )

if (
    collator_test[
        "input_ids"
    ].shape[1]
    > MAX_SEQ_LENGTH
):

    raise RuntimeError(
        "Collator sequence exceeds 768."
    )

if RANK == 0:

    print("\n" + "-" * 80)
    print("COLLATOR")
    print("-" * 80)

    print("\nTest shape:")
    print(
        tuple(
            collator_test[
                "input_ids"
            ].shape
        )
    )

    print("\nActive loss tokens:")
    print(active_tokens)

    print("\nCollator: PASS")


del collator_test

gc.collect()


# ============================================================
# 11. LOAD EXACT 4-BIT BASE GEMMA
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

model = (
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

# ============================================================
# 12. LOAD CHECKPOINT-750 ADAPTER
# ============================================================

model = (
    PeftModel
    .from_pretrained(
        model,
        str(CHECKPOINT_DIR),
        is_trainable=True,
    )
)

# ============================================================
# 13. MODEL CONFIG ALIGNMENT
# ============================================================

if hasattr(
    model.config,
    "use_cache",
):

    model.config.use_cache = False

# Keep tokenizer special-token IDs aligned.
if tokenizer.pad_token_id is not None:

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

if tokenizer.eos_token_id is not None:

    model.config.eos_token_id = (
        tokenizer.eos_token_id
    )

if tokenizer.bos_token_id is not None:

    model.config.bos_token_id = (
        tokenizer.bos_token_id
    )

if hasattr(
    model,
    "generation_config",
):

    if tokenizer.pad_token_id is not None:

        model.generation_config.pad_token_id = (
            tokenizer.pad_token_id
        )

    if tokenizer.eos_token_id is not None:

        model.generation_config.eos_token_id = (
            tokenizer.eos_token_id
        )

    if tokenizer.bos_token_id is not None:

        model.generation_config.bos_token_id = (
            tokenizer.bos_token_id
        )


# ============================================================
# 14. TRAINABLE PARAMETER GATE
# ============================================================

model.train()

if hasattr(
    model,
    "enable_input_require_grads",
):

    model.enable_input_require_grads()

trainable_count = 0
trainable_tensors = 0

bad_trainable = []

for name, param in (
    model.named_parameters()
):

    if param.requires_grad:

        trainable_count += (
            param.numel()
        )

        trainable_tensors += 1

        if (
            "vision_tower" in name
            or "audio_tower" in name
        ):

            bad_trainable.append(
                name
            )

if trainable_count != 24_158_208:

    raise RuntimeError(
        f"Unexpected trainable parameter count: "
        f"{trainable_count}"
    )

if trainable_tensors != 410:

    raise RuntimeError(
        f"Expected 410 trainable tensors, "
        f"found {trainable_tensors}"
    )

if bad_trainable:

    raise RuntimeError(
        "Vision/audio parameters "
        "are trainable."
    )


# ============================================================
# 15. MODEL DEVICE GATE
# ============================================================

first_device = (
    next(
        model.parameters()
    ).device
)

if str(first_device) != (
    f"cuda:{LOCAL_RANK}"
):

    raise RuntimeError(
        f"Model is on {first_device}; "
        f"expected cuda:{LOCAL_RANK}"
    )


# ============================================================
# 16. CREATE SFT TRAINER
# ============================================================

sft_args = SFTConfig(

    output_dir=str(
        OUTPUT_DIR
    ),

    max_length=MAX_SEQ_LENGTH,

    packing=False,

    num_train_epochs=NUM_EPOCHS,

    max_steps=TOTAL_STEPS,

    per_device_train_batch_size=(
        PER_DEVICE_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        PER_DEVICE_EVAL_BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    learning_rate=LEARNING_RATE,

    lr_scheduler_type="cosine",

    warmup_steps=WARMUP_STEPS,

    weight_decay=0.01,

    max_grad_norm=MAX_GRAD_NORM,

    # IMPORTANT:
    # No AMP because the T4 previously produced
    # the BFloat16 GradScaler failure.
    fp16=False,

    bf16=False,

    gradient_checkpointing=True,

    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },

    optim="paged_adamw_8bit",

    logging_steps=10,

    report_to="none",

    eval_strategy="steps",

    eval_steps=250,

    save_strategy="steps",

    save_steps=250,

    save_total_limit=3,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    dataset_kwargs={
        "skip_prepare_dataset": True
    },

    remove_unused_columns=False,

    seed=SEED,

    push_to_hub=False,

    run_name="MR-FT-001",

    ddp_find_unused_parameters=False,

    dataloader_num_workers=0,
)


# ============================================================
# 17. CREATE TRAIN DATASET
# ============================================================

train_dataset = (
    Dataset.from_pandas(
        train_df[
            [
                "pair_id",
                "prompt",
                "completion",
            ]
        ],
        preserve_index=False,
    )
)

val_dataset = (
    Dataset.from_pandas(
        val_df[
            [
                "pair_id",
                "prompt",
                "completion",
            ]
        ],
        preserve_index=False,
    )
)


# ============================================================
# 18. SFT TRAINER
# ============================================================

trainer = (
    SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=data_collator,
        peft_config=None,
    )
)


# ============================================================
# 19. DISTRIBUTED TRAINING GEOMETRY GATE
# ============================================================

train_loader = (
    trainer.get_train_dataloader()
)

loader_length = len(
    train_loader
)

steps_per_epoch = math.ceil(
    loader_length
    / GRADIENT_ACCUMULATION_STEPS
)

if (
    trainer.args.world_size
    != 2
):

    raise RuntimeError(
        f"Trainer world_size="
        f"{trainer.args.world_size}; "
        f"expected 2"
    )

if loader_length != 7010:

    raise RuntimeError(
        f"Expected 7010 batches/process, "
        f"got {loader_length}"
    )

if steps_per_epoch != 877:

    raise RuntimeError(
        f"Expected 877 steps/epoch, "
        f"got {steps_per_epoch}"
    )

if trainer.args.max_steps != 2631:

    raise RuntimeError(
        f"Expected max_steps=2631, "
        f"got {trainer.args.max_steps}"
    )


# ============================================================
# 20. OPTIMIZER CONFIGURATION GATE
# ============================================================

optimizer_cls, optimizer_kwargs = (
    trainer.get_optimizer_cls_and_kwargs(
        trainer.args,
        trainer.model,
    )
)

if (
    optimizer_kwargs.get(
        "optim_bits"
    )
    != 8
):

    raise RuntimeError(
        "Optimizer is not configured for 8-bit."
    )

if (
    optimizer_kwargs.get(
        "is_paged"
    )
    is not True
):

    raise RuntimeError(
        "Optimizer is not configured as paged."
    )


# ============================================================
# 21. PRECISION GATE
# ============================================================

if trainer.args.fp16:

    raise RuntimeError(
        "AMP fp16 must be False."
    )

if trainer.args.bf16:

    raise RuntimeError(
        "AMP bf16 must be False."
    )


# ============================================================
# 22. MASTER PREFLIGHT REPORT
# ============================================================

if RANK == 0:

    print("\n" + "=" * 80)
    print(
        "MR-FT-001 — 2-GPU RESUME PREFLIGHT: PASS"
    )
    print("=" * 80)

    print("\nWorld size:")
    print(WORLD_SIZE)

    print("\nWorker GPUs:")
    print("GPU 0 + GPU 1")

    print("\nPer-device batch:")
    print(PER_DEVICE_BATCH_SIZE)

    print("\nGradient accumulation:")
    print(GRADIENT_ACCUMULATION_STEPS)

    print("\nGlobal effective batch:")
    print(
        PER_DEVICE_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
        * WORLD_SIZE
    )

    print("\nDataloader batches/process:")
    print(loader_length)

    print("\nSteps/epoch:")
    print(steps_per_epoch)

    print("\nTotal optimizer steps:")
    print(trainer.args.max_steps)

    print("\nCheckpoint:")
    print(CHECKPOINT_DIR)

    print("\nCheckpoint step:")
    print(saved_global_step)

    print("\nTrainable parameters:")
    print(
        f"{trainable_count:,}"
    )

    print("\nOptimizer:")
    print(
        optimizer_cls
    )

    print("\nOptimizer module:")
    print(
        optimizer_cls.__module__
    )

    print("\nOptimizer kwargs:")
    for key, value in (
        optimizer_kwargs.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print("\nFP16 AMP:")
    print(False)

    print("\nBF16 AMP:")
    print(False)

    print("\n" + "=" * 80)
    print(
        "🔥 STARTING RESUME FROM CHECKPOINT-750"
    )
    print("=" * 80)


# ============================================================
# 23. ACTUAL RESUME
# ============================================================

resume_result = trainer.train(
    resume_from_checkpoint=str(
        CHECKPOINT_DIR
    )
)


# ============================================================
# 24. SAVE FINAL BEST ADAPTER
# ============================================================

if RANK == 0:

    FINAL_ADAPTER_DIR = (
        OUTPUT_DIR
        / "final_adapter"
    )

    if FINAL_ADAPTER_DIR.exists():

        import shutil

        shutil.rmtree(
            FINAL_ADAPTER_DIR
        )

    FINAL_ADAPTER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(
            FINAL_ADAPTER_DIR
        )
    )

    processor.save_pretrained(
        FINAL_ADAPTER_DIR
    )

    final_manifest = {

        "experiment_id":
            EXPERIMENT_ID,

        "model_id":
            MODEL_ID,

        "model_revision":
            MODEL_REVISION,

        "resume_checkpoint":
            str(
                CHECKPOINT_DIR
            ),

        "start_step":
            saved_global_step,

        "final_step":
            trainer.state.global_step,

        "world_size":
            WORLD_SIZE,

        "per_device_batch_size":
            PER_DEVICE_BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "effective_global_batch_size":
            (
                PER_DEVICE_BATCH_SIZE
                * GRADIENT_ACCUMULATION_STEPS
                * WORLD_SIZE
            ),

        "steps_per_epoch":
            steps_per_epoch,

        "total_steps":
            TOTAL_STEPS,

        "max_seq_length":
            MAX_SEQ_LENGTH,

        "epochs":
            NUM_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "warmup_steps":
            WARMUP_STEPS,

        "optimizer":
            "bitsandbytes AdamW "
            "8-bit paged",

        "fp16_amp":
            False,

        "bf16_amp":
            False,

        "best_eval_loss":
            trainer.state.best_metric,

        "best_model_checkpoint":
            trainer.state.best_model_checkpoint,

        "final_adapter":
            str(
                FINAL_ADAPTER_DIR
            ),

        "train_size":
            len(train_df),

        "validation_size":
            len(val_df),
    }

    MANIFEST_PATH = (
        OUTPUT_DIR
        / "MR-FT-001_final_manifest.json"
    )

    with open(
        MANIFEST_PATH,
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

    print("\nFinal global step:")
    print(
        trainer.state.global_step
    )

    print("\nBest validation loss:")
    print(
        trainer.state.best_metric
    )

    print("\nBest checkpoint:")
    print(
        trainer.state.best_model_checkpoint
    )

    print("\nFinal adapter:")
    print(
        FINAL_ADAPTER_DIR
    )

    print("\nManifest:")
    print(
        MANIFEST_PATH
    )
