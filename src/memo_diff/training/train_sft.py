"""Full-weight causal LM fine-tune on clinical canaries (no Pile)."""

from __future__ import annotations

import atexit
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from memo_diff.training.data import load_unique_canaries, pack_canary_epoch, pad_packed_sequence
from memo_diff.training.utils import cosine_lr, probe_canaries, slim_trainer_state


@dataclass
class SFTConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "outputs/sft"
    canaries_path: str = (
        "data/training/llama31_8b_balanced_2k_phi_benign_v2_luna_9f/"
        "train_canaries.jsonl"
    )
    seq_len: int = 1024
    pack: bool = True
    add_special_tokens: bool = True
    seed: int = 42
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    warmup_steps: int = 10
    max_epochs: int = 30
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8
    gradient_checkpointing: bool = False
    adam_foreach: Optional[bool] = None
    log_every: int = 5
    save_every_epoch: int = 1
    keep_full_last: int = 3
    keep_optimizer_last: int = 1
    sparse_keep_epochs: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 30])
    probe_every_epoch: int = 1
    probe_canaries: int = 16
    resume: bool = True
    init_from: Optional[str] = None
    wandb: bool = True
    wandb_entity: str = ""
    wandb_project: str = "finetuning"
    wandb_run_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "SFTConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class PackedEpochDataset(Dataset):
    def __init__(self, sequences: list[dict[str, Any]], *, seq_len: int, pad_id: int):
        self.sequences = sequences
        self.seq_len = seq_len
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.sequences[idx]
        packed = pad_packed_sequence(row["input_ids"], seq_len=self.seq_len, pad_id=self.pad_id)
        packed["canary_ids"] = list(row.get("canary_ids") or [])
        return packed


def _list_step_dirs(ckpt_dir: Path) -> list[tuple[int, Path]]:
    steps: list[tuple[int, Path]] = []
    if not ckpt_dir.exists():
        return steps
    for p in ckpt_dir.iterdir():
        if p.is_dir() and p.name.startswith("step_"):
            try:
                steps.append((int(p.name.split("_")[1]), p))
            except ValueError:
                continue
    steps.sort()
    return steps


def _read_epoch(path: Path) -> Optional[int]:
    meta = path / "epoch.json"
    if meta.exists():
        try:
            return int(json.loads(meta.read_text(encoding="utf-8")).get("epoch"))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    state_path = path / "trainer_state.pt"
    if not state_path.exists():
        return None
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(state, dict) and state.get("epoch") is not None:
        try:
            return int(state["epoch"])
        except (TypeError, ValueError):
            return None
    return None


def _has_checkpoint_weights(path: Path) -> bool:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        file = path / filename
        if file.is_file() and file.stat().st_size > 0:
            return True
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            index = json.loads((path / filename).read_text(encoding="utf-8"))
            shards = set(index["weight_map"].values())
            if shards and all(
                isinstance(shard, str) and Path(shard).name == shard
                and (path / shard).is_file() and (path / shard).stat().st_size > 0
                for shard in shards
            ):
                return True
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return False


def _latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    for _, path in reversed(_list_step_dirs(ckpt_dir)):
        if not _has_checkpoint_weights(path):
            continue
        state_path = path / "trainer_state.pt"
        if not state_path.exists() or state_path.stat().st_size < 50_000_000:
            continue
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if isinstance(state, dict) and "optimizer" in state:
            return path
    return None


def _save_checkpoint(
    *,
    path: Path,
    model,
    optimizer,
    step: int,
    epoch: int,
    exposure: dict[str, dict[str, Any]],
    cfg: SFTConfig,
    data_loader_rng_state: Optional[torch.Tensor] = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(path)
    tmp_state = path / "trainer_state.pt.tmp"
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "exposure": exposure,
            "config": asdict(cfg),
            "data_loader_rng_state": data_loader_rng_state,
        },
        tmp_state,
    )
    tmp_state.replace(path / "trainer_state.pt")
    (path / "epoch.json").write_text(
        json.dumps({"epoch": epoch, "step": step}, indent=2) + "\n", encoding="utf-8"
    )
    (path / "exposure.json").write_text(
        json.dumps(exposure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _prune_sft_checkpoints(ckpt_dir: Path, cfg: SFTConfig) -> None:
    """Keep last N full dirs, last optimizer, and a sparse model-only epoch set."""
    steps = _list_step_dirs(ckpt_dir)
    if not steps:
        return
    sparse = {int(x) for x in (cfg.sparse_keep_epochs or [])}
    keep_full = max(1, int(cfg.keep_full_last))
    keep_opt = max(1, int(cfg.keep_optimizer_last))
    newest = {path for _, path in steps[-keep_full:]}
    newest_opt = {path for _, path in steps[-keep_opt:]}
    for _, path in steps:
        epoch = _read_epoch(path)
        keep_weights = path in newest or (epoch is not None and epoch in sparse)
        if not keep_weights:
            print(f"[sft] pruning checkpoint {path}", flush=True)
            shutil.rmtree(path, ignore_errors=True)
            continue
        if path not in newest_opt:
            slim_trainer_state(path)


def _wandb_init(cfg: SFTConfig, *, rank: int, extra: dict[str, Any]) -> Any:
    if rank != 0 or not cfg.wandb:
        return None
    key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")
    if not key:
        raise RuntimeError(
            "wandb logging is enabled but WANDB_API_KEY is unset. "
            "Add it to .env or pass --no-wandb."
        )
    import wandb

    os.environ.setdefault("WANDB_API_KEY", key)
    entity = os.environ.get("WANDB_ENTITY") or cfg.wandb_entity
    project = os.environ.get("WANDB_PROJECT") or cfg.wandb_project
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    id_path = out / "wandb_run_id.txt"
    run_id = id_path.read_text(encoding="utf-8").strip() if id_path.exists() else ""
    run_name = cfg.wandb_run_name or out.name
    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        id=run_id or None,
        resume="allow" if run_id else None,
        config={**asdict(cfg), **extra},
        dir=str(out),
    )
    if run is not None and run.id:
        id_path.write_text(run.id + "\n", encoding="utf-8")
        print(f"[sft] wandb run={run.id} url={run.url}", flush=True)
    return run


def _wandb_log(run: Any, payload: dict[str, Any], *, step: int) -> None:
    if run is None:
        return
    clean = {k: v for k, v in payload.items() if isinstance(v, (int, float, str, bool))}
    run.log(clean, step=step)


def train(cfg: SFTConfig) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from torch._native.registry import deregister_op_overrides

        deregister_op_overrides(disable_dsl_names=["triton", "cutedsl"])
    except Exception as exc:  # noqa: BLE001
        print(f"[sft] warning: could not disable torch._native overrides: {exc}", flush=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1

    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = Path(cfg.output_dir)
    ckpt_dir = out / "checkpoints"
    log_path = out / "train_log.jsonl"
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pad_id = int(tokenizer.pad_token_id)

    canaries = load_unique_canaries(Path(cfg.canaries_path))
    if not canaries:
        raise FileNotFoundError(f"No canaries in {cfg.canaries_path}")
    canary_label = {str(r.get("id")): str(r.get("label") or "") for r in canaries}

    start_step = 0
    completed_epoch = 0
    exposure: dict[str, dict[str, Any]] = {}
    resume_path = _latest_checkpoint(ckpt_dir) if cfg.resume else None
    seed_path = Path(cfg.init_from) if cfg.init_from else None

    if resume_path is not None:
        if rank == 0:
            print(f"[sft] resuming from {resume_path}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            resume_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        state = torch.load(resume_path / "trainer_state.pt", map_location="cpu", weights_only=False)
        start_step = int(state.get("step", 0))
        completed_epoch = int(state.get("epoch", 0))
        exposure = dict(state.get("exposure") or {})
    elif seed_path is not None:
        if rank == 0:
            print(f"[sft] init weights from {seed_path} (fresh optimizer)", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            seed_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        completed_epoch = int(_read_epoch(seed_path) or 0)
        epoch_meta = seed_path / "epoch.json"
        if epoch_meta.exists():
            try:
                meta = json.loads(epoch_meta.read_text(encoding="utf-8"))
                start_step = int(meta.get("step") or 0)
                completed_epoch = int(meta.get("epoch") or completed_epoch)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        exp_path = seed_path / "exposure.json"
        if exp_path.exists():
            try:
                loaded = json.loads(exp_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    exposure = loaded
            except (json.JSONDecodeError, TypeError):
                pass
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )

    model.config.use_cache = False
    model.to(device)
    model.train()

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        foreach=cfg.adam_foreach,
    )
    if resume_path is not None:
        state = torch.load(resume_path / "trainer_state.pt", map_location="cpu", weights_only=False)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        canary_ids: list[str] = []
        for b in batch:
            canary_ids.extend(str(x) for x in (b.get("canary_ids") or []) if x)
        return {"input_ids": input_ids, "labels": labels, "canary_ids": canary_ids}

    step = start_step
    t0 = time.time()
    approx_steps = max(1, cfg.max_epochs * max(1, math.ceil(len(canaries) / max(1, cfg.per_device_batch_size * cfg.grad_accum_steps))))

    if rank == 0:
        print(
            f"[sft] model={cfg.model_name} world_size={world_size} "
            f"completed_epoch={completed_epoch} max_epochs={cfg.max_epochs} "
            f"canaries={len(canaries)} seq_len={cfg.seq_len} pack={cfg.pack} "
            f"lr={cfg.learning_rate}",
            flush=True,
        )

    wb = _wandb_init(
        cfg,
        rank=rank,
        extra={
            "n_canaries": len(canaries),
            "world_size": world_size,
            "resume_step": start_step,
            "completed_epoch": completed_epoch,
        },
    )

    def _finish_wandb() -> None:
        if wb is None:
            return
        try:
            wb.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[sft] wandb.finish failed: {exc}", flush=True)

    atexit.register(_finish_wandb)

    rng = torch.Generator()
    rng.manual_seed(cfg.seed)
    if resume_path is not None:
        saved_rng = state.get("data_loader_rng_state")
        if saved_rng is not None:
            rng.set_state(saved_rng)
        else:
            print("[sft] legacy checkpoint lacks DataLoader RNG state; shuffle continuity is not guaranteed", flush=True)

    for epoch in range(completed_epoch + 1, cfg.max_epochs + 1):
        epoch_rng = __import__("random").Random(cfg.seed + epoch * 1009 + rank)
        sequences = pack_canary_epoch(
            canaries,
            tokenizer,
            seq_len=cfg.seq_len,
            rng=epoch_rng,
            add_special_tokens=cfg.add_special_tokens,
            pack=cfg.pack,
        )
        dataset = PackedEpochDataset(sequences, seq_len=cfg.seq_len, pad_id=pad_id)
        loader = DataLoader(
            dataset,
            batch_size=cfg.per_device_batch_size,
            shuffle=True,
            collate_fn=_collate,
            num_workers=0,
            generator=rng,
        )
        running_loss = 0.0
        logged = 0
        micro = 0
        optimizer.zero_grad(set_to_none=True)
        accum_canary_ids: list[str] = []

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / cfg.grad_accum_steps
            loss.backward()
            running_loss += float(loss.item()) * cfg.grad_accum_steps
            logged += 1
            micro += 1

            raw_canaries = batch.get("canary_ids", [])
            batch_canary_ids: list[str] = []
            if isinstance(raw_canaries, list):
                for item in raw_canaries:
                    if isinstance(item, (list, tuple)):
                        batch_canary_ids.extend(str(x) for x in item if x)
                    elif item:
                        batch_canary_ids.append(str(item))
            accum_canary_ids.extend(batch_canary_ids)

            if micro % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = cosine_lr(
                    step,
                    base_lr=cfg.learning_rate,
                    warmup=cfg.warmup_steps,
                    max_steps=approx_steps,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                for cid in dict.fromkeys(accum_canary_ids):
                    rec = exposure.setdefault(
                        cid,
                        {
                            "first_step": step,
                            "last_step": step,
                            "count": 0,
                            "label": canary_label.get(cid, ""),
                        },
                    )
                    rec["last_step"] = step
                    rec["count"] = int(rec["count"]) + 1
                accum_canary_ids = []

                if rank == 0 and step % cfg.log_every == 0:
                    avg = running_loss / max(1, logged)
                    running_loss = 0.0
                    logged = 0
                    msg = {
                        "step": step,
                        "epoch": epoch,
                        "loss": avg,
                        "lr": lr,
                        "elapsed_sec": round(time.time() - t0, 1),
                        "n_canaries_seen": len(exposure),
                    }
                    print(json.dumps(msg), flush=True)
                    with log_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(msg) + "\n")
                    _wandb_log(
                        wb,
                        {
                            "train/loss": avg,
                            "train/lr": lr,
                            "train/epoch": epoch,
                            "train/n_canaries_seen": len(exposure),
                        },
                        step=step,
                    )

        if micro % cfg.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = cosine_lr(
                step,
                base_lr=cfg.learning_rate,
                warmup=cfg.warmup_steps,
                max_steps=approx_steps,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            for cid in dict.fromkeys(accum_canary_ids):
                rec = exposure.setdefault(
                    cid,
                    {
                        "first_step": step,
                        "last_step": step,
                        "count": 0,
                        "label": canary_label.get(cid, ""),
                    },
                )
                rec["last_step"] = step
                rec["count"] = int(rec["count"]) + 1
            accum_canary_ids = []

        if rank == 0 and cfg.probe_every_epoch > 0 and epoch % cfg.probe_every_epoch == 0:
            probe = probe_canaries(
                model=model,
                tokenizer=tokenizer,
                canaries=canaries,
                device=device,
                n=cfg.probe_canaries,
                add_special_tokens=cfg.add_special_tokens,
            )
            probe_msg = {"step": step, "epoch": epoch, "type": "canary_probe", **probe}
            print(json.dumps(probe_msg), flush=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(probe_msg) + "\n")
            _wandb_log(
                wb,
                {
                    "probe/nll": probe["probe_nll"],
                    "probe/token_acc": probe["probe_token_acc"],
                    "probe/n": probe["probe_n"],
                    "train/epoch": epoch,
                },
                step=step,
            )

        if rank == 0 and (epoch % max(1, cfg.save_every_epoch) == 0 or epoch >= cfg.max_epochs):
            path = ckpt_dir / f"step_{step:06d}"
            print(f"[sft] saving checkpoint epoch={epoch} {path}", flush=True)
            _save_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                step=step,
                epoch=epoch,
                exposure=exposure,
                cfg=cfg,
                data_loader_rng_state=rng.get_state(),
            )
            (ckpt_dir / "LATEST").write_text(str(path) + "\n", encoding="utf-8")
            _prune_sft_checkpoints(ckpt_dir, cfg)

        if distributed:
            torch.distributed.barrier()

    if distributed:
        torch.distributed.destroy_process_group()

    if wb is not None:
        wb.finish()

    return {
        "final_step": step,
        "final_epoch": cfg.max_epochs,
        "output_dir": str(out),
        "n_canaries_seen": len(exposure),
    }
