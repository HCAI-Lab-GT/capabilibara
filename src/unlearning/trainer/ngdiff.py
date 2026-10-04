# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportCallIssue=false, reportArgumentType=false
"""
NGDiff trainer based on Algorithm 1 from Bu & Xu (NAACL 2025).

Normalized Gradient Difference:
    g = g_R / ||g_R|| - g_F / ||g_F||

Two extra forward passes on retain every AUTO_LR_INTERVAL optimizer steps
fit a quadratic to find the optimal learning rate automatically.

This implementation also supports length-normalized loss, periodic retain
resampling, optional retain-only cooldown phases, and periodic PPL checks.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, cast
import weakref

import torch
from transformers import TrainerCallback

from unlearning.data.dolma_pool import ForgetRetainDataset

from .base import UnlearnTrainer
from .cooldown import CooldownConfig, CooldownManager
from .utils import compute_perplexity, per_document_loss

logger = logging.getLogger(__name__)

AUTO_LR_INTERVAL = 10
LOG_INTERVAL = 50
PPL_CHECK_INTERVAL = 200
PPL_LOG_INTERVAL = 200
MAX_STEPS = 5000
MMLU_SAFETY_THRESHOLD = 0.90
EPS = 1e-8


def _normalize_max_walltime_minutes(value: float | None) -> float | None:
    if value is None or value == 0:
        return None
    if not math.isfinite(value) or value < 0:
        raise ValueError("max_walltime_minutes must be finite and non-negative")
    return value


def _grad_norm(grads: dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum((g.float() ** 2).sum().item() for g in grads.values())) + EPS


def _cast_grad_for_param(param: torch.nn.Parameter, grad: torch.Tensor) -> torch.Tensor:
    return grad.to(device=param.device, dtype=param.dtype)


def _set_grad_for_param(param: torch.nn.Parameter, grad: torch.Tensor) -> None:
    cast_grad = _cast_grad_for_param(param, grad)
    if param.grad is None or param.grad.shape != cast_grad.shape:
        param.grad = cast_grad
    else:
        param.grad.detach().copy_(cast_grad)


def _persist_mmlu_stop_evidence(
    output_dir: str | os.PathLike[str],
    *,
    global_step: int,
    micro_step: int,
    baseline: float,
    observed_accuracy: float,
) -> Path:
    """Atomically bind the exact MMLU measurement that stopped training.

    The two counters are not interchangeable and must not be conflated:

    ``global_step`` is HF's ``TrainerState.global_step`` -- the number of real
    optimizer steps (parameter updates) the saved adapter received. It is the
    counter ``training_metadata.json`` reports, so it is the one the DCLM
    receipt binds against.

    ``micro_step`` is NGDiff's own ``_step_count`` -- the ``training_step`` call
    at which the MMLU accuracy was measured.

    Deriving the first from the second (``_step_count // grad_accum``)
    undercounts, because HF forces an optimizer step on the last batch of every
    epoch whether or not the accumulation group is full. At 50 batches/epoch
    with ``grad_accum=4`` that is 13 real updates per epoch against a derived
    12.5, which drifted 51 steps apart over 102 epochs in production.
    """

    threshold = baseline * MMLU_SAFETY_THRESHOLD
    if (
        global_step <= 0
        or micro_step <= 0
        or not all(
            math.isfinite(value) for value in (baseline, threshold, observed_accuracy)
        )
        or observed_accuracy >= threshold
    ):
        raise ValueError("MMLU stop evidence does not describe a threshold crossing")
    destination = Path(output_dir) / "mmlu_guard_evidence.json"
    payload = {
        "baseline": baseline,
        "global_step": global_step,
        "micro_step": micro_step,
        "observed_accuracy": observed_accuracy,
        "threshold": threshold,
        "threshold_fraction": MMLU_SAFETY_THRESHOLD,
        "triggered": True,
    }
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.read_bytes() != content:
            raise ValueError("existing MMLU stop evidence differs")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != content:
                raise ValueError("existing MMLU stop evidence differs")
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _persist_hard_max_steps_evidence(
    output_dir: str | os.PathLike[str],
    *,
    ceiling: int,
    global_step: int,
) -> Path:
    """Bind the trainer's own step ceiling and the step at which it tripped.

    ``hard_max_steps`` means NGDiff's backstop fired, which by definition is a
    ceiling *tighter* than the plan's ``max_steps`` -- otherwise HF stops the
    run first and the reason is plain ``max_steps``. The receipt therefore
    cannot assume the run halted at the plan's ceiling, so the trip records
    which ceiling fired and where.

    Both numbers are HF's ``TrainerState.global_step``, never a floor-division
    proxy over micro-steps.
    """

    if ceiling <= 0 or global_step != ceiling:
        raise ValueError("hard max-steps evidence does not describe an exact trip")
    destination = Path(output_dir) / "hard_max_steps_evidence.json"
    payload = {"ceiling": ceiling, "global_step": global_step, "triggered": True}
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.read_bytes() != content:
            raise ValueError("existing hard max-steps evidence differs")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != content:
                raise ValueError("existing hard max-steps evidence differs")
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _snapshot_existing_grads(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def _accumulate_microstep_grads(
    model: torch.nn.Module,
    previous_grads: dict[str, torch.Tensor],
    current_grads: dict[str, torch.Tensor],
    grad_scale: float,
) -> None:
    for n, p in model.named_parameters():
        previous_grad = previous_grads.get(n)
        current_grad = current_grads.get(n)
        if previous_grad is None and current_grad is None:
            p.grad = None
        elif previous_grad is None and current_grad is not None:
            _set_grad_for_param(p, current_grad * grad_scale)
        elif previous_grad is not None and current_grad is None:
            _set_grad_for_param(p, previous_grad)
        elif previous_grad is not None and current_grad is not None:
            previous_cast = _cast_grad_for_param(p, previous_grad)
            current_cast = _cast_grad_for_param(p, current_grad)
            if previous_cast.shape == current_cast.shape:
                _set_grad_for_param(p, previous_cast + current_cast * grad_scale)
            else:
                logger.warning(
                    "Dropping incompatible accumulated gradient for %s: previous=%s current=%s",
                    n,
                    tuple(previous_cast.shape),
                    tuple(current_cast.shape),
                )
                _set_grad_for_param(p, current_cast * grad_scale)


class MmluStopEvidenceFlush(TrainerCallback):
    """Seal pending MMLU-stop evidence against HF's authoritative step counter.

    NGDiff detects the MMLU crossing inside ``training_step``, before the
    in-flight accumulation group has stepped the optimizer, so
    ``state.global_step`` is not yet terminal there. HF then takes exactly one
    of two paths before it breaks out of the training loop:

    * the micro-step was a sync step -- HF steps the optimizer, increments
      ``global_step``, then calls ``on_step_end``;
    * it was not -- HF calls ``on_substep_end`` and breaks without stepping,
      leaving ``global_step`` already terminal.

    Flushing from both hooks therefore records the same value the run's
    ``training_metadata.json`` will report, whichever path is taken.
    """

    def __init__(self, trainer: NGDiff) -> None:
        self._trainer = weakref.ref(trainer)

    def _flush(self, state) -> None:
        trainer = self._trainer()
        if trainer is not None:
            trainer.flush_pending_mmlu_stop_evidence(int(state.global_step))

    def on_step_end(self, args, state, control, **kwargs) -> None:
        self._flush(state)
        trainer = self._trainer()
        if trainer is not None:
            # Only here, never on_substep_end: the ceiling counts completed
            # optimizer steps, and global_step has just been incremented.
            trainer.check_hard_step_ceiling(int(state.global_step))

    def on_substep_end(self, args, state, control, **kwargs) -> None:
        self._flush(state)


class NGDiff(UnlearnTrainer):
    """Normalized Gradient Difference unlearning trainer.

    Core constructor args:
        auto_lr, lr_delta, mmlu_baseline, mmlu_eval_samples,
        mmlu_choice_tokens, forget_eval_loader, max_walltime_minutes,
        ppl_stopping_threshold

    Optional method args:
        retain_pool             - RetainPool for periodic resampling
        length_normalized_loss  - per-doc mean then batch mean
        cooldown_config         - CooldownConfig for retain-only phases
    """

    # Class-level default so the attribute exists even on instances built
    # without __init__ (test harnesses use object.__new__).
    _pending_mmlu_stop: dict[str, object] | None = None

    def __init__(
        self,
        *args,
        auto_lr: bool = True,
        lr_delta: float = 1e-5,
        mmlu_baseline: float | None = None,
        mmlu_eval_samples=None,
        mmlu_choice_tokens=None,
        forget_eval_loader=None,
        retain_eval_loader=None,
        max_walltime_minutes: float | None = None,
        ppl_stopping_threshold: float | None = None,
        retain_pool=None,
        length_normalized_loss: bool = True,
        cooldown_config: CooldownConfig | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.auto_lr = auto_lr
        self.lr_delta = lr_delta
        self.mmlu_baseline = mmlu_baseline
        self.mmlu_eval_samples = mmlu_eval_samples or []
        self.mmlu_choice_tokens = mmlu_choice_tokens
        self.forget_eval_loader = forget_eval_loader
        self.retain_eval_loader = retain_eval_loader
        self.max_walltime_minutes = _normalize_max_walltime_minutes(
            max_walltime_minutes
        )
        self.ppl_stopping_threshold = ppl_stopping_threshold
        self.retain_pool = retain_pool
        self.length_normalized_loss = length_normalized_loss
        self.cooldown_mgr = CooldownManager(cooldown_config or CooldownConfig())
        self._step_count = 0
        self._optimizer_steps = 0
        self._job_start_time = time.time()
        self._ppl_log: dict[int, float] = {}
        self._resumed = False
        self._stopped_permanently = False
        self._stop_reason: str | None = None
        self._pending_mmlu_stop = None
        self.add_callback(MmluStopEvidenceFlush(self))

    def flush_pending_mmlu_stop_evidence(self, global_step: int) -> None:
        """Write the stashed MMLU crossing against HF's terminal step count.

        Idempotent: the pending record is cleared on the first flush, so the
        second hook of the same step is a no-op.
        """
        pending = self._pending_mmlu_stop
        if pending is None:
            return
        self._pending_mmlu_stop = None
        _persist_mmlu_stop_evidence(
            self.args.output_dir,
            global_step=global_step,
            micro_step=cast(int, pending["micro_step"]),
            baseline=cast(float, pending["baseline"]),
            observed_accuracy=cast(float, pending["observed_accuracy"]),
        )

    def check_hard_step_ceiling(self, global_step: int) -> None:
        """Trip NGDiff's own backstop, counted in real optimizer steps.

        The backstop only means anything when it is tighter than the plan's
        ``max_steps``: otherwise HF ends the run at its own ceiling first and
        the reason is plain ``max_steps``. Firing in that case would relabel
        every ordinary max-steps run as ``hard_max_steps``, so it is skipped.
        """
        configured = getattr(self.args, "max_steps", 0) or 0
        if configured > 0 and MAX_STEPS >= configured:
            return
        if global_step < MAX_STEPS or self._stopped_permanently:
            return
        logger.info("Hard ceiling %d optimizer steps reached.", MAX_STEPS)
        _persist_hard_max_steps_evidence(
            self.args.output_dir, ceiling=MAX_STEPS, global_step=global_step
        )
        self._stopped_permanently = True
        self._stop_reason = "hard_max_steps"
        self._signal_stop()

    def _signal_stop(self, *, save: bool = True) -> None:
        if hasattr(self, "control") and self.control is not None:
            self.control.should_training_stop = True
            self.control.should_save = save

    def _sync_step_count_on_resume(self) -> None:
        if self._resumed:
            return
        self._resumed = True
        if (
            hasattr(self, "state")
            and self.state is not None
            and self.state.global_step > 0
        ):
            grad_accum = self.args.gradient_accumulation_steps
            self._step_count = self.state.global_step * grad_accum
            logger.info(
                "Resumed from checkpoint: global_step=%d, _step_count restored to %d",
                self.state.global_step,
                self._step_count,
            )

    def _compute_loss(self, model_out, inputs) -> torch.Tensor:
        if self.length_normalized_loss:
            return per_document_loss(model_out.logits, inputs["labels"])
        return model_out.loss

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._sync_step_count_on_resume()
        model.train()

        forget_inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v
            for k, v in inputs["forget"].items()
        }
        retain_inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v
            for k, v in inputs["retain"].items()
        }

        previous_grads = _snapshot_existing_grads(model)
        in_cooldown = self.cooldown_mgr.in_cooldown()

        norm_F = 0.0
        norm_R = 0.0
        microstep_grads: dict[str, torch.Tensor] = {}

        if in_cooldown:
            model.zero_grad()
            retain_out = model(**retain_inputs)
            retain_loss = self._compute_loss(retain_out, retain_inputs)
            self.accelerator.backward(retain_loss)
            g_R = {
                n: p.grad.clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            norm_R = _grad_norm(g_R)
            for n, _p in model.named_parameters():
                if n in g_R:
                    microstep_grads[n] = g_R[n] / norm_R
            forget_loss = retain_loss.new_zeros(())
            self.cooldown_mgr.step()
        else:
            model.zero_grad()
            forget_out = model(**forget_inputs)
            forget_loss = self._compute_loss(forget_out, forget_inputs)
            self.accelerator.backward(forget_loss)
            g_F = {
                n: p.grad.clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }

            model.zero_grad()
            retain_out = model(**retain_inputs)
            retain_loss = self._compute_loss(retain_out, retain_inputs)
            self.accelerator.backward(retain_loss)
            g_R = {
                n: p.grad.clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }

            norm_F = _grad_norm(g_F)
            norm_R = _grad_norm(g_R)

            for n, _p in model.named_parameters():
                if n in g_F and n in g_R:
                    microstep_grads[n] = g_R[n] / norm_R - g_F[n] / norm_F
                elif n in g_R:
                    microstep_grads[n] = g_R[n] / norm_R

        grad_accum = self.args.gradient_accumulation_steps
        grad_scale = 1.0 / max(1, int(grad_accum))
        _accumulate_microstep_grads(
            model,
            previous_grads=previous_grads,
            current_grads=microstep_grads,
            grad_scale=grad_scale,
        )

        self._step_count += 1
        optimizer_steps = self._step_count // grad_accum
        is_optimizer_step = self._step_count % grad_accum == 0

        # The hard ceiling is checked in MmluStopEvidenceFlush.on_step_end
        # against HF's global_step. Testing optimizer_steps here would trip it
        # late, because floor division over micro-steps undercounts real
        # updates by one per partial epoch-end accumulation group.

        if self.max_walltime_minutes is not None:
            elapsed = (time.time() - self._job_start_time) / 60.0
            remaining = self.max_walltime_minutes - elapsed
            if remaining < 30.0:
                logger.info(
                    "Wall-time guard: %.1f min elapsed, %.1f min remaining.",
                    elapsed,
                    remaining,
                )
                if not self._stopped_permanently:
                    self._stop_reason = "walltime_checkpoint"
                self._signal_stop(save=False)

        if is_optimizer_step and not in_cooldown:
            if self.cooldown_mgr.should_cooldown(optimizer_steps):
                self.cooldown_mgr.enter_cooldown(optimizer_steps)

        if (
            is_optimizer_step
            and self.retain_pool is not None
            and self.retain_pool.should_resample(optimizer_steps)
        ):
            new_retain = self.retain_pool.resample(optimizer_steps)
            if isinstance(self.train_dataset, ForgetRetainDataset):
                self.train_dataset.retain = new_retain
            else:
                logger.warning("RetainPool configured for unsupported train dataset.")
            logger.info(
                "Retain resampled at optimizer_step=%d (%d docs)",
                optimizer_steps,
                len(new_retain),
            )

        if (
            self.auto_lr
            and is_optimizer_step
            and optimizer_steps > 0
            and optimizer_steps % AUTO_LR_INTERVAL == 0
        ):
            self._auto_lr_step(model, retain_inputs)

        if self._step_count % LOG_INTERVAL == 0:
            mmlu_acc = self._run_mmlu_eval(model)
            self._log_metrics(
                model,
                forget_loss,
                retain_loss,
                mmlu_acc=mmlu_acc,
                norm_F=norm_F,
                norm_R=norm_R,
                in_cooldown=in_cooldown,
            )
            if not math.isnan(mmlu_acc) and self.mmlu_baseline is not None:
                threshold = self.mmlu_baseline * MMLU_SAFETY_THRESHOLD
                if mmlu_acc < threshold:
                    logger.warning(
                        "MMLU safety gate at step %d: acc %.4f < %.4f",
                        self._step_count,
                        mmlu_acc,
                        threshold,
                    )
                    # Stash rather than write: state.global_step is not yet
                    # terminal here, and optimizer_steps is a floor-division
                    # proxy that undercounts real updates. The registered
                    # MmluStopEvidenceFlush seals this against HF's counter.
                    self._pending_mmlu_stop = {
                        "micro_step": self._step_count,
                        "baseline": self.mmlu_baseline,
                        "observed_accuracy": mmlu_acc,
                    }
                    self._stopped_permanently = True
                    self._stop_reason = "mmlu_stop"
                    self._signal_stop()

        if (
            self._step_count % PPL_CHECK_INTERVAL == 0
            and self.forget_eval_loader is not None
        ):
            device = next(model.parameters()).device
            forget_ppl = compute_perplexity(
                model,
                self.forget_eval_loader,
                device,
            )
            retain_ppl = float("nan")
            if self.retain_eval_loader is not None:
                retain_ppl = compute_perplexity(
                    model,
                    self.retain_eval_loader,
                    device,
                )
            model.train()
            logger.info(
                "PPL at optimizer_step=%d: forget=%.4f  retain=%.4f",
                optimizer_steps,
                forget_ppl,
                retain_ppl if not math.isnan(retain_ppl) else -1,
            )
            self._ppl_log[optimizer_steps] = round(forget_ppl, 4)
            self._save_ppl_log()
            self._log_wandb_ppl(
                optimizer_steps,
                forget_ppl,
                retain_ppl,
            )
            if (
                self.ppl_stopping_threshold is not None
                and forget_ppl >= self.ppl_stopping_threshold
            ):
                logger.info(
                    "PPL stopping triggered at optimizer_step=%d: "
                    "PPL=%.4f >= threshold=%.4f",
                    optimizer_steps,
                    forget_ppl,
                    self.ppl_stopping_threshold,
                )
                self._stopped_permanently = True
                self._stop_reason = "ppl_stop"
                self._signal_stop()

        model.train()
        return (forget_loss + retain_loss).detach()

    def _save_ppl_log(self) -> None:
        try:
            path = os.path.join(str(self.args.output_dir or "."), "forget_ppl_log.json")
            with open(path, "w") as f:
                json.dump(self._ppl_log, f, indent=2)
        except Exception as e:
            logger.warning("Could not save forget_ppl_log.json: %s", e)

    def _auto_lr_step(
        self,
        model: torch.nn.Module,
        retain_inputs: dict,
    ) -> None:
        optimizer = cast(Any, self.optimizer)
        if optimizer is None:
            return

        current_lr = optimizer.param_groups[0]["lr"]
        delta = self.lr_delta

        probe_inputs = {
            k: v[:1] if isinstance(v, torch.Tensor) else v
            for k, v in retain_inputs.items()
        }
        param_backup = {n: p.data.clone() for n, p in model.named_parameters()}

        def probe_loss(lr_probe: float) -> float:
            torch.cuda.empty_cache()
            for pg in optimizer.param_groups:
                pg["lr"] = lr_probe
            optimizer.step()
            model.eval()
            with torch.no_grad():
                out = model(**probe_inputs)
                loss = out.loss.item()
            model.train()
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(param_backup[n])
            return loss

        try:
            l_minus = probe_loss(current_lr - delta)
            l_plus = probe_loss(current_lr + delta)
            l_curr = probe_loss(current_lr)
            a = (l_minus - 2 * l_curr + l_plus) / (2 * delta**2)
            b = (l_plus - l_minus) / (2 * delta)
            if abs(a) < 1e-30:
                return
            optimal_lr = max(1e-7, min(1e-3, -b / (2 * a)))
            for pg in optimizer.param_groups:
                pg["lr"] = optimal_lr
        except Exception as e:
            logger.warning("AutoLR probe failed at step %d: %s", self._step_count, e)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
        finally:
            torch.cuda.empty_cache()

    def _run_mmlu_eval(self, model) -> float:
        if not self.mmlu_eval_samples or self.mmlu_choice_tokens is None:
            return float("nan")
        device = next(model.parameters()).device
        choice_tokens = self.mmlu_choice_tokens.to(device)
        model.eval()
        correct = 0
        with torch.no_grad():
            for input_ids, attn_mask, answer_idx in self.mmlu_eval_samples:
                out = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attn_mask.to(device),
                )
                last_logits = out.logits[0, -1, :]
                if last_logits[choice_tokens].argmax().item() == answer_idx:
                    correct += 1
        model.train()
        return correct / len(self.mmlu_eval_samples)

    def _log_wandb_ppl(
        self,
        optimizer_steps: int,
        forget_ppl: float,
        retain_ppl: float,
    ) -> None:
        if not (self.args.report_to and "wandb" in self.args.report_to):
            return
        try:
            import wandb

            payload: dict = {
                "eval/forget_ppl": forget_ppl,
                "train/optimizer_step": optimizer_steps,
            }
            if not math.isnan(retain_ppl):
                payload["eval/retain_ppl"] = retain_ppl
            if self.ppl_stopping_threshold is not None:
                payload["eval/ppl_target"] = self.ppl_stopping_threshold
            wandb.log(payload)
        except Exception:
            pass

    def _log_metrics(
        self,
        model,
        forget_loss: torch.Tensor,
        retain_loss: torch.Tensor,
        mmlu_acc: float = float("nan"),
        norm_F: float = 0.0,
        norm_R: float = 0.0,
        in_cooldown: bool = False,
    ):
        optimizer = cast(Any, self.optimizer)
        lr = optimizer.param_groups[0]["lr"] if optimizer else float("nan")
        grad_accum = self.args.gradient_accumulation_steps
        optimizer_steps = self._step_count // grad_accum
        mmlu_str = f"{mmlu_acc:.4f}" if not math.isnan(mmlu_acc) else "n/a"
        logger.info(
            "step=%d  opt_step=%d  f_loss=%.4f  r_loss=%.4f  mmlu=%s  lr=%.2e",
            self._step_count,
            optimizer_steps,
            forget_loss.item(),
            retain_loss.item(),
            mmlu_str,
            lr,
        )
        if self.args.report_to and "wandb" in self.args.report_to:
            try:
                import wandb

                payload: dict = {
                    "train/forget_loss": forget_loss.item(),
                    "train/retain_loss": retain_loss.item(),
                    "train/lr": lr,
                    "train/training_step": self._step_count,
                    "train/optimizer_step": optimizer_steps,
                    "train/grad_norm_forget": norm_F,
                    "train/grad_norm_retain": norm_R,
                    "train/in_cooldown": int(in_cooldown),
                }
                if not math.isnan(mmlu_acc):
                    payload["train/mmlu_acc"] = mmlu_acc
                wandb.log(payload)
            except Exception:
                pass
