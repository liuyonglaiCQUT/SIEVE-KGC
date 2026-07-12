import math
import os
import time
from typing import Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from dict_hub import EntityDict, PositiveTripleIndex
from doc import Example
from evaluate import evaluate_model
from logger_config import logger
from triplet import TripletCollator, TripletDataset
from utils import dump_args, move_to_device, save_checkpoint


class Trainer:
    def __init__(
        self,
        args,
        model,
        tokenizer,
        train_examples: Sequence[Example],
        valid_examples: Sequence[Example],
        entity_dict: EntityDict,
        eval_positive_index: PositiveTripleIndex,
        evidence_map: Mapping[Tuple[str, str], Tuple[str, ...]],
        device: torch.device,
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.train_examples = list(train_examples)
        self.valid_examples = list(valid_examples)
        self.entity_dict = entity_dict
        self.eval_positive_index = eval_positive_index
        self.evidence_map = evidence_map
        self.device = device

        dataset = TripletDataset(self.train_examples)
        self.loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            drop_last=int(args.pre_batch) > 0,
            num_workers=int(args.num_workers),
            collate_fn=TripletCollator(
                tokenizer=tokenizer,
                max_length=args.max_num_tokens,
                use_self_negative=args.use_self_negative,
                entity_dict=entity_dict,
                evidence_map=evidence_map,
                use_sieve=bool(args.use_sieve and args.sieve_train),
                num_anchors=args.sieve_num_anchors,
                random_train_anchors=args.sieve_random_train_anchors,
                training=True,
            ),
            pin_memory=device.type == "cuda",
        )

        self.optimizer = AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        update_steps_per_epoch = math.ceil(
            len(self.loader) / max(1, int(args.gradient_accumulation_steps))
        )
        total_steps = max(1, update_steps_per_epoch * int(args.epochs))
        warmup_steps = int(total_steps * float(args.warmup_ratio))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.amp_enabled = bool(args.use_amp) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled,
            init_scale=float(args.amp_init_scale),
            growth_interval=4000,
        )
        logger.info(
            "AMP enabled=%s initial_scale=%.1f",
            self.amp_enabled,
            float(self.scaler.get_scale()),
        )

        os.makedirs(args.output_dir, exist_ok=True)
        dump_args(args, os.path.join(args.output_dir, "args.json"))

    def _compute_training_loss(self, logit_output):
        labels = logit_output["labels"]
        base_logits = logit_output["base_logits"]
        base_inbatch = logit_output["base_inbatch_logits"]

        base_forward = F.cross_entropy(base_logits, labels)
        if bool(self.args.bidirectional_inbatch_loss):
            base_reverse = F.cross_entropy(base_inbatch.t(), labels)
            base_loss = base_forward + base_reverse
        else:
            base_loss = base_forward

        sieve_loss = base_loss.new_zeros(())
        sieve_logits = logit_output.get("sieve_logits")
        if (
            bool(self.args.use_sieve)
            and bool(self.args.sieve_train)
            and sieve_logits is not None
        ):
            sieve_forward = F.cross_entropy(sieve_logits, labels)
            if bool(self.args.bidirectional_inbatch_loss):
                sieve_reverse = F.cross_entropy(sieve_logits.t(), labels)
                sieve_loss = sieve_forward + sieve_reverse
            else:
                sieve_loss = sieve_forward

        total = base_loss + float(self.args.sieve_aux_weight) * sieve_loss
        return total, base_loss, sieve_loss

    def train(self):
        best_mrr = -1.0
        bad_count = 0
        global_step = 0
        accumulation = max(1, int(self.args.gradient_accumulation_steps))
        best_path = os.path.join(self.args.output_dir, "model_best.mdl")
        last_path = os.path.join(self.args.output_dir, "model_last.mdl")

        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, int(self.args.epochs) + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_base_loss = 0.0
            epoch_sieve_loss = 0.0
            start = time.time()

            for step, batch in enumerate(self.loader, start=1):
                batch = move_to_device(batch, self.device)
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    outputs = self.model(**batch)
                    logit_output = self.model.compute_logits(outputs, batch)
                    total_loss, base_loss, sieve_loss = self._compute_training_loss(
                        logit_output
                    )
                    scaled_loss = total_loss / accumulation

                self.scaler.scale(scaled_loss).backward()
                epoch_loss += float(total_loss.detach().cpu())
                epoch_base_loss += float(base_loss.detach().cpu())
                epoch_sieve_loss += float(sieve_loss.detach().cpu())

                if step % accumulation == 0 or step == len(self.loader):
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(self.args.grad_clip)
                    )
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = self.scaler.get_scale()
                    optimizer_was_run = scale_after >= scale_before
                    self.optimizer.zero_grad(set_to_none=True)

                    if optimizer_was_run:
                        self.scheduler.step()
                        global_step += 1
                    else:
                        logger.warning(
                            "AMP skipped optimizer update at epoch=%d batch=%d: "
                            "scale %.1f -> %.1f grad_norm=%s",
                            epoch,
                            step,
                            scale_before,
                            scale_after,
                            str(float(grad_norm)),
                        )

                    if optimizer_was_run and global_step % int(self.args.print_freq) == 0:
                        logger.info(
                            "epoch=%d step=%d loss=%.5f base=%.5f sieve=%.5f "
                            "inv_t=%.3f lr=%.3e amp_scale=%.1f",
                            epoch,
                            global_step,
                            epoch_loss / max(1, step),
                            epoch_base_loss / max(1, step),
                            epoch_sieve_loss / max(1, step),
                            float(logit_output["inv_t"].cpu()),
                            self.optimizer.param_groups[0]["lr"],
                            scale_after,
                        )

            avg_loss = epoch_loss / max(1, len(self.loader))
            avg_base = epoch_base_loss / max(1, len(self.loader))
            avg_sieve = epoch_sieve_loss / max(1, len(self.loader))
            logger.info(
                "Epoch %d finished | loss=%.6f base=%.6f sieve=%.6f | time=%.1fs",
                epoch,
                avg_loss,
                avg_base,
                avg_sieve,
                time.time() - start,
            )
            save_checkpoint(
                last_path,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                best_mrr,
                self.args,
            )

            if epoch % int(self.args.eval_every) != 0:
                continue

            metrics = evaluate_model(
                model=self.model,
                examples=self.valid_examples,
                entity_dict=self.entity_dict,
                tokenizer=self.tokenizer,
                args=self.args,
                positive_index=self.eval_positive_index,
                evidence_map=self.evidence_map,
                device=self.device,
            )
            logger.info("Validation epoch=%d: %s", epoch, metrics)

            if metrics["mrr"] > best_mrr:
                best_mrr = float(metrics["mrr"])
                bad_count = 0
                save_checkpoint(
                    best_path,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    best_mrr,
                    self.args,
                )
                logger.info("New best MRR %.6f saved to %s", best_mrr, best_path)
            else:
                bad_count += 1
                if int(self.args.early_stop_patience) > 0 and bad_count >= int(
                    self.args.early_stop_patience
                ):
                    logger.info("Early stop: best MRR %.6f", best_mrr)
                    break

        return best_path
