from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from triplet_mask import construct_mask


def build_model(args) -> nn.Module:
    return SimKGCSIEVERAAStyleModel(args)


@dataclass
class ModelOutput:
    base_logits: torch.Tensor
    base_inbatch_logits: torch.Tensor
    sieve_logits: torch.Tensor
    labels: torch.Tensor
    inv_t: torch.Tensor


class SimKGCSIEVERAAStyleModel(nn.Module, ABC):
    """SimKGC with an RAA-style SIEVE auxiliary query branch.

    The base and SIEVE-enhanced queries share the same `hr_bert` encoder.  One
    enhanced query is encoded per selected structural evidence entity and those
    vectors are averaged, matching the anchor-query aggregation pattern of the
    public RAA-KGC implementation.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        local_only = os.path.isdir(str(args.pretrained_model))
        self.config = AutoConfig.from_pretrained(
            args.pretrained_model, local_files_only=local_only
        )
        self.log_inv_t = nn.Parameter(
            torch.tensor(1.0 / float(args.t)).log(),
            requires_grad=bool(args.finetune_t),
        )
        self.add_margin = float(args.additive_margin)
        self.batch_size = int(args.batch_size)
        self.pre_batch = int(args.pre_batch)
        self.use_sieve = bool(getattr(args, "use_sieve", True))

        pre_batch_size = max(1, self.pre_batch) * self.batch_size
        random_vector = F.normalize(
            torch.randn(pre_batch_size, self.config.hidden_size), dim=1
        )
        self.register_buffer("pre_batch_vectors", random_vector, persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(pre_batch_size)]

        self.hr_bert = AutoModel.from_pretrained(
            args.pretrained_model, local_files_only=local_only
        )
        self.tail_bert = deepcopy(self.hr_bert)

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(
            input_ids=token_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        return _pool_output(
            self.args.pooling, cls_output, mask, last_hidden_state
        )

    def encode_hr(self, token_ids, mask, token_type_ids):
        return self._encode(self.hr_bert, token_ids, mask, token_type_ids)

    def encode_entities(self, token_ids, mask, token_type_ids):
        return self._encode(self.tail_bert, token_ids, mask, token_type_ids)

    def _aggregate_sieve_queries(
        self,
        flat_vectors: torch.Tensor,
        owner: torch.Tensor,
        count: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        output = flat_vectors.new_zeros(batch_size, flat_vectors.size(-1))
        output.index_add_(0, owner, flat_vectors)
        output = output / count.to(output.dtype).unsqueeze(1)
        # RAA code averages anchor-query vectors without an extra normalization.
        return output

    def encode_query_batch(self, batch_dict):
        base_vector = self.encode_hr(
            batch_dict["hr_token_ids"],
            batch_dict["hr_mask"],
            batch_dict["hr_token_type_ids"],
        )
        sieve_vector = None
        if self.use_sieve and "sieve_hr_token_ids" in batch_dict:
            flat = self.encode_hr(
                batch_dict["sieve_hr_token_ids"],
                batch_dict["sieve_hr_mask"],
                batch_dict["sieve_hr_token_type_ids"],
            )
            sieve_vector = self._aggregate_sieve_queries(
                flat,
                batch_dict["sieve_owner"],
                batch_dict["sieve_count"],
                base_vector.size(0),
            )
        return base_vector, sieve_vector

    def forward(
        self,
        hr_token_ids,
        hr_mask,
        hr_token_type_ids,
        tail_token_ids,
        tail_mask,
        tail_token_type_ids,
        head_token_ids,
        head_mask,
        head_token_type_ids,
        sieve_hr_token_ids=None,
        sieve_hr_mask=None,
        sieve_hr_token_type_ids=None,
        sieve_owner=None,
        sieve_count=None,
        only_ent_embedding=False,
        **kwargs
    ) -> dict:
        if only_ent_embedding:
            return {
                "ent_vectors": self.encode_entities(
                    tail_token_ids, tail_mask, tail_token_type_ids
                ).detach()
            }

        base_vector = self.encode_hr(
            hr_token_ids, hr_mask, hr_token_type_ids
        )
        tail_vector = self.encode_entities(
            tail_token_ids, tail_mask, tail_token_type_ids
        )
        head_vector = self.encode_entities(
            head_token_ids, head_mask, head_token_type_ids
        )

        sieve_vector = None
        if self.use_sieve and sieve_hr_token_ids is not None:
            flat_sieve = self.encode_hr(
                sieve_hr_token_ids,
                sieve_hr_mask,
                sieve_hr_token_type_ids,
            )
            sieve_vector = self._aggregate_sieve_queries(
                flat_sieve,
                sieve_owner,
                sieve_count,
                base_vector.size(0),
            )

        return {
            "hr_vector": base_vector,
            "sieve_hr_vector": sieve_vector,
            "tail_vector": tail_vector,
            "head_vector": head_vector,
        }

    def _apply_inbatch_margin_mask(self, logits, triplet_mask):
        batch_size = logits.size(0)
        if self.training:
            logits = logits - torch.eye(
                batch_size, device=logits.device, dtype=logits.dtype
            ) * self.add_margin
        logits = logits * self.log_inv_t.exp()
        if triplet_mask is not None:
            logits = logits.masked_fill(
                ~triplet_mask.to(logits.device), -1e4
            )
        return logits

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector = output_dict["hr_vector"]
        tail_vector = output_dict["tail_vector"]
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size, device=hr_vector.device)
        triplet_mask = batch_dict.get("triplet_mask")

        base_inbatch_logits = hr_vector.mm(tail_vector.t())
        base_inbatch_logits = self._apply_inbatch_margin_mask(
            base_inbatch_logits, triplet_mask
        )
        base_logits = base_inbatch_logits

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(
                hr_vector, tail_vector, batch_dict
            )
            base_logits = torch.cat([base_logits, pre_batch_logits], dim=-1)

        if bool(self.args.use_self_negative) and self.training:
            head_vector = output_dict["head_vector"]
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1)
            self_neg_logits = self_neg_logits * self.log_inv_t.exp()
            self_negative_mask = batch_dict["self_negative_mask"].to(
                base_logits.device
            )
            self_neg_logits = self_neg_logits.masked_fill(
                ~self_negative_mask, -1e4
            )
            base_logits = torch.cat(
                [base_logits, self_neg_logits.unsqueeze(1)], dim=-1
            )

        sieve_logits = None
        sieve_vector = output_dict.get("sieve_hr_vector")
        if sieve_vector is not None:
            sieve_logits = sieve_vector.mm(tail_vector.t())
            sieve_logits = self._apply_inbatch_margin_mask(
                sieve_logits, triplet_mask
            )

        return {
            "base_logits": base_logits,
            "base_inbatch_logits": base_inbatch_logits,
            "sieve_logits": sieve_logits,
            "labels": labels,
            "inv_t": self.log_inv_t.detach().exp(),
        }

    def _compute_pre_batch_logits(self, hr_vector, tail_vector, batch_dict):
        if tail_vector.size(0) != self.batch_size:
            raise ValueError(
                "Pre-batch negatives require a fixed dropped-last batch: "
                "expected {}, got {}".format(
                    self.batch_size, tail_vector.size(0)
                )
            )
        batch_examples = batch_dict["batch_data"]
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits = (
            pre_batch_logits
            * self.log_inv_t.exp()
            * float(self.args.pre_batch_weight)
        )
        if self.pre_batch_exs[-1] is not None:
            mask = construct_mask(batch_examples, self.pre_batch_exs).to(
                hr_vector.device
            )
            pre_batch_logits = pre_batch_logits.masked_fill(~mask, -1e4)

        end = self.offset + self.batch_size
        self.pre_batch_vectors[self.offset:end] = tail_vector.detach().clone()
        self.pre_batch_exs[self.offset:end] = list(batch_examples)
        self.offset = end % len(self.pre_batch_exs)
        return pre_batch_logits


CustomBertModel = SimKGCSIEVERAAStyleModel


def _pool_output(pooling, cls_output, mask, last_hidden_state):
    if pooling == "cls":
        output = cls_output
    elif pooling == "max":
        expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).bool()
        output = last_hidden_state.masked_fill(~expanded, -1e4).max(dim=1).values
    elif pooling == "mean":
        expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        output = (last_hidden_state * expanded).sum(dim=1)
        output = output / expanded.sum(dim=1).clamp_min(1e-4)
    else:
        raise ValueError("Unknown pooling mode: {}".format(pooling))
    return F.normalize(output, dim=1)
