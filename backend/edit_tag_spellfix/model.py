# edit_tag_spellfix/model.py
"""
RoBERTa edit-tag model, now with an option to *freeze* the encoder so
quick warm-start runs finish in minutes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    RobertaConfig,
    RobertaPreTrainedModel,
)

class RobertaTagger(RobertaPreTrainedModel):
    """
    Token-classification head → edit-tags.
    """

    def __init__(self, config: RobertaConfig, num_tags: int, freeze_encoder: bool = False):
        super().__init__(config)
        self.num_tags = num_tags

        # encoder
        self.roberta = AutoModel.from_config(config)
        if freeze_encoder:
            for p in self.roberta.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_tags)

        self.post_init()

    # --------------------------------------------------
    # Convenient ctor
    # --------------------------------------------------
    @classmethod
    def from_pretrained_with_tags(
        cls,
        base_model_name_or_path: str,
        tag2id_path: str | Path,
        freeze_encoder: bool = False,
        **kwargs,
    ):
        tag2id = json.load(open(tag2id_path, encoding="utf-8"))

        num_tags = len(tag2id)

        cfg = AutoConfig.from_pretrained(
            base_model_name_or_path,
            num_labels=num_tags,
            hidden_dropout_prob=kwargs.pop("dropout", 0.1),
        )
        # load weights (will ignore size-mismatch in classifier)
        model = cls.from_pretrained(
            base_model_name_or_path,
            config=cfg,
            num_tags=num_tags,
            freeze_encoder=freeze_encoder,
            ignore_mismatched_sizes=True,   # new tags → new classifier rows
        )
        model.tag2id = tag2id
        model.id2tag = {v: k for k, v in tag2id.items()}
        return model

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.num_tags),
                labels.view(-1),
            )

        return {"loss": loss, "logits": logits}
