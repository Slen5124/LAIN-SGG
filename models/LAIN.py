"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

import numpy as np
import wandb

from utils.hico_list import hico_verbs_sentence
from utils.vcoco_list import vcoco_verbs_sentence
from utils.hico_utils import reserve_indices
from utils.postprocessor import PostProcess
from utils.ops import binary_focal_loss_with_logits
from utils import hico_text_label

from CLIP.clip import build_model
from CLIP.customCLIP import CustomCLIP, tokenize

sys.path.insert(0, 'detr')
from detr.models.backbone import build_backbone
from detr.models.transformer import build_transformer
from detr.models.detr import DETR
from detr.util import box_ops
from detr.util.misc import nested_tensor_from_tensor_list
sys.path.pop(0)
from models.egtr_detector import (
    EgtrPostProcess,
    load_egtr_vg_detector,
)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LAIN(nn.Module):
    def __init__(self,
        args,
        detector: nn.Module,
        postprocessor: nn.Module,
        model: nn.Module,
        object_embedding: torch.tensor,
        human_idx: int, num_classes: int,
        alpha: float = 0.5, gamma: float = 2.0,
        box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
        min_instances: int = 3, max_instances: int = 15,
        object_class_to_target_class: List[list] = None,
        object_n_verb_to_interaction: List[list] = None,

    ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.clip_head = model

        self.register_buffer("object_embedding",object_embedding)

        self.visual_output_dim = model.image_encoder.output_dim
        self.object_n_verb_to_interaction = np.asarray(
                                object_n_verb_to_interaction, dtype=float
                            )

        self.args = args

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.object_class_to_target_class = object_class_to_target_class

        self.num_classes = num_classes

        self.dataset = args.dataset
        self.hyper_lambda = args.hyper_lambda

        self.use_insadapter = args.use_insadapter
        self.tp = None
        self.reserve_indices = reserve_indices


        self.priors_initial_dim = self.visual_output_dim + 5
        self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.priors_downproj = MLP(self.priors_initial_dim, 128, args.adapt_dim, 3) # old 512+5


        self.query_proj = MLP(512, 128, 768, 2)

        if self.dataset == "vg":
            # [SGG compositional text]
            # Build a cheap ordered S-O condition from frozen VG object
            # text embeddings. The condition modulates the 50 learned
            # predicate features instead of re-encoding every literal
            # S-P-O sentence with the CLIP text transformer.
            self.triplet_pair_composer = MLP(
                self.visual_output_dim * 4,
                self.visual_output_dim,
                self.visual_output_dim * 2,
                2,
            )

            # Start close to the original predicate text features while
            # retaining a small, direction-sensitive S-O contribution.
            nn.init.normal_(
                self.triplet_pair_composer.layers[-1].weight,
                std=1e-3,
            )
            nn.init.zeros_(
                self.triplet_pair_composer.layers[-1].bias
            )

    def _reset_parameters(self):  ## xxx
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
    ) -> Tensor:

        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else self.hyper_lambda
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])

    def compute_sim_scores(self, region_props: List[dict], image, priors=None):
        device = image.tensors.device
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        # pairwise_tokens_collated = []
        all_logits = []

        # Encode the active predicate vocabulary once per forward pass.
        # [SGG compositional text]
        # VG reuses these 50 predicate features for every directed S-O
        # pair; only the inexpensive pair-conditioned composition happens
        # inside the image loop.
        text_features = None
        if self.args.use_prompt:
            if not self.training:
                if self.tp is None: # when evaluation, compute text embeds once.
                    prompts = self.clip_head.prompt_learner()
                    text_features = self.clip_head.text_encoder(prompts, self.clip_head.tokenized_prompts)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    self.tp = text_features
                else:
                    text_features = self.tp
            else:
                prompts = self.clip_head.prompt_learner()
                text_features = self.clip_head.text_encoder(prompts, self.clip_head.tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)



        # get updated HO tokens.
        for b_idx, props in enumerate(region_props):
            # local_features = features[b_idx]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            feats = props['feat']

            if self.dataset != "vg":
                is_human = labels == self.human_idx
                n_h = torch.sum(is_human)

                # Preserve the original LAIN ordering for HOI datasets.
                if not torch.all(labels[:n_h] == self.human_idx):
                    h_idx = torch.nonzero(is_human).squeeze(1)
                    o_idx = torch.nonzero(is_human == 0).squeeze(1)
                    perm = torch.cat([h_idx, o_idx])
                    boxes = boxes[perm]
                    scores = scores[perm]
                    labels = labels[perm]
                    feats = feats[perm]

            x_keep, y_keep = self.generate_pair_indices(labels)

            # Skip images that do not contain a valid relation pair.
            if len(x_keep) == 0:
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                continue

            if self.dataset == "vg":
                # [SGG compositional text]
                # Produce [num_pairs, num_predicates, text_dim] prototypes
                # without running the CLIP text transformer per triplet.
                pair_text_features = self.compose_vg_text_features(
                    labels[x_keep],
                    labels[y_keep],
                    text_features,
                )

            if self.args.use_hotoken:
                # mask for each HO tokens + CLS
                num_tokens = len(x_keep) + 1

                # Create covered_mask of size (num_tokens + 196) x (num_tokens + 196)
                mask = torch.zeros((num_tokens + 196, num_tokens + 196), dtype=torch.bool, device=device)
                mask[:num_tokens, :num_tokens] = ~torch.eye(num_tokens, dtype=torch.bool, device=device)
                mask[-197:, :-197] = True

                ho_tokens = self.query_proj(torch.cat([feats[x_keep],feats[y_keep]],dim=-1))


                la_masks = (boxes,x_keep,y_keep)
                global_feat, local_feat = self.clip_head.image_encoder(image.decompose()[0][b_idx:b_idx + 1],
                                                                       priors[b_idx] if self.args.use_prior else None,
                                                                       ho_tokens,
                                                                       mask,la_masks)


            global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
            global_feat = global_feat[:, :-1]


            if self.dataset == "vg":
                # [SGG compositional text]
                # Compare each visual pair with its 50 pair-conditioned
                # predicate prototypes.
                logits_text = torch.einsum(
                    "bpd,pcd->bpc",
                    global_feat,
                    pair_text_features,
                )
            else:
                logits_text = global_feat @ text_features.T
            logits = logits_text.squeeze(0) * self.logit_scale_text.exp()

            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            all_logits.append(logits)

        return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated

    def compose_vg_text_features(
        self,
        subject_labels: Tensor,
        object_labels: Tensor,
        predicate_features: Tensor,
    ) -> Tensor:
        """Compose ordered VG S-P-O text prototypes without literal text."""
        # [SGG compositional text]
        # Subject and object occupy different ordered positions. Their
        # difference and elementwise product add directional and pairwise
        # information before FiLM modulation of predicate text features.
        if self.dataset != "vg":
            raise RuntimeError(
                "VG text composition was called for a non-VG dataset"
            )
        if len(subject_labels) != len(object_labels):
            raise ValueError(
                "VG subject/object pair counts do not match"
            )
        if predicate_features.ndim != 2:
            raise ValueError(
                "Predicate features must have shape "
                "[num_predicates, text_dim]"
            )
        if predicate_features.shape[0] != self.num_classes:
            raise ValueError(
                "VG predicate feature count mismatch: "
                f"features={predicate_features.shape[0]}, "
                f"num_classes={self.num_classes}"
            )

        num_object_classes = self.object_embedding.shape[0]
        if len(subject_labels) > 0:
            minimum_label = min(
                subject_labels.min().item(),
                object_labels.min().item(),
            )
            maximum_label = max(
                subject_labels.max().item(),
                object_labels.max().item(),
            )
            if minimum_label < 0 or maximum_label >= num_object_classes:
                raise ValueError(
                    "VG pair label is outside the object vocabulary: "
                    f"min={minimum_label}, max={maximum_label}, "
                    f"num_objects={num_object_classes}"
                )

        object_vocabulary_features = F.normalize(
            self.object_embedding.to(predicate_features.dtype),
            dim=-1,
        )
        subject_features = object_vocabulary_features[
            subject_labels.long()
        ]
        object_features = object_vocabulary_features[
            object_labels.long()
        ]

        ordered_pair_features = torch.cat(
            [
                subject_features,
                object_features,
                subject_features - object_features,
                subject_features * object_features,
            ],
            dim=-1,
        )
        modulation = self.triplet_pair_composer(
            ordered_pair_features
        )
        scale, shift = modulation.chunk(2, dim=-1)

        # Bounded scaling prevents the newly initialized composer from
        # overwhelming the pretrained CLIP predicate representation.
        scale = torch.tanh(scale)
        triplet_features = (
            predicate_features.unsqueeze(0)
            * (1.0 + scale.unsqueeze(1))
            + shift.unsqueeze(1)
        )

        return F.normalize(
            triplet_features,
            dim=-1,
        )

    def generate_pair_indices(self, labels: Tensor):
        """Generate directed relation pairs for the active dataset."""
        n = len(labels)
        device = labels.device

        if n <= 1:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return empty, empty

        x, y = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device),
            indexing="ij",
        )

        if self.dataset == "vg":
            valid = x != y
        else:
            valid = torch.logical_and(
                x != y,
                labels[x] == self.human_idx,
            )

        return torch.nonzero(valid, as_tuple=True)

    def recover_boxes(self, boxes, size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(
        self,
        boxes_h,
        boxes_o,
        subject_labels,
        object_labels,
        targets,
    ):
        """Associate predicted relation pairs with ground-truth triplets."""
        n = boxes_h.shape[0]
        labels = torch.zeros(
            n,
            self.num_classes,
            device=boxes_h.device,
        )

        if not (
            len(boxes_h)
            == len(boxes_o)
            == len(subject_labels)
            == len(object_labels)
        ):
            raise ValueError(
                "Predicted relation-pair fields must have equal lengths"
            )

        gt_bx_h = self.recover_boxes(
            targets["boxes_h"],
            targets["size"],
        )
        gt_bx_o = self.recover_boxes(
            targets["boxes_o"],
            targets["size"],
        )

        if len(gt_bx_h) != len(gt_bx_o):
            raise ValueError(
                "Ground-truth subject/object box counts do not match"
            )

        # Both sides of a relation pair must overlap the corresponding
        # ground-truth subject and object boxes.
        subject_iou = box_iou(boxes_h, gt_bx_h)
        object_iou = box_iou(boxes_o, gt_bx_o)
        pair_matches = torch.min(
            subject_iou,
            object_iou,
        ) >= self.fg_iou_thresh

        if self.dataset == "vg":
            # [SGG class-aware matching]
            # VG permits every object class to be either the subject or
            # object, so box overlap alone is not a valid triplet match.
            required_fields = {
                "subject",
                "object",
                "verb",
            }
            missing_fields = required_fields.difference(targets)

            if missing_fields:
                raise KeyError(
                    "VG target is missing fields required for "
                    f"class-aware matching: {sorted(missing_fields)}"
                )

            gt_subject_labels = targets["subject"].long()
            gt_object_labels = targets["object"].long()
            gt_predicates = targets["verb"].long()
            num_gt_relations = len(gt_bx_h)

            if not (
                len(gt_subject_labels)
                == len(gt_object_labels)
                == len(gt_predicates)
                == num_gt_relations
            ):
                raise ValueError(
                    "VG GT triplet fields must have equal lengths: "
                    f"boxes={num_gt_relations}, "
                    f"subject={len(gt_subject_labels)}, "
                    f"object={len(gt_object_labels)}, "
                    f"verb={len(gt_predicates)}"
                )

            subject_class_matches = (
                subject_labels.long()[:, None]
                == gt_subject_labels[None, :]
            )
            object_class_matches = (
                object_labels.long()[:, None]
                == gt_object_labels[None, :]
            )
            pair_matches = (
                pair_matches
                & subject_class_matches
                & object_class_matches
            )
            target_predicates = gt_predicates

        else:
            # [Compatibility]
            # Preserve the original HICO-DET/V-COCO matching semantics.
            if self.num_classes in [24, 117, 407]:
                target_predicates = targets["labels"].long()
            else:
                target_predicates = targets["hoi"].long()

        predicted_pair_indices, gt_relation_indices = torch.nonzero(
            pair_matches,
            as_tuple=True,
        )

        if len(gt_relation_indices) > 0:
            selected_predicates = target_predicates[
                gt_relation_indices
            ]

            if (
                selected_predicates.min().item() < 0
                or selected_predicates.max().item() >= self.num_classes
            ):
                raise ValueError(
                    "Predicate label is outside the active vocabulary: "
                    f"min={selected_predicates.min().item()}, "
                    f"max={selected_predicates.max().item()}, "
                    f"num_classes={self.num_classes}"
                )

            # Repeated assignments preserve multiple predicates attached
            # to the same directed subject-object pair.
            labels[
                predicted_pair_indices,
                selected_predicates,
            ] = 1

        return labels

    def compute_interaction_loss(
        self,
        boxes,
        bh,
        bo,
        logits,
        prior,
        proposal_labels,
        targets,
    ):
        """Compute the LAIN relation classification loss."""
        batch_field_lengths = {
            len(boxes),
            len(bh),
            len(bo),
            len(logits),
            len(prior),
            len(proposal_labels),
            len(targets),
        }

        if len(batch_field_lengths) != 1:
            raise ValueError(
                "Interaction-loss batch fields have different lengths: "
                f"boxes={len(boxes)}, "
                f"bh={len(bh)}, "
                f"bo={len(bo)}, "
                f"logits={len(logits)}, "
                f"prior={len(prior)}, "
                f"proposal_labels={len(proposal_labels)}, "
                f"targets={len(targets)}"
            )

        labels = torch.cat([
            self.associate_with_ground_truth(
                bx[h],
                bx[o],
                proposal_label[h],
                proposal_label[o],
                target,
            )
            for (
                bx,
                h,
                o,
                proposal_label,
                target,
            ) in zip(
                boxes,
                bh,
                bo,
                proposal_labels,
                targets,
            )
        ])

        # The original LAIN prior remains downstream weighting/filtering;
        # it is not used as the VG ground-truth matching criterion.
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(
            prior,
            as_tuple=True,
        )
        logits = torch.cat(logits)
        logits = logits[x, y]
        prior = prior[x, y]
        labels = labels[x, y]

        n_p = torch.count_nonzero(labels)
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor(
                [n_p],
                device=logits.device,
            )
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss_with_logits(
            torch.log(
                prior
                / (
                    1
                    + torch.exp(-logits)
                    - prior
                )
                + 1e-8
            ),
            labels,
            reduction="sum",
            alpha=self.alpha,
            gamma=self.gamma,
        )

        return loss / max(n_p, 1)

    def prepare_region_proposals(self, results):
        region_props = []
        for res in results:
            # Explicit keys keep detector adapters independent of
            # dictionary insertion order.
            sc = res["scores"]
            lb = res["labels"]
            bx = res["boxes"]
            feat = res["feats"]

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            feat = feat[keep].view(-1,256)

            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            if self.dataset == "vg":
                # SGG allows every object class to act as either subject or
                # object. Select one generic proposal set without imposing
                # the original LAIN human/object quotas.
                if len(keep) < self.min_instances:
                    keep = sc.argsort(descending=True)[:self.min_instances]
                elif len(keep) > self.max_instances:
                    order = sc[keep].argsort(descending=True)
                    keep = keep[order[:self.max_instances]]
            else:
                # Original LAIN proposal selection for HICO-DET/V-COCO.
                is_human = lb == self.human_idx
                hum = torch.nonzero(is_human).squeeze(1)
                obj = torch.nonzero(is_human == 0).squeeze(1)
                n_human = is_human[keep].sum()
                n_object = len(keep) - n_human

                if n_human < self.min_instances:
                    keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                    keep_h = hum[keep_h]
                elif n_human > self.max_instances:
                    keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                    keep_h = hum[keep_h]
                else:
                    keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                    keep_h = keep[keep_h]

                if n_object < self.min_instances:
                    keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                    keep_o = obj[keep_o]
                elif n_object > self.max_instances:
                    keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                    keep_o = obj[keep_o]
                else:
                    keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                    keep_o = keep[keep_o]

                keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                feat=feat[keep]
            ))

        return region_props

    def build_vg_prior_features(self, region_props, image_size):
        """Build a sentinel-free 14x14 object-layout prior for VG."""
        if len(region_props) != len(image_size):
            raise ValueError(
                "VG prior batch mismatch: "
                f"proposals={len(region_props)}, "
                f"image_sizes={len(image_size)}"
            )

        grid_size = 14
        device = image_size.device
        dtype = self.object_embedding.dtype

        prior_features = torch.zeros(
            (
                len(region_props),
                grid_size,
                grid_size,
                self.priors_initial_dim,
            ),
            dtype=dtype,
            device=device,
        )

        image_height, image_width = image_size.unbind(-1)
        scale_factors = torch.stack(
            [
                image_width,
                image_height,
                image_width,
                image_height,
            ],
            dim=1,
        )

        for batch_index, props in enumerate(region_props):
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels'].long()

            if not (
                len(boxes) == len(scores) == len(labels)
            ):
                raise ValueError(
                    "VG proposal fields must have equal lengths"
                )

            if len(boxes) == 0:
                continue

            if labels.min().item() < 0 or labels.max().item() >= 150:
                raise ValueError(
                    "VG object labels must be in [0, 149], "
                    f"got [{labels.min().item()}, "
                    f"{labels.max().item()}]"
                )

            scaled_boxes = boxes * (
                grid_size
                / scale_factors[batch_index][None, :]
            )
            scaled_boxes = scaled_boxes.clamp(
                min=0,
                max=grid_size,
            )

            object_embeddings = self.object_embedding[
                labels
            ].to(dtype=dtype)

            proposal_features = torch.cat(
                [
                    scores.to(dtype=dtype).unsqueeze(-1),
                    scaled_boxes.to(dtype=dtype),
                    object_embeddings,
                ],
                dim=-1,
            )

            if proposal_features.shape[-1] != self.priors_initial_dim:
                raise ValueError(
                    "VG prior feature dimension mismatch: "
                    f"expected={self.priors_initial_dim}, "
                    f"actual={proposal_features.shape[-1]}"
                )

            # Draw low-score proposals first so a higher-score proposal
            # owns pixels in overlapping regions.
            draw_order = scores.argsort(descending=False)

            for proposal_index in draw_order.tolist():
                x1, y1, x2, y2 = scaled_boxes[
                    proposal_index
                ]

                x1 = int(torch.floor(x1).clamp(0, grid_size - 1).item())
                y1 = int(torch.floor(y1).clamp(0, grid_size - 1).item())
                x2 = int(torch.ceil(x2).clamp(1, grid_size).item())
                y2 = int(torch.ceil(y2).clamp(1, grid_size).item())

                x2 = max(x2, x1 + 1)
                y2 = max(y2, y1 + 1)

                prior_features[
                    batch_index,
                    y1:y2,
                    x1:x2,
                ] = proposal_features[proposal_index]

        return prior_features

    def get_prior(self, region_props, image_size): ##  for adapter module training

        if self.dataset == "vg":
            prior_features = self.build_vg_prior_features(
                region_props,
                image_size,
            )
            occupied = prior_features.abs().sum(
                dim=-1,
                keepdim=True,
            ) > 0

            priors = self.priors_downproj(
                prior_features
            )

            # Keep the background explicitly zero even when the MLP has
            # learned bias terms.
            return priors * occupied.to(priors.dtype)

        max_feat = self.priors_initial_dim
        max_length = max(rep['boxes'].shape[0] for rep in region_props)
        priors = torch.zeros((len(region_props),14,14), dtype=torch.float32, device=region_props[0]['boxes'].device)


        priors_dim = torch.zeros((len(region_props),14,14,max_feat), dtype=torch.float32, device=region_props[0]['boxes'].device)
        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        for b_idx, props in enumerate(region_props):
            boxes = props['boxes'] * (14 / scale_fct[b_idx][None,:])
            scores = props['scores']
            labels = props['labels']
            priors[b_idx] = len(boxes)


            boxes[:, 2:] += 0.5
            new_boxes = torch.round(boxes).long()

            for inb, nb in enumerate(new_boxes):
                x1_scaled, y1_scaled, x2_scaled, y2_scaled = nb
                #idx_mask = torch.zeros((14, 14), dtype=torch.bool).to(mask.device)
                priors[b_idx,y1_scaled:y2_scaled, x1_scaled:x2_scaled] = inb

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if n_h == 0 or n <= 1:
                print(n_h,n)
                # sys.exit()

            boxes = torch.cat([boxes,torch.tensor([[-1,-1,-1,-1.]]).to(boxes)],dim=0)
            labels = torch.cat([labels,torch.tensor([80]).to(boxes)],dim=0).long()
            scores = torch.cat([scores,torch.tensor([-1.]).to(boxes)],dim=0)

            object_embs = self.object_embedding[labels]

            sb = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
            sb_feat = sb[priors[b_idx].long()]
            obj_feat = object_embs[priors[b_idx].long()]

            prior_feat = torch.cat([sb_feat,obj_feat],dim=-1)
            priors_dim[b_idx] = prior_feat

        priors = self.priors_downproj(priors_dim)

        return priors


    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        batch_size = len(images)
        images_orig = [im[0].float() for im in images]
        images_clip = [im[1] for im in images]
        device = images_clip[0].device
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images_clip
        ], device=device)
        image_sizes_orig = torch.as_tensor([
            im.size()[-2:] for im in images_orig
            ], device=device)

        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)

        if self.dataset == "vg":
            if images_orig.mask is None:
                raise RuntimeError(
                    "EGTR detector requires a valid image padding mask"
                )

            detector_outputs = self.detector(
                pixel_values=images_orig.tensors,
                pixel_mask=(~images_orig.mask).long(),
            )

            results = {
                "pred_logits": detector_outputs.logits,
                "pred_boxes": detector_outputs.pred_boxes,
                "feats": detector_outputs.last_hidden_state,
            }
        else:
            # Original LAIN DETR path for HICO-DET and V-COCO.
            features, pos = self.detector.backbone(images_orig)
            src, mask = features[-1].decompose()

            hs, _ = self.detector.transformer(
                self.detector.input_proj(src),
                mask,
                self.detector.query_embed.weight,
                pos[-1],
            )

            outputs_class = self.detector.class_embed(hs)
            outputs_coord = self.detector.bbox_embed(hs).sigmoid()

            if self.dataset == "vcoco" and outputs_class.shape[-1] == 92:
                outputs_class = outputs_class[:, :, :, self.reserve_indices]
                assert outputs_class.shape[-1] == 81

            results = {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
                "feats": hs[-1],
            }

        # LAIN consumes boxes in the 224x224 CLIP-image coordinate space.
        results = self.postprocessor(results, image_sizes)
        region_props = self.prepare_region_proposals(results)


        priors = self.get_prior(region_props,image_sizes)

        # with amp.autocast(enabled=True):
        images_clip = nested_tensor_from_tensor_list(images_clip)

        logits, prior, bh, bo, objects = self.compute_sim_scores(region_props,images_clip,priors)
        boxes = [r['boxes'] for r in region_props]

        # [SGG class-aware matching]
        # Passing code: carry detector class IDs to the training matcher;
        # relation scoring itself is unchanged here.
        proposal_labels = [
            region_prop["labels"]
            for region_prop in region_props
        ]

        if self.training:
            interaction_loss = self.compute_interaction_loss(
                boxes,
                bh,
                bo,
                logits,
                prior,
                proposal_labels,
                targets,
            )

            loss_dict = dict(
                interaction_loss=interaction_loss
            )

            # [Incidental logging]
            # Debug runs intentionally skip wandb.init(), so only log when
            # an active run exists. This does not affect the loss tensor.
            if self.args.local_rank == 0 and wandb.run is not None:
                wandb.log(loss_dict)

            return loss_dict

        if len(logits) == 0:
            print(targets)
            return None

        detections = self.postprocessing(
            boxes,
            bh,
            bo,
            logits,
            prior,
            objects,
            proposal_labels,
            image_sizes,
        )
        return detections

    def postprocessing(
        self,
        boxes,
        bh,
        bo,
        logits,
        prior,
        objects,
        proposal_labels,
        image_sizes,
    ):
        n = [len(b) for b in bh]
        logits = torch.cat(logits)
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, prop_labels, size in zip(
                boxes, bh, bo, logits, prior, objects,
                proposal_labels, image_sizes,
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])

            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                # [SGG evaluation]
                # Passing fields: expose both endpoint classes so SGDet
                # evaluation can match complete S-P-O triplets. Existing
                # HICO consumers continue to use the unchanged objects key.
                subjects=prop_labels[h[x]],
                objects=obj[x], size=size
            ))

        return detections

@torch.no_grad()
def get_obj_text_emb(args, clip_model, obj_class_names):
    obj_text_inputs = torch.cat([tokenize(obj_text) for obj_text in obj_class_names])
    with torch.no_grad():
        obj_text_embedding = clip_model.encode_text(obj_text_inputs)
        object_embedding = obj_text_embedding
        # obj_text_embedding = obj_text_embedding[hoi_obj_list,:]
    return object_embedding


def build_detector(
    args,
    class_corr,
    object_n_verb_to_interaction,
    clip_model_path,
):
    if args.dataset == "vg":
        if not args.egtr_detector_dir:
            raise ValueError(
                "--egtr-detector-dir is required for VG"
            )
        # [SGG compositional text]
        # Learned predicate contexts remain part of VG composition.
        if not args.use_prompt:
            raise ValueError(
                "--use_prompt is required for VG compositional text"
            )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "Load the EGTR VG detector from "
                f"{args.egtr_detector_dir}"
            )

        detr = load_egtr_vg_detector(
            args.egtr_detector_dir
        )
        postprocessor = EgtrPostProcess()

    else:
        # Original LAIN detector construction for HOI datasets.
        num_classes = 80

        if (
            args.dataset == "vcoco"
            and "e632da11" in args.pretrained
        ):
            num_classes = 91

        backbone = build_backbone(args)
        transformer = build_transformer(args)

        detr = DETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
        )

        postprocessor = PostProcess()

        if os.path.exists(args.pretrained):
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(
                    "Load weights for the object detector from "
                    f"{args.pretrained}"
                )

            checkpoint = torch.load(
                args.pretrained,
                map_location="cpu",
                weights_only=False,
            )

            if "e632da11" in args.pretrained:
                detr.load_state_dict(checkpoint["model"])
            else:
                detr.load_state_dict(
                    checkpoint["model_state_dict"]
                )
    clip_state_dict = torch.load(clip_model_path, map_location="cpu", weights_only=False).state_dict()
    clip_model = build_model(state_dict=clip_state_dict, use_adapter=args.use_insadapter, adapter_pos=args.adapter_pos, args=args)

    if args.num_classes == 117:
        classnames = hico_verbs_sentence

    elif args.num_classes == 24:
        classnames = vcoco_verbs_sentence

    elif args.num_classes == 600:
        classnames = list(
            hico_text_label.hico_text_label.values()
        )

    elif args.dataset == 'vg' and args.num_classes == 50:
        from utils.vg_list import (
            VG150_PREDICATES,
            get_vg_object_names,
        )

        # [SGG compositional text]
        # Learn one context per bare predicate. Ordered subject/object
        # information is injected later by the lightweight pair composer.
        classnames = list(VG150_PREDICATES)

    else:
        raise NotImplementedError(
            'Unsupported dataset/class configuration: '
            f'dataset={args.dataset}, '
            f'num_classes={args.num_classes}'
        )

    if len(classnames) != args.num_classes:
        raise ValueError(
            'Relation classname count mismatch: '
            f'{len(classnames)} classnames versus '
            f'num_classes={args.num_classes}.'
        )

    model = CustomCLIP(args, classnames=classnames, clip_model=clip_model)

    if args.dataset == "vg":
        vg_object_names = get_vg_object_names(
            args.data_root
        )
        obj_class_names = [
            f"a photo of a {name}"
            for name in vg_object_names
        ]
    else:
        obj_class_names = [
            obj[1]
            for obj in hico_text_label.hico_obj_text_label
        ]

    object_embedding = get_obj_text_emb(args, clip_model=clip_model, obj_class_names=obj_class_names)
    object_embedding = object_embedding.clone().detach()

    expected_object_classes = 150 if args.dataset == "vg" else len(obj_class_names)
    if object_embedding.shape[0] != expected_object_classes:
        raise ValueError(
            "Object embedding count mismatch: "
            f"expected={expected_object_classes}, "
            f"actual={object_embedding.shape[0]}"
        )

    detector = LAIN(
        args,
        detr,
        postprocessor,
        model,
        object_embedding,
        human_idx=args.human_idx,
        num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        object_class_to_target_class=class_corr,
        object_n_verb_to_interaction=object_n_verb_to_interaction,
    )

    return detector
