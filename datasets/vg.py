"""
Visual Genome VG150 dataset loader for LAIN.

Converts Visual Genome SGG annotations into an HICODet-compatible
target dictionary while preserving complete subject-predicate-object
triplet information.

Return format:
    ((PIL image, target dict), filename)

Target fields:
    boxes_h: [N, 4] subject boxes in xyxy pixel coordinates
    boxes_o: [N, 4] object boxes in xyxy pixel coordinates
    verb:    [N]    predicate indices (0-based, 0-49)
    subject: [N]    subject class indices (0-based, 0-149)
    object:  [N]    object class indices (0-based, 0-149)
    hoi:     [N]    compatibility alias of verb

Verified:
    - HDF5/image filename alignment: 108,073 images
    - boxes_1024 format: [xc, yc, w, h]
    - HDF5 relationships contain global box indices
    - predicate and object labels are converted from 1-based to 0-based

OvR behavior when args.vg_ovr is enabled:
    - use the official OvSGTR split_GLIPunseen image split
    - retain only base-predicate relations in training targets
    - exclude training images that contain no base relation
    - retain all base and novel relations for test evaluation

Official OvSGTR protocol when args.vg_ovsgtr_protocol is enabled:
    - reserve the first 5,000 valid training images as validation
    - remove training relations whose subject/object boxes do not overlap
    - merge same-class entities with IoU greater than 0.9
    - sample one predicate for each duplicate directed pair during training

The official protocol is opt-in so earlier LAIN-SGG experiments remain
reproducible without changing their data contract.
"""

import json
import os

import h5py
import numpy as np
import torch
from PIL import Image

from utils.vg_ov_split import resolve_ovsgtr_predicate_split


# Standard corrupted VG images excluded by common VG150 preprocessing.
CORRUPTED_IMS = {
    "1592.jpg",
    "1722.jpg",
    "4616.jpg",
    "4617.jpg",
}


class VGDataset:
    """Visual Genome 150 dataset for scene graph generation."""

    def __init__(
        self,
        root,
        split="train",
        num_relations=50,
        args=None,
    ):
        if split not in {"train", "test"}:
            raise ValueError(
                f"Unknown VG split '{split}'. Expected 'train' or 'test'."
            )

        if num_relations != 50:
            raise ValueError(
                f"VG150 requires 50 predicate classes, got {num_relations}."
            )

        self.root = root
        self.split = split
        self.args = args

        # [VG OvR mode]
        # Keep the fully-supervised VG path unchanged unless explicitly
        # enabled by --vg-ovr.
        self.vg_ovr = bool(
            getattr(args, "vg_ovr", False)
        )
        self.vg_ovsgtr_protocol = bool(
            getattr(args, "vg_ovsgtr_protocol", False)
        )
        self.vg_ovsgtr_num_val_images = int(
            getattr(args, "vg_ovsgtr_num_val_images", 5000)
        )

        if self.vg_ovsgtr_protocol and not self.vg_ovr:
            raise ValueError(
                "--vg-ovsgtr-protocol requires --vg-ovr."
            )

        if self.vg_ovsgtr_num_val_images < 0:
            raise ValueError(
                "--vg-ovsgtr-num-val-images must be non-negative."
            )
        self.num_object_cls = 150
        self.num_relation_cls = num_relations

        stanford_root = os.path.join(
            root,
            "vg_data",
            "stanford_filtered",
        )
        h5_path = os.path.join(stanford_root, "VG-SGG.h5")
        dict_path = os.path.join(
            stanford_root,
            "VG-SGG-dicts.json",
        )
        image_data_path = os.path.join(
            stanford_root,
            "image_data.json",
        )

        for path in (h5_path, dict_path, image_data_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Required VG file was not found: {path}"
                )

        # VG150 class dictionaries use 1-based string indices.
        with open(dict_path, encoding="utf-8") as file:
            class_dict = json.load(file)

        self.idx_to_label = class_dict["idx_to_label"]
        self.idx_to_predicate = class_dict["idx_to_predicate"]

        if len(self.idx_to_label) != self.num_object_cls:
            raise ValueError(
                "Unexpected number of VG object classes: "
                f"{len(self.idx_to_label)}"
            )

        if len(self.idx_to_predicate) != self.num_relation_cls:
            raise ValueError(
                "Unexpected number of VG predicates: "
                f"{len(self.idx_to_predicate)}"
            )

        # [Official OvSGTR predicate split]
        # Resolve names against the active VG dictionary instead of assuming
        # that predicate indices are always stored in the same order.
        self.base_predicate_indices = tuple()
        self.novel_predicate_indices = tuple()

        if self.vg_ovr:
            predicate_names = [
                self.idx_to_predicate[str(index + 1)]
                for index in range(self.num_relation_cls)
            ]
            predicate_split = resolve_ovsgtr_predicate_split(
                predicate_names
            )
            self.base_predicate_indices = predicate_split["base"]
            self.novel_predicate_indices = predicate_split["novel"]

        self.base_predicate_index_array = np.asarray(
            self.base_predicate_indices,
            dtype=np.int64,
        )

        # Build image paths in the same order used by VG-SGG.h5.
        with open(image_data_path, encoding="utf-8") as file:
            image_data = json.load(file)

        all_filenames = []

        for item in image_data:
            image_url = item["url"]
            basename = os.path.basename(image_url)

            if basename in CORRUPTED_IMS:
                continue

            if "VG_100K_2" in image_url:
                folder = "VG_100K_2"
            else:
                folder = "VG_100K"

            all_filenames.append(
                os.path.join(folder, basename)
            )

        # Load HDF5 annotations into memory. This avoids retaining an open
        # h5py handle when DataLoader workers are created.
        with h5py.File(h5_path, "r") as file:
            # [Official OvSGTR image split]
            # The fully-supervised baseline keeps the original split.
            # OvR uses split_GLIPunseen to exclude test images exposed
            # during visual-backbone pretraining.
            self.split_key = (
                "split_GLIPunseen"
                if self.vg_ovr
                else "split"
            )

            if self.split_key not in file:
                raise KeyError(
                    "VG HDF5 is missing the requested split key: "
                    f"{self.split_key}"
                )

            split_array = file[self.split_key][:]
            self.boxes = file["boxes_1024"][:]
            self.labels = file["labels"][:]
            self.relationships = file["relationships"][:]
            self.predicates = file["predicates"][:]
            self.img_to_first_box = file["img_to_first_box"][:]
            self.img_to_last_box = file["img_to_last_box"][:]
            self.img_to_first_rel = file["img_to_first_rel"][:]
            self.img_to_last_rel = file["img_to_last_rel"][:]

        if len(all_filenames) != len(split_array):
            raise ValueError(
                "VG filename/HDF5 alignment failed: "
                f"{len(all_filenames)} filenames versus "
                f"{len(split_array)} HDF5 entries."
            )

        # Stanford VG150 split codes:
        #     0 = train
        #     2 = test
        # split_GLIPunseen additionally uses -2 for excluded test images.
        split_code = 0 if split == "train" else 2

        # Build the valid-image list before the OvSGTR validation-prefix
        # slicing. This order matches the official loader.
        candidate_indices = [
            image_index
            for image_index, split_value in enumerate(split_array)
            if split_value == split_code
            and self.img_to_first_box[image_index] >= 0
            and self.img_to_first_rel[image_index] >= 0
        ]

        if self.vg_ovsgtr_protocol and split == "train":
            candidate_indices = candidate_indices[
                self.vg_ovsgtr_num_val_images:
            ]

        self.image_index = []
        self.filenames = []

        for image_index in candidate_indices:

            # [Base-only OvR training images]
            # Official OvR training excludes images that contain no base
            # relation. Test images retain both base and novel relations.
            if self.vg_ovr and split == "train":
                first_relation = int(
                    self.img_to_first_rel[image_index]
                )
                last_relation = int(
                    self.img_to_last_rel[image_index]
                )

                image_relationships = self.relationships[
                    first_relation:last_relation + 1
                ].astype(np.int64, copy=True)
                image_predicates = self.predicates[
                    first_relation:last_relation + 1
                ].reshape(-1).astype(np.int64) - 1

                keep_relation = np.isin(
                    image_predicates,
                    self.base_predicate_index_array,
                )

                # Official OvSGTR additionally removes non-overlapping
                # subject-object annotations from the training set.
                if self.vg_ovsgtr_protocol:
                    first_box = int(
                        self.img_to_first_box[image_index]
                    )
                    last_box = int(
                        self.img_to_last_box[image_index]
                    )
                    image_relationships[:, :2] -= first_box
                    image_boxes = self._xcywh_1024_to_xyxy(
                        self.boxes[first_box:last_box + 1],
                        1024,
                        1024,
                    )
                    keep_relation &= self._relationship_overlap_mask(
                        image_boxes,
                        image_relationships,
                    )

                if not keep_relation.any():
                    continue

            self.image_index.append(image_index)
            self.filenames.append(
                all_filenames[image_index]
            )

        # [Incidental logging]
        # This reports which split is active but does not affect samples.
        print(
            f"[VG] split={split}, "
            f"split_key={self.split_key}, "
            f"ovr={self.vg_ovr}, "
            f"ovsgtr_protocol={self.vg_ovsgtr_protocol}, "
            f"{len(self.image_index)} images loaded "
            f"(objects={self.num_object_cls}, "
            f"relations={self.num_relation_cls})"
        )

    def __len__(self):
        return len(self.image_index)

    @staticmethod
    def _xcywh_1024_to_xyxy(
        boxes_1024,
        image_width,
        image_height,
    ):
        """
        Convert VG boxes from 1024-normalized [xc, yc, w, h] to
        image-pixel [x1, y1, x2, y2].
        """

        scale = max(image_width, image_height) / 1024.0

        center_x = boxes_1024[:, 0]
        center_y = boxes_1024[:, 1]
        width = boxes_1024[:, 2]
        height = boxes_1024[:, 3]

        x1 = (center_x - width / 2.0) * scale
        y1 = (center_y - height / 2.0) * scale
        x2 = (center_x + width / 2.0) * scale
        y2 = (center_y + height / 2.0) * scale

        boxes_xyxy = np.stack(
            [x1, y1, x2, y2],
            axis=1,
        )

        boxes_xyxy[:, 0::2] = np.clip(
            boxes_xyxy[:, 0::2],
            0,
            image_width,
        )
        boxes_xyxy[:, 1::2] = np.clip(
            boxes_xyxy[:, 1::2],
            0,
            image_height,
        )

        return boxes_xyxy

    @staticmethod
    def _box_iou_matrix(boxes):
        """Return pairwise IoU using the same continuous-box convention."""

        boxes = np.asarray(boxes, dtype=np.float32)
        top_left = np.maximum(
            boxes[:, None, :2],
            boxes[None, :, :2],
        )
        bottom_right = np.minimum(
            boxes[:, None, 2:],
            boxes[None, :, 2:],
        )
        intersection_size = np.clip(
            bottom_right - top_left,
            0,
            None,
        )
        intersection = (
            intersection_size[..., 0]
            * intersection_size[..., 1]
        )
        box_size = np.clip(
            boxes[:, 2:] - boxes[:, :2],
            0,
            None,
        )
        area = box_size[:, 0] * box_size[:, 1]
        union = area[:, None] + area[None, :] - intersection
        return np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )

    @classmethod
    def _relationship_overlap_mask(cls, boxes, relationships):
        """Select directed relations whose endpoint boxes overlap."""

        if len(relationships) == 0:
            return np.zeros((0,), dtype=bool)

        num_boxes = len(boxes)
        valid = (
            (relationships[:, 0] >= 0)
            & (relationships[:, 0] < num_boxes)
            & (relationships[:, 1] >= 0)
            & (relationships[:, 1] < num_boxes)
        )
        result = np.zeros(len(relationships), dtype=bool)
        if valid.any():
            iou = cls._box_iou_matrix(boxes)
            relation_ids = np.nonzero(valid)[0]
            result[relation_ids] = iou[
                relationships[relation_ids, 0],
                relationships[relation_ids, 1],
            ] > 0
        return result

    @classmethod
    def _merge_duplicate_entities(
        cls,
        boxes,
        labels,
        relationships,
    ):
        """Match official OvSGTR same-class IoU>0.9 entity merging."""

        if len(boxes) == 0:
            return boxes, labels, relationships

        # ``boxes`` is retained in VG's stored [xc, yc, w, h] form so it
        # can later be converted using the real image dimensions.  IoU,
        # however, must be computed from [x1, y1, x2, y2] coordinates.
        boxes_xyxy = cls._xcywh_1024_to_xyxy(
            boxes,
            1024,
            1024,
        )
        entity_match = cls._box_iou_matrix(boxes_xyxy) > 0.9
        entity_match &= labels[:, None] == labels[None, :]

        keep_entity_ids = []
        old_to_new = {}

        # This intentionally mirrors OvSGTR's greedy row-order merge.
        for entity_id in range(len(boxes)):
            matched_ids = np.nonzero(entity_match[entity_id])[0]
            if len(matched_ids) == 0:
                continue

            new_id = len(keep_entity_ids)
            keep_entity_ids.append(entity_id)
            for matched_id in matched_ids:
                old_to_new[int(matched_id)] = new_id
            entity_match[:, matched_ids] = False

        remapped_relationships = relationships.copy()
        for relation in remapped_relationships:
            relation[0] = old_to_new[int(relation[0])]
            relation[1] = old_to_new[int(relation[1])]

        keep_entity_ids = np.asarray(
            keep_entity_ids,
            dtype=np.int64,
        )
        return (
            boxes[keep_entity_ids],
            labels[keep_entity_ids],
            remapped_relationships,
        )

    @staticmethod
    def _sample_one_predicate_per_pair(
        relationships,
        predicates,
    ):
        """Apply official train-time duplicate-pair predicate sampling."""

        pair_to_predicates = {}
        for relation, predicate in zip(relationships, predicates):
            pair = (int(relation[0]), int(relation[1]))
            pair_to_predicates.setdefault(pair, []).append(int(predicate))

        sampled_relations = []
        sampled_predicates = []
        for pair, pair_predicates in pair_to_predicates.items():
            sampled_relations.append(pair)
            sampled_predicates.append(
                np.random.choice(pair_predicates)
            )

        return (
            np.asarray(sampled_relations, dtype=np.int64).reshape(-1, 2),
            np.asarray(sampled_predicates, dtype=np.int64),
        )

    def __getitem__(self, index):
        image_index = self.image_index[index]
        filename = self.filenames[index]
        image_path = os.path.join(self.root, filename)

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        first_box = int(
            self.img_to_first_box[image_index]
        )
        last_box = int(
            self.img_to_last_box[image_index]
        )

        if first_box < 0 or last_box < first_box:
            raise RuntimeError(
                f"Invalid VG box range for {filename}: "
                f"{first_box} to {last_box}"
            )

        boxes_local = self.boxes[
            first_box:last_box + 1
        ].copy()
        labels_local = self.labels[
            first_box:last_box + 1
        ].squeeze(1).copy()

        first_relation = int(
            self.img_to_first_rel[image_index]
        )
        last_relation = int(
            self.img_to_last_rel[image_index]
        )

        relationships = self.relationships[
            first_relation:last_relation + 1
        ].astype(np.int64, copy=True)
        relationships[:, :2] -= first_box
        predicates = self.predicates[
            first_relation:last_relation + 1
        ].squeeze(1).astype(np.int64, copy=True)

        # [Base-only OvR train annotations]
        # Novel relations remain available in the test target, but are not
        # exposed as positive labels during OvR training.
        if self.vg_ovr and self.split == "train":
            predicate_indices = (
                predicates.astype(np.int64) - 1
            )
            keep_base = np.isin(
                predicate_indices,
                self.base_predicate_index_array,
            )

            relationships = relationships[keep_base]
            predicates = predicates[keep_base]

            if len(predicates) == 0:
                raise RuntimeError(
                    "OvR train image contains no base relation after "
                    "dataset initialization filtering."
                )

        if self.vg_ovsgtr_protocol:
            # The official code caps GT entities at 100 before merging.
            # The audited VG split currently never exceeds this limit, but
            # retaining it protects the exact contract for other copies.
            if len(boxes_local) > 100:
                selected = np.random.choice(
                    len(boxes_local),
                    100,
                    replace=False,
                )
                old_to_new = {
                    int(old_id): new_id
                    for new_id, old_id in enumerate(selected)
                }
                keep_relation = np.asarray(
                    [
                        int(relation[0]) in old_to_new
                        and int(relation[1]) in old_to_new
                        for relation in relationships
                    ],
                    dtype=bool,
                )
                relationships = relationships[keep_relation]
                predicates = predicates[keep_relation]
                for relation in relationships:
                    relation[0] = old_to_new[int(relation[0])]
                    relation[1] = old_to_new[int(relation[1])]
                boxes_local = boxes_local[selected]
                labels_local = labels_local[selected]

            boxes_for_protocol = self._xcywh_1024_to_xyxy(
                boxes_local,
                1024,
                1024,
            )

            if self.split == "train":
                keep_overlap = self._relationship_overlap_mask(
                    boxes_for_protocol,
                    relationships,
                )
                relationships = relationships[keep_overlap]
                predicates = predicates[keep_overlap]

            boxes_local, labels_local, relationships = (
                self._merge_duplicate_entities(
                    boxes_local,
                    labels_local,
                    relationships,
                )
            )

            if self.split == "train":
                relationships, predicates = (
                    self._sample_one_predicate_per_pair(
                        relationships,
                        predicates,
                    )
                )

        num_objects = len(boxes_local)
        boxes_xyxy = self._xcywh_1024_to_xyxy(
            boxes_local,
            image_width,
            image_height,
        )

        subject_boxes = []
        object_boxes = []
        predicate_classes = []
        subject_classes = []
        object_classes = []

        for relation, predicate in zip(
            relationships,
            predicates,
        ):
            subject_index = int(relation[0])
            object_index = int(relation[1])

            valid_subject = (
                0 <= subject_index < num_objects
            )
            valid_object = (
                0 <= object_index < num_objects
            )

            if not (valid_subject and valid_object):
                continue

            predicate_index = int(predicate) - 1
            subject_class = (
                int(labels_local[subject_index]) - 1
            )
            object_class = (
                int(labels_local[object_index]) - 1
            )

            if not 0 <= predicate_index < self.num_relation_cls:
                continue

            if not 0 <= subject_class < self.num_object_cls:
                continue

            if not 0 <= object_class < self.num_object_cls:
                continue

            subject_boxes.append(
                boxes_xyxy[subject_index]
            )
            object_boxes.append(
                boxes_xyxy[object_index]
            )
            predicate_classes.append(
                predicate_index
            )
            subject_classes.append(
                subject_class
            )
            object_classes.append(
                object_class
            )

        num_relations = len(subject_boxes)

        if num_relations == 0:
            subject_boxes_tensor = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )
            object_boxes_tensor = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )
            predicate_tensor = torch.zeros(
                (0,),
                dtype=torch.int64,
            )
            subject_tensor = torch.zeros(
                (0,),
                dtype=torch.int64,
            )
            object_tensor = torch.zeros(
                (0,),
                dtype=torch.int64,
            )
        else:
            subject_boxes_tensor = torch.as_tensor(
                np.stack(subject_boxes),
                dtype=torch.float32,
            )
            object_boxes_tensor = torch.as_tensor(
                np.stack(object_boxes),
                dtype=torch.float32,
            )
            predicate_tensor = torch.as_tensor(
                predicate_classes,
                dtype=torch.int64,
            )
            subject_tensor = torch.as_tensor(
                subject_classes,
                dtype=torch.int64,
            )
            object_tensor = torch.as_tensor(
                object_classes,
                dtype=torch.int64,
            )

        target = {
            # Keep the original LAIN/HICODet field names as a passing
            # interface. In VG, boxes_h represents a general subject box.
            "boxes_h": subject_boxes_tensor,
            "boxes_o": object_boxes_tensor,
            "verb": predicate_tensor,
            "subject": subject_tensor,
            "object": object_tensor,
            # LAIN currently expects an HOI label in some execution paths.
            # VG has no separate HOI class, so predicate is retained as a
            # temporary compatibility alias.
            "hoi": predicate_tensor.clone(),
        }

        expected_length = len(predicate_tensor)

        for field_name in (
            "boxes_h",
            "boxes_o",
            "subject",
            "object",
            "hoi",
        ):
            if len(target[field_name]) != expected_length:
                raise RuntimeError(
                    f"Misaligned VG target field '{field_name}' "
                    f"for {filename}: "
                    f"{len(target[field_name])} versus "
                    f"{expected_length} relations."
                )

        return (image, target), filename

    @property
    def objects(self):
        """Return the 150 VG object names in 0-based class order."""

        return [
            self.idx_to_label[str(index + 1)]
            for index in range(self.num_object_cls)
        ]

    @property
    def verbs(self):
        """Return the 50 VG predicates in 0-based class order."""

        return [
            self.idx_to_predicate[str(index + 1)]
            for index in range(self.num_relation_cls)
        ]

    @property
    def object_class_to_target_class(self):
        """
        Compatibility mapping used by the original LAIN/HICODet code.

        The minimal VG baseline permits every predicate for every object
        class. Dataset-statistics restrictions can be introduced later as
        a separate ablation.
        """

        return [
            list(range(self.num_relation_cls))
            for _ in range(self.num_object_cls)
        ]

    @property
    def object_n_verb_to_interaction(self):
        """
        Compatibility placeholder for HICODet's object-verb-to-HOI map.

        VG does not define a separate HOI interaction index.
        """

        return np.full(
            (
                self.num_object_cls,
                self.num_relation_cls,
            ),
            None,
            dtype=object,
        ).tolist()
