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
"""

import json
import os

import h5py
import numpy as np
import torch
from PIL import Image


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
            split_array = file["split"][:]
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
        split_code = 0 if split == "train" else 2

        self.image_index = []
        self.filenames = []

        for image_index, split_value in enumerate(split_array):
            if split_value != split_code:
                continue

            # Images without relationships cannot contribute to SGG training
            # or relation recall evaluation.
            if self.img_to_first_rel[image_index] < 0:
                continue

            self.image_index.append(image_index)
            self.filenames.append(
                all_filenames[image_index]
            )

        print(
            f"[VG] split={split}, "
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
        ]
        labels_local = self.labels[
            first_box:last_box + 1
        ].squeeze(1)

        num_objects = len(boxes_local)

        boxes_xyxy = self._xcywh_1024_to_xyxy(
            boxes_local,
            image_width,
            image_height,
        )

        first_relation = int(
            self.img_to_first_rel[image_index]
        )
        last_relation = int(
            self.img_to_last_rel[image_index]
        )

        relationships = self.relationships[
            first_relation:last_relation + 1
        ]
        predicates = self.predicates[
            first_relation:last_relation + 1
        ].squeeze(1)

        subject_boxes = []
        object_boxes = []
        predicate_classes = []
        subject_classes = []
        object_classes = []

        for relation, predicate in zip(
            relationships,
            predicates,
        ):
            # VG-SGG.h5 stores global indices into the complete box array.
            # Convert them to image-local indices before indexing boxes_local.
            subject_index = int(relation[0]) - first_box
            object_index = int(relation[1]) - first_box

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