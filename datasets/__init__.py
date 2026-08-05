from torch.utils.data import Dataset

from datasets.vcoco import VCOCO
from datasets.hicodet import HICODet
from datasets.vg import VGDataset
import datasets.transform as T
import os

import pocket
import pocket.ops
from PIL import Image

import numpy as np
import torch

from utils.hico_text_label import hico_unseen_index


def custom_collate(batch):
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets


class DataFactory(Dataset):
    def __init__(
        self,
        name,
        partition,
        data_root,
        clip_model_name,
        zero_shot=False,
        zs_type="rare_first",
        num_classes=600,
        args=None,
    ):
        if name not in ["hicodet", "vcoco", "vg"]:
            raise ValueError(f"Unknown dataset '{name}'.")

        if name == "vg" and num_classes != 50:
            raise ValueError(
                "VG150 requires 50 predicate classes, "
                f"but num_classes={num_classes}."
            )

        if clip_model_name not in [
            "ViT-L/14@336px",
            "ViT-B/16",
        ]:
            raise ValueError(
                f"Unknown CLIP model '{clip_model_name}'."
            )

        self.clip_model_name = clip_model_name

        if self.clip_model_name == "ViT-B/16":
            self.clip_input_resolution = 224
        else:
            self.clip_input_resolution = 336

        if name == "vg":
            if partition not in ["train", "test"]:
                raise ValueError(
                    f"Unknown VG partition '{partition}'."
                )

            self.dataset = VGDataset(
                root=data_root,
                split=partition,
                num_relations=num_classes,
                args=args,
            )

        elif name == "hicodet":
            if partition not in [
                "train2015",
                "test2015",
            ]:
                raise ValueError(
                    "Unknown HICO-DET partition "
                    f"'{partition}'."
                )

            self.dataset = HICODet(
                root=os.path.join(
                    data_root,
                    "hico_20160224_det/images",
                    partition,
                ),
                anno_file=os.path.join(
                    data_root,
                    f"instances_{partition}.json",
                ),
                target_transform=pocket.ops.ToTensor(
                    input_format="dict"
                ),
                args=args,
            )

        else:
            if partition not in [
                "train",
                "val",
                "trainval",
                "test",
            ]:
                raise ValueError(
                    f"Unknown V-COCO partition '{partition}'."
                )

            image_dir = {
                "train": "mscoco2014/train2014",
                "val": "mscoco2014/train2014",
                "trainval": "mscoco2014/train2014",
                "test": "mscoco2014/val2014",
            }

            self.dataset = VCOCO(
                root=os.path.join(
                    data_root,
                    image_dir[partition],
                ),
                anno_file=os.path.join(
                    data_root,
                    f"instances_vcoco_{partition}.json",
                ),
                target_transform=pocket.ops.ToTensor(
                    input_format="dict"
                ),
            )

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

        scales = [
            480,
            512,
            544,
            576,
            608,
            640,
            672,
            704,
            736,
            768,
            800,
        ]

        if partition.startswith("train"):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(
                    0.4,
                    0.4,
                    0.4,
                ),
                T.RandomSelect(
                    T.RandomResize(
                        scales,
                        max_size=1333,
                    ),
                    T.Compose([
                        T.RandomResize(
                            [400, 500, 600]
                        ),
                        T.RandomSizeCrop(
                            384,
                            600,
                        ),
                        T.RandomResize(
                            scales,
                            max_size=1333,
                        ),
                    ]),
                ),
            ])
        else:
            self.transforms = T.Compose([
                T.RandomResize(
                    [800],
                    max_size=1333,
                ),
            ])

        self.clip_transforms = T.Compose([
            T.IResize([
                self.clip_input_resolution,
                self.clip_input_resolution,
            ]),
        ])

        self.partition = partition
        self.name = name
        self.count = 0
        self.zero_shot = zero_shot

        if (
            self.name == "hicodet"
            and self.zero_shot
            and self.partition == "train2015"
        ):
            self.zs_type = zs_type
            self.filtered_hoi_idx = hico_unseen_index[
                self.zs_type
            ]

        self.keep = list(range(len(self.dataset)))

        if (
            self.name == "hicodet"
            and self.zero_shot
            and self.partition == "train2015"
        ):
            self.zs_keep = []
            self.remain_hoi_idx = [
                index
                for index in np.arange(600)
                if index not in self.filtered_hoi_idx
            ]

            cache_path = (
                f"datasets/zs_{args.zs_type}_"
                f"{self.partition}_idx.pkl"
            )

            if os.path.exists(cache_path):
                import pickle

                with open(cache_path, "rb") as file:
                    self.zs_keep = pickle.load(file)

                print(f"{cache_path} is loaded")

            else:
                for index in self.keep:
                    (_, target), _ = self.dataset[index]

                    target_hoi = {
                        hoi.item()
                        for hoi in target["hoi"]
                    }
                    mutual_hoi = (
                        set(self.remain_hoi_idx)
                        & target_hoi
                    )

                    if mutual_hoi:
                        self.zs_keep.append(index)

                import pickle

                with open(cache_path, "wb") as file:
                    pickle.dump(
                        self.zs_keep,
                        file,
                    )

            self.keep = self.zs_keep
            self.dataset.zs_object_to_target = [
                []
                for _ in range(
                    self.dataset.num_object_cls
                )
            ]

            if num_classes == 600:
                for correlation in self.dataset.class_corr:
                    if (
                        correlation[0]
                        not in self.filtered_hoi_idx
                    ):
                        self.dataset.zs_object_to_target[
                            correlation[1]
                        ].append(correlation[0])
            else:
                for correlation in self.dataset.class_corr:
                    if (
                        correlation[0]
                        not in self.filtered_hoi_idx
                    ):
                        self.dataset.zs_object_to_target[
                            correlation[1]
                        ].append(correlation[2])

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, index):
        (image, target), filename = self.dataset[
            self.keep[index]
        ]

        if (
            self.name == "hicodet"
            and self.zero_shot
            and self.partition == "train2015"
        ):
            boxes_h = []
            boxes_o = []
            hoi_labels = []
            object_labels = []
            verb_labels = []

            for relation_index, hoi in enumerate(
                target["hoi"]
            ):
                if hoi in self.filtered_hoi_idx:
                    continue

                boxes_h.append(
                    target["boxes_h"][relation_index]
                )
                boxes_o.append(
                    target["boxes_o"][relation_index]
                )
                hoi_labels.append(
                    target["hoi"][relation_index]
                )
                object_labels.append(
                    target["object"][relation_index]
                )
                verb_labels.append(
                    target["verb"][relation_index]
                )

            target["boxes_h"] = torch.stack(boxes_h)
            target["boxes_o"] = torch.stack(boxes_o)
            target["hoi"] = torch.stack(hoi_labels)
            target["object"] = torch.stack(
                object_labels
            )
            target["verb"] = torch.stack(verb_labels)

        width, height = image.size
        target["orig_size"] = torch.tensor([
            height,
            width,
        ])

        if self.name == "vg":
            # VG boxes are already zero-based xyxy pixels.
            target["labels"] = target["verb"]

        elif self.name == "hicodet":
            target["labels"] = target["verb"]

            # HICO top-left coordinates are one-based.
            target["boxes_h"][:, :2] -= 1
            target["boxes_o"][:, :2] -= 1

        else:
            target["labels"] = target["actions"]
            target["object"] = target.pop("objects")
            # TODO: add target["hoi"] for V-COCO if needed.

        image, target = self.transforms(
            image,
            target,
        )

        image_clip, target = self.clip_transforms(
            image,
            target,
        )

        image, _ = self.normalize(
            image,
            None,
        )
        image_clip, target = self.normalize(
            image_clip,
            target,
        )

        target["filename"] = filename

        if self.name == "vg":
            filename_stem = os.path.splitext(
                os.path.basename(filename)
            )[0]
            target["filename_num"] = torch.tensor(
                int(filename_stem)
            )
        else:
            target["filename_num"] = torch.tensor(
                int(
                    filename
                    .split(".")[0]
                    .split("_")[-1]
                )
            )

        return (image, image_clip), target