# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, List, Optional, Tuple, Union

import lmdb
from torchvision.datasets import VisionDataset

from .decoders import TargetDecoder, ImageDataDecoder
from .extended import ExtendedVisionDataset

class SimpleLMDBDataset(ExtendedVisionDataset):
    def __init__(
        self,
        *,
        split: "ImageNet.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.lmdb_env_imgs = lmdb.open(
            os.path.join(root, split),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.lmdb_txn_imgs = self.lmdb_env_imgs.begin()


    def get_image_data(self, index: int) -> bytes:
        img = self.lmdb_txn_imgs.get(("tmp" + str(index)).encode("utf-8"))
        return img

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = 0
        # target = self.get_target(index)
        # target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return self.lmdb_env_imgs.stat()["entries"]
