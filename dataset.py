"""MARCO image and target features for frozen-backbone classification.

Store source images, not model-specific embeddings. The model owns resizing,
normalization, and frozen feature extraction. The split field preserves the
official training and validation assignments within one dataset.
"""

import zygo


# Preserve MARCO's numeric labels from the TFRecords and extracted class folders.
CLASS_NAMES = ("clear", "crystals", "other", "precipitate")


@zygo.features
class MarcoFeatures(zygo.Features):
    image: zygo.Image
    split: str
    label: zygo.ClassLabel = zygo.ClassLabel(*CLASS_NAMES)
