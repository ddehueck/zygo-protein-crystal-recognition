# protein-crystal-recognition
Train a computer vision model to detect protein crystals, then deploy it as a reproducible workflow for screening microscopy images

# Fetch the Data

Run `uv run scripts/fetch_marco.py` to download the official MARCO JPEG-encoded train and validation TFRecord shards and the label CSVs into `data/marco/`.

The shards are saved in `data/marco/train-jpg/` and `data/marco/test-jpg/`. They contain JPEG images in TFRecord format, not extracted image files.

The download is large and may require tens of GB of disk space. Re-running the command skips completed files and resumes interrupted `.part` downloads.

# Prepare Images

Run `uv run scripts/prepare_marco.py` after downloading to extract the JPEG images without re-encoding them:

```text
data/marco/images/
  train/
    clear/
    crystals/
    other/
    precipitate/
  validation/
    clear/
    crystals/
    other/
    precipitate/
```

MARCO's `test` split becomes `validation`, preserving the official split. Filenames use the original image IDs. Use `--split train` or `--split test` to process just one split. Re-running skips images whose bytes already match and reports conflicting files instead of overwriting them.

This layout works with `torchvision.datasets.ImageFolder` and Hugging Face Datasets' `imagefolder` loader. For Hugging Face, install `datasets` in your training environment and load it with:

```python
from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="data/marco/images")
```

Folder-based loaders infer class indices alphabetically. Save the loader's class mapping for inference rather than assuming MARCO's numeric IDs. The TFRecord files remain in place, so allow additional disk space for the extracted images.

# Build Zygo Datasets

After extracting the images, run these plain Python scripts. Configure paths, sample sizes, shard sizes, and the random seed by editing the constants at the top of each script. There are no command-line options.

```bash
uv run scripts/build_marco_dataset.py
uv run scripts/sample_marco_dataset.py
```

The first script uses `zygo.Dataset.builder` and the `MarcoFeatures` schema in `dataset.py` to build one sharded Parquet dataset at `data/marco/zygo/`. Each row contains the original JPEG bytes, its MARCO class label, and a `split` string of `"train"` or `"validation"`.

The second script samples without replacement using seed 42 into one dataset at `data/marco/zygo-small/`, preserving each row's official split:

- `split="train"`: 1,000 images per class, 4,000 total.
- `split="validation"`: 1,250 total, with 313 clear, 313 crystals, 312 other, and 312 precipitate images.

The balanced validation subset has a different class distribution from the full MARCO validation set. Use the full validation set when measuring performance under the original distribution.

Both scripts refuse to overwrite nonempty output directories. The sampler fails if any class has fewer images than requested. Neither script downloads data or pretrained model weights. Conversion adds another copy of the encoded images on disk, and sampling retains the JPEG encoding without re-encoding.

Open the dataset once and filter by the row-level split:

```python
import zygo
from dataset import MarcoFeatures

dataset = zygo.Dataset.open("data/marco/zygo-small", features=MarcoFeatures)
train = dataset.where(split="train")
validation = dataset.where(split="validation")
```

# Train the Classifier

Run the Zygo training entry point over a frozen pretrained DINOv2 Small backbone using timm:

```bash
uv run train.py
```

Edit `DATASET_PATH` and `PARAMS` in `train.py` to configure training. The script defaults to `data/marco/zygo-small`, 10 epochs, batch size 8, and 224×224 inputs. Only the linear classification head is trained. The backbone stays in evaluation mode with gradients disabled. Device selection automatically prefers CUDA, then Apple Silicon MPS, then CPU. Set `device="cpu"` in `PARAMS` to override it.

Set `image_size=None` to use the model's pretrained input resolution, which is 518×518 for the default DINOv2 model. The script's 224×224 setting uses less memory and compute. Input sizes must be positive multiples of the model's 14-pixel patch size. Normalization and resizing come from timm's model configuration, with training augmentation and deterministic validation transforms.

The first run downloads pretrained weights via timm, normally from Hugging Face. It does not upload your images. Each run creates a unique persistent store under `zygo/models/` and prints its location. The store contains:

- `params.json`: the validated training parameters.
- `config.json`: training options, class mapping, and preprocessing.
- `metrics.jsonl`: per-epoch loss, accuracy, macro-F1, per-class recall, and confusion matrices.
- `best.pt`: the classification head from the epoch with the highest validation macro-F1, along with its model name, class mapping, and preprocessing settings.

Macro-F1 gives each class equal weight when selecting a checkpoint. The checkpoint does not duplicate the frozen DINOv2 weights. For inference, recreate the same pretrained timm backbone at the saved input size, attach a linear head with the saved class count, and load `classifier_state_dict` into that head.

# Training and Prediction API

`main.py` defines the Zygo model's training, loading, and single-image inference hooks. Business logic lives in `src/model.py`, `src/data.py`, and `src/training.py`. Training streams the Parquet dataset in-process with bounded-buffer shuffling and filters its row-level `train` and `validation` splits. MARCO's numeric label mapping is preserved.

Call Zygo directly to train and predict:

```python
import zygo
from PIL import Image
from main import TrainingParams, app

store = zygo.train(
    model=app,
    dataset="data/marco/zygo-small",
    params=TrainingParams(epochs=10, batch_size=8, image_size=224),
)
classifier = app.run_load(store)
with Image.open("microscopy.jpg") as image:
    prediction = app.run_infer(classifier, image)
print(prediction)
```

Loading recreates the pretrained backbone, so its weights must be cached locally or downloadable. Predictions contain the MARCO label, class name, and softmax confidence. Validation metrics use the split that selects the checkpoint, not an independent held-out test set.

For direct training without a Zygo model store, use `src.train_model(dataset, output_dir, **params)`. It accepts a Zygo dataset and returns the best checkpoint path. The output directory must not already exist.

Run offline regression tests with `uv run python -m unittest discover -s tests`. The end-to-end test uses a tiny test backbone and temporary Parquet data without downloading pretrained weights.
