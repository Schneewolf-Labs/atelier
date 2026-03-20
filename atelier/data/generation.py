import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GenerationDataset(Dataset):
    """Dataset for text-to-image training.

    For DPO: expects columns (prompt, chosen, rejected) — preferred/dispreferred images.
    For SFT: expects columns (prompt, image) — single target image.

    Supports dual tokenizers (e.g. SDXL's two CLIP tokenizers) via tokenizer_2.
    """

    def __init__(self, dataset, tokenizer=None, tokenizer_2=None, image_size=1024, max_samples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.image_size = image_size
        self.is_dpo = "rejected" in dataset.column_names

        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        result = {}

        # Tokenize prompt
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                item["prompt"],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"] = text_inputs.input_ids.squeeze(0)
            result["attention_mask"] = text_inputs.attention_mask.squeeze(0)

        if self.tokenizer_2 is not None:
            text_inputs_2 = self.tokenizer_2(
                item["prompt"],
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids_2"] = text_inputs_2.input_ids.squeeze(0)
            result["attention_mask_2"] = text_inputs_2.attention_mask.squeeze(0)

        # Process images
        chosen_key = "chosen" if self.is_dpo else "image"
        result["chosen_image"] = _image_to_tensor(item[chosen_key], self.image_size)

        if self.is_dpo:
            result["rejected_image"] = _image_to_tensor(item["rejected"], self.image_size)

        return result


class GenerationCollator:
    """Default collator for generation datasets — stacks all tensors."""

    def __call__(self, examples):
        batch = {}
        keys = examples[0].keys()
        for key in keys:
            values = [ex[key] for ex in examples]
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            else:
                batch[key] = values
        return batch


def _image_to_tensor(image, image_size):
    """Convert a PIL image to a normalized [C, H, W] tensor in [-1, 1]."""
    if not isinstance(image, Image.Image):
        if isinstance(image, str):
            image = Image.open(image)
        else:
            image = Image.fromarray(np.uint8(image))

    image = image.convert("RGB").resize((image_size, image_size), Image.LANCZOS)
    tensor = torch.from_numpy(np.array(image).astype(np.float32))
    tensor = tensor / 127.5 - 1.0
    tensor = tensor.permute(2, 0, 1)
    return tensor
