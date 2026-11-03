# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any, Dict, List

import torch
from transformers.image_utils import load_image

from angelslim.utils import rank0_print

__all__ = [
    "process_token_dict_to_mappings",
    "convert_sharegpt_data",
    "convert_ultrachat_data",
    "DataCollatorWithPadding",
    "VLMDataCollatorWithPadding",
    "VLMHunyuanDataCollatorWithPadding",
    "AudioDataCollatorWithPadding",
    "CosyVoice3DataCollatorWithPadding",
    "build_image_processor_kwargs",
    "build_qwen_vl_mm_token_type_ids",
]


def build_qwen_vl_mm_token_type_ids(
    input_ids: torch.Tensor,
    processor,
) -> torch.Tensor:
    """Image-pad region between <|vision_start|> and <|vision_end|> (Qwen3-VL)."""
    tokenizer = getattr(processor, "tokenizer", processor)
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    mm = torch.zeros_like(input_ids)
    if vision_start is None or vision_end is None:
        return mm
    for batch_idx in range(input_ids.shape[0]):
        seq = input_ids[batch_idx]
        starts = (seq == vision_start).nonzero(as_tuple=False).view(-1)
        ends = (seq == vision_end).nonzero(as_tuple=False).view(-1)
        pair_count = min(len(starts), len(ends))
        for idx in range(pair_count):
            start = int(starts[idx].item())
            end = int(ends[idx].item())
            if end > start + 1:
                mm[batch_idx, start + 1 : end] = 1
    return mm


def build_image_processor_kwargs(image_processor, max_pixels=None, min_pixels=None):
    """
    convert max_pixels/min_pixels to the format required by the specific image_processor.
      - Qwen2.5-VL: directly use max_pixels / min_pixels
      - Qwen3-VL:   convert to size={"longest_edge": max_pixels, "shortest_edge": min_pixels}

    Args:
        image_processor: model's image_processor instance
        max_pixels: maximum pixels (total area), None means no limit
        min_pixels: minimum pixels (total area), None means no limit

    Returns:
        dict: can be directly passed to image_processor(...)
    """
    if max_pixels is None and min_pixels is None:
        return {}

    processor_class = type(image_processor).__name__
    # Qwen3-VL uses size={"longest_edge": ..., "shortest_edge": ...}
    if "Qwen3" in processor_class:
        size = {}
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
        return {"size": size}
    else:
        # Qwen2.5-VL's accept max_pixels and min_pixels
        kwargs = {}
        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
        if min_pixels is not None:
            kwargs["min_pixels"] = min_pixels
        return kwargs


def convert_sharegpt_data(row, dataset_column="conversations"):
    converted_messages = []

    role_mapping = {"human": "user", "gpt": "assistant"}
    messages = row[dataset_column]
    for message in messages:
        converted_messages.append(
            {"role": role_mapping[message["from"]], "content": message["value"]}
        )

    return {"conversations": converted_messages, "id": row["id"]}


def convert_ultrachat_data(row, dataset_column="messages"):
    converted_messages = []

    messages = row[dataset_column]
    for message in messages:
        converted_messages.append({"role": message["role"], "content": message["content"]})
    return {"conversations": converted_messages, "id": row["prompt_id"]}


def process_token_dict_to_mappings(
    token_dict,
    draft_vocab_size: int,
    target_vocab_size: int,
):
    """
    Process token_dict to create d2t and t2d mappings, with optional caching.

    Args:
        token_dict: A Counter object mapping token ids to their frequencies.
        draft_vocab_size: The size of the draft vocabulary.
        target_vocab_size: The size of the target vocabulary.

    Returns:
        A tuple containing:
            - d2t: A tensor mapping draft token ids to target token ids.
            - t2d: A tensor mapping target token ids to draft token ids.
    """
    if len(token_dict) < draft_vocab_size:
        existing_tokens = set(token_dict.keys())
        missing_tokens = set(range(draft_vocab_size)) - existing_tokens
        for token in missing_tokens:
            token_dict[token] = 0
            if len(token_dict) >= draft_vocab_size:
                break
    rank0_print(f"Added missing tokens to reach draft vocab size: {draft_vocab_size}")
    rank0_print(f"Total tokens after addition: {len(token_dict)}")
    total_frequency = sum(token_dict.values())
    top_N = token_dict.most_common(draft_vocab_size)
    top_N_frequency_sum = sum(freq for key, freq in top_N)

    if total_frequency == 0:
        rank0_print("Warning: Total token frequency is zero. All tokens will have zero ratio.")
        top_N_ratio = 0.0
    else:
        top_N_ratio = top_N_frequency_sum / total_frequency

    rank0_print(f"top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")
    used_tokens = [key for key, freq in top_N]
    used_tokens.sort()

    used_set = set(used_tokens)
    d2t = torch.tensor(
        [used_tokens[i] - i for i in range(len(used_tokens))],
        dtype=torch.int64,  # must match register_buffer dtype in Eagle3LlamaForCausalLM
    )
    t2d = torch.tensor(
        [i in used_set for i in range(target_vocab_size)],
        dtype=torch.bool,  # must match register_buffer dtype in Eagle3LlamaForCausalLM
    )

    assert d2t.shape == (draft_vocab_size,), f"d2t shape {d2t.shape} != ({draft_vocab_size},)"
    assert t2d.shape == (target_vocab_size,), f"t2d shape {t2d.shape} != ({target_vocab_size},)"
    assert (
        t2d.sum().item() == draft_vocab_size
    ), f"t2d has {t2d.sum().item()} True entries, expected {draft_vocab_size}"

    return d2t, t2d


def paddingtensor(intensors, N):
    B, n, S = intensors.shape
    # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
    padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
    outtensors = torch.cat((intensors, padding_tensor), dim=1)
    return outtensors


def paddingtensor2D(intensors, N):
    B, n = intensors.shape
    padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
    outtensors = torch.cat((intensors, padding_tensor), dim=1)
    return outtensors


def paddingtensor3D_CBN(tensor_list):
    if all(tensor is None for tensor in tensor_list):
        return None
    N = max(tensor.shape[-1] for tensor in tensor_list if tensor is not None)
    out_tensor_list = []
    for tensor in tensor_list:
        c, b, n = tensor.shape
        outtensor = torch.zeros(c, b, N, dtype=tensor_list[0].dtype)
        if tensor is not None:
            outtensor[:, :, :n] = tensor
        out_tensor_list.append(outtensor)
    return torch.cat(out_tensor_list, dim=1)


def paddingtensor3D_BCN(tensor_list):
    if all(tensor is None for tensor in tensor_list):
        return None
    N = max(tensor.shape[-1] for tensor in tensor_list if tensor is not None)
    out_tensor_list = []
    for tensor in tensor_list:
        b, c, n = tensor.shape
        outtensor = torch.zeros(b, c, N, dtype=tensor_list[0].dtype)
        if tensor is not None:
            outtensor[:, :, :n] = tensor
        out_tensor_list.append(outtensor)
    return torch.cat(out_tensor_list, dim=0)


def paddingtensor3D_BHW(tensor_list, pad_value=0):
    if all(tensor is None for tensor in tensor_list):
        return None
    max_h = max(tensor.shape[-2] for tensor in tensor_list if tensor is not None)
    max_w = max(tensor.shape[-1] for tensor in tensor_list if tensor is not None)
    out_tensor_list = []
    for tensor in tensor_list:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        b, h, w = tensor.shape
        outtensor = torch.full(
            (b, max_h, max_w),
            pad_value,
            dtype=tensor.dtype,
        )
        if tensor is not None:
            outtensor[:, :h, :w] = tensor
        out_tensor_list.append(outtensor)
    return torch.cat(out_tensor_list)


class DataCollatorWithPadding:

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [paddingtensor2D(item["attention_mask"], max_length) for item in features]
        )
        batch_loss_mask = torch.cat(
            [paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_states": None,
            "target_hiddens": None,
        }

        # Handle hidden_states and target_hiddens independently
        if all("hidden_states" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [paddingtensor(item["hidden_states"], max_length) for item in features]
            )

        if all("target_hiddens" in item for item in features):
            batch["target_hiddens"] = torch.cat(
                [paddingtensor(item["target_hiddens"], max_length) for item in features]
            )
        return batch


class VLMDataCollatorWithPadding:

    def __init__(self, processor=None, image_processor_kwargs=None):
        """
        Args:
            processor: VLM processor (e.g. AutoProcessor for qwen3_vl).
                       When provided, image_paths in features will be decoded
                       on-the-fly to pixel_values (used in online training).
            image_processor_kwargs: Additional kwargs passed to image_processor,
                       e.g. {"max_pixels": 1003520, "min_pixels": 200704}.
        """
        self.processor = processor
        if image_processor_kwargs is None:
            image_processor_kwargs = {}
        max_pixels = image_processor_kwargs.get("max_pixels", None)
        min_pixels = image_processor_kwargs.get("min_pixels", "1024")
        if (
            processor is not None
            and (max_pixels is not None or min_pixels is not None)
            and hasattr(processor, "image_processor")
        ):
            self._resolved_image_processor_kwargs = build_image_processor_kwargs(
                processor.image_processor, max_pixels, min_pixels
            )
        else:
            self._resolved_image_processor_kwargs = {}
        rank0_print(f"_resolved_image_processor_kwargs: {self._resolved_image_processor_kwargs}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        vision_encodings = None
        raw_images = None
        if self.processor is not None and "image_paths" in features[0]:
            prepared_features = []
            vision_encodings = []
            raw_images = []
            image_token_id = getattr(self.processor, "image_token_id", None)
            if image_token_id is None and hasattr(self.processor, "tokenizer"):
                image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                    "<|image|>"
                )
            for item in features:
                prepared = dict(item)
                image_paths = json.loads(item["image_paths"])
                vision_enc = None
                if image_paths:
                    images = [load_image(path) for path in image_paths]
                    raw_images.append(images)
                    if hasattr(self.processor, "image_processor"):
                        vision_enc = self.processor.image_processor(
                            images=images,
                            return_tensors="pt",
                            **self._resolved_image_processor_kwargs,
                        )
                    else:
                        vision_enc = self.processor(
                            images=images,
                            return_tensors="pt",
                            **self._resolved_image_processor_kwargs,
                        )

                    # Gemma-4 emits one textual image marker but its vision
                    # tower returns one model token per soft visual token.
                    soft_counts = vision_enc.get("num_soft_tokens_per_image")
                    if soft_counts is not None and image_token_id is not None:
                        counts = [int(value) for value in soft_counts.view(-1)]
                        input_ids = prepared["input_ids"].view(-1)
                        repeats = torch.ones_like(input_ids)
                        image_positions = (input_ids == int(image_token_id)).nonzero(
                            as_tuple=False
                        ).view(-1)
                        if len(image_positions) != len(counts):
                            raise ValueError(
                                "Gemma image token/image count mismatch: "
                                f"{len(image_positions)} tokens for {len(counts)} images"
                            )
                        for position, count in zip(image_positions, counts):
                            repeats[position] = count
                        for key in ("input_ids", "attention_mask", "loss_mask"):
                            source = prepared[key].view(-1)
                            prepared[key] = torch.repeat_interleave(
                                source,
                                repeats,
                            ).view(1, -1)
                else:
                    raw_images.append([])
                prepared_features.append(prepared)
                vision_encodings.append(vision_enc)
            features = prepared_features

        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [paddingtensor2D(item["attention_mask"], max_length) for item in features]
        )
        batch_loss_mask = torch.cat(
            [paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_states": None,
            "target_hiddens": None,
            "inputs_embeds": None,
            "position_ids": None,
        }
        if raw_images is not None:
            batch["raw_images"] = raw_images

        # Online training: decode image_paths -> pixel_values on-the-fly
        if self.processor is not None and "image_paths" in features[0]:
            all_pixel_values, all_image_grid_thw, all_image_position_ids = [], [], []
            all_pixel_values_videos, all_video_grid_thw = [], []
            assert vision_encodings is not None
            for vision_enc in vision_encodings:
                if vision_enc is not None:
                    all_pixel_values.append(vision_enc["pixel_values"])
                    if "image_grid_thw" in vision_enc:
                        all_image_grid_thw.append(vision_enc["image_grid_thw"])
                    if "image_position_ids" in vision_enc:
                        all_image_position_ids.append(
                            vision_enc["image_position_ids"]
                        )
                    if "pixel_values_videos" in vision_enc:
                        all_pixel_values_videos.append(vision_enc["pixel_values_videos"])
                    if "video_grid_thw" in vision_enc:
                        all_video_grid_thw.append(vision_enc["video_grid_thw"])
            if all_pixel_values:
                batch["pixel_values"] = paddingtensor3D_BHW(all_pixel_values)
            if all_image_grid_thw:
                batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
            if all_image_position_ids:
                batch["image_position_ids"] = paddingtensor3D_BHW(
                    all_image_position_ids,
                    pad_value=-1,
                )
            if all_pixel_values_videos:
                batch["pixel_values_videos"] = paddingtensor3D_BHW(all_pixel_values_videos)
            if all_video_grid_thw:
                batch["video_grid_thw"] = torch.cat(all_video_grid_thw, dim=0)
            if batch.get("image_grid_thw") is not None and self.processor is not None:
                if "Qwen3" in type(self.processor).__name__:
                    batch["mm_token_type_ids"] = build_qwen_vl_mm_token_type_ids(
                        batch["input_ids"],
                        self.processor,
                    )
        else:
            if "pixel_values" in features[0]:
                batch["pixel_values"] = paddingtensor3D_BHW(
                    [item["pixel_values"] for item in features]
                )
            if "pixel_values_videos" in features[0]:
                batch["pixel_values_videos"] = paddingtensor3D_BHW(
                    [item["pixel_values_videos"] for item in features]
                )
            if all(
                "image_grid_thw" in item and item["image_grid_thw"] is not None
                for item in features
            ):
                batch["image_grid_thw"] = torch.cat(
                    [item["image_grid_thw"] for item in features], dim=0
                )
            if all(
                "image_position_ids" in item
                and item["image_position_ids"] is not None
                for item in features
            ):
                batch["image_position_ids"] = paddingtensor3D_BHW(
                    [item["image_position_ids"] for item in features],
                    pad_value=-1,
                )
            if all(
                "video_grid_thw" in item and item["video_grid_thw"] is not None
                for item in features
            ):
                batch["video_grid_thw"] = torch.cat(
                    [item["video_grid_thw"] for item in features], dim=0
                )

        # Check if both hidden_states and target_hiddens exist in all features
        if all("hidden_states" in item and "target_hiddens" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [paddingtensor(item["hidden_states"], max_length) for item in features]
            )
            batch["target_hiddens"] = torch.cat(
                [paddingtensor(item["target_hiddens"], max_length) for item in features]
            )
        if all("inputs_embeds" in item and item["inputs_embeds"] is not None for item in features):
            batch["inputs_embeds"] = torch.cat(
                [paddingtensor(item["inputs_embeds"], max_length) for item in features]
            )
        if all("position_ids" in item and item["position_ids"] is not None for item in features):
            batch["position_ids"] = paddingtensor3D_CBN(
                [item["position_ids"] for item in features]
            )

        return batch


class VLMHunyuanDataCollatorWithPadding:

    def __init__(self, processor=None, image_processor_kwargs=None):
        """
        Args:
            processor: VLM processor (e.g. AutoProcessor for hunyuan_vl).
                       When provided, image_paths in features will be decoded
                       on-the-fly to pixel_values (used in online training).
            image_processor_kwargs: Additional kwargs passed to image_processor,
                       e.g. {"max_pixels": 1003520, "min_pixels": 200704}.
        """
        self.processor = processor
        if image_processor_kwargs is None:
            image_processor_kwargs = {}
        max_pixels = image_processor_kwargs.get("max_pixels", None)
        min_pixels = image_processor_kwargs.get("min_pixels", "1024")
        if (
            processor is not None
            and (max_pixels is not None or min_pixels is not None)
            and hasattr(processor, "image_processor")
        ):
            self._resolved_image_processor_kwargs = build_image_processor_kwargs(
                processor.image_processor, max_pixels, min_pixels
            )
        else:
            self._resolved_image_processor_kwargs = {}
        rank0_print(f"_resolved_image_processor_kwargs: {self._resolved_image_processor_kwargs}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [paddingtensor2D(item["attention_mask"], max_length) for item in features]
        )
        batch_loss_mask = torch.cat(
            [paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_states": None,
            "target_hiddens": None,
            "inputs_embeds": None,
            "position_ids": None,
            "input_position_ids": None,
        }

        # Online training: decode image_paths -> pixel_values on-the-fly
        if self.processor is not None and "image_paths" in features[0]:
            all_pixel_values, all_image_grid_thw = [], []
            for item in features:
                image_paths = json.loads(item["image_paths"])
                if image_paths:
                    images = [load_image(p) for p in image_paths]
                    if hasattr(self.processor, "image_processor"):
                        vision_enc = self.processor.image_processor(
                            images=images,
                            return_tensors="pt",
                            **self._resolved_image_processor_kwargs,
                        )
                    else:
                        vision_enc = self.processor(
                            images=images,
                            return_tensors="pt",
                            **self._resolved_image_processor_kwargs,
                        )
                    all_pixel_values.append(vision_enc["pixel_values"])
                    if "image_grid_thw" in vision_enc:
                        all_image_grid_thw.append(vision_enc["image_grid_thw"])
            if all_pixel_values:
                batch["pixel_values"] = paddingtensor3D_BHW(all_pixel_values)
            if all_image_grid_thw:
                batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
        else:
            if "pixel_values" in features[0]:
                batch["pixel_values"] = paddingtensor3D_BHW(
                    [item["pixel_values"] for item in features]
                )
            if all(
                "image_grid_thw" in item and item["image_grid_thw"] is not None
                for item in features
            ):
                batch["image_grid_thw"] = torch.cat(
                    [item["image_grid_thw"] for item in features], dim=0
                )

        # Check if both hidden_states and target_hiddens exist in all features
        if all("hidden_states" in item and "target_hiddens" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [paddingtensor(item["hidden_states"], max_length) for item in features]
            )
            batch["target_hiddens"] = torch.cat(
                [paddingtensor(item["target_hiddens"], max_length) for item in features]
            )
        if all("inputs_embeds" in item and item["inputs_embeds"] is not None for item in features):
            batch["inputs_embeds"] = torch.cat(
                [paddingtensor(item["inputs_embeds"], max_length) for item in features]
            )
        if all(
            "input_position_ids" in item and item["input_position_ids"] is not None
            for item in features
        ):
            batch["input_position_ids"] = paddingtensor3D_BCN(
                [item["input_position_ids"] for item in features]
            )
        if all("position_ids" in item and item["position_ids"] is not None for item in features):
            batch["position_ids"] = torch.cat(
                [paddingtensor2D(item["position_ids"], max_length) for item in features]
            )
        return batch


class AudioDataCollatorWithPadding:

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [paddingtensor2D(item["attention_mask"], max_length) for item in features]
        )
        batch_loss_mask = torch.cat(
            [paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "feature_attention_mask": None,
            "input_features": None,
            "hidden_states": None,
            "target_hiddens": None,
            "inputs_embeds": None,
            "position_ids": None,
        }

        # Check if both hidden_states and target_hiddens exist in all features
        if all("hidden_states" in item and "target_hiddens" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [paddingtensor(item["hidden_states"], max_length) for item in features]
            )
            batch["target_hiddens"] = torch.cat(
                [paddingtensor(item["target_hiddens"], max_length) for item in features]
            )
        if all("inputs_embeds" in item and item["inputs_embeds"] is not None for item in features):
            batch["inputs_embeds"] = torch.cat(
                [paddingtensor(item["inputs_embeds"], max_length) for item in features]
            )
        if all("position_ids" in item and item["position_ids"] is not None for item in features):
            batch["position_ids"] = torch.cat(
                [paddingtensor2D(item["position_ids"], max_length) for item in features]
            )
        if all(
            "feature_attention_mask" in item and item["feature_attention_mask"] is not None
            for item in features
        ):
            batch["feature_attention_mask"] = torch.cat(
                [(item["feature_attention_mask"]) for item in features]
            )
        if all(
            "input_features" in item and item["input_features"] is not None for item in features
        ):
            batch["input_features"] = torch.cat([(item["input_features"]) for item in features])
        return batch


class CosyVoice3DataCollatorWithPadding:

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["text"].shape[-1] for item in features)
        batch_text_tokens = torch.cat(
            [paddingtensor2D(item["text"].unsqueeze(0), max_length) for item in features]
        )
        max_length = max(item["speech_token"].shape[-1] for item in features)
        batch_speech_tokens = torch.cat(
            [paddingtensor2D(item["speech_token"].unsqueeze(0), max_length) for item in features]
        )
        max_length = max(item["prompt_text"].shape[-1] for item in features)
        batch_prompt_text = torch.cat(
            [paddingtensor2D(item["prompt_text"].unsqueeze(0), max_length) for item in features]
        )
        max_length = max(item["prompt_speech_token"].shape[-1] for item in features)
        batch_prompt_speech_tokens = torch.cat(
            [
                paddingtensor2D(item["prompt_speech_token"].unsqueeze(0), max_length)
                for item in features
            ]
        )
        batch_text_token_lens = torch.stack([item["text_len"] for item in features])
        batch_speech_token_lens = torch.stack([item["speech_token_len"] for item in features])
        batch_prompt_text_lens = torch.stack([item["prompt_text_len"] for item in features])
        batch_prompt_speech_token_lens = torch.stack(
            [item["prompt_speech_token_len"] for item in features]
        )

        batch = {
            "text": batch_text_tokens,
            "text_len": batch_text_token_lens,
            "speech_token": batch_speech_tokens,
            "speech_token_len": batch_speech_token_lens,
            "prompt_speech_token": batch_prompt_speech_tokens,
            "prompt_speech_token_len": batch_prompt_speech_token_lens,
            "prompt_text": batch_prompt_text,
            "prompt_text_len": batch_prompt_text_lens,
            "hidden_states": None,
            "target_hiddens": None,
        }

        # Check if both hidden_states and target_hiddens exist in all features
        if all("hidden_states" in item and "target_hiddens" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [paddingtensor(item["hidden_states"], max_length) for item in features]
            )
            batch["target_hiddens"] = torch.cat(
                [paddingtensor(item["target_hiddens"], max_length) for item in features]
            )
        return batch
