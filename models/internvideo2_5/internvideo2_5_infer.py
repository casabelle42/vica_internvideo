#!/usr/bin/env python3
"""Offline InternVideo2.5 GUI inference on NVIDIA Thor.

Usage:
  python3 internvideo2_5_infer.py VIDEO_PATH MAX_FRAMES QUESTION

Video requests are capped at 120 frames and rounded down to a multiple
of the model's temporal grouping size. Images use one image token.
"""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = '/opt/models/InternVideo2_5_Chat_8B'
TOKENIZER_PATH = '/opt/models/internvideo_tokenizer_test'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
MAX_VIDEO_FRAMES = 120


def transform_image(size):
    return T.Compose([
        T.Lambda(lambda image: image.convert('RGB')),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def read_video(path, frame_count, size):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total <= 0:
            raise RuntimeError(f'Video contains no readable frames: {path}')
        indices = np.linspace(0, total - 1, frame_count, dtype=int)
        convert = transform_image(size)
        tensors = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Could not decode frame {index} in {path}')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(convert(Image.fromarray(rgb)))
        print(f'Video FPS: {fps:.2f}; total frames: {total}; sampled: {len(tensors)}',
              file=sys.stderr, flush=True)
        return torch.stack(tensors), len(tensors)
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description='InternVideo2.5 GUI positional inference')
    parser.add_argument('input_path', type=Path)
    parser.add_argument('max_frames', type=int, help='Requested number of uniformly sampled video frames')
    parser.add_argument('question', help='Natural-language prompt')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()
    if not args.input_path.is_file():
        parser.error(f'Input not found: {args.input_path}')
    if args.max_frames < 1 or args.max_new_tokens < 1:
        parser.error('Frame count and max-new-tokens must be positive')

    print('Loading tokenizer and model...', file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation='eager').eval().cuda()

    if args.input_path.suffix.lower() in IMAGE_EXTENSIONS:
        with Image.open(args.input_path) as image:
            pixels = transform_image(448)(image).unsqueeze(0)
        count = 1
    else:
        group_size = int(model.local_num_frames)
        if group_size < 1:
            raise RuntimeError(f'Invalid model.local_num_frames: {group_size}')
        capture = cv2.VideoCapture(str(args.input_path))
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open video: {args.input_path}')
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        requested = min(args.max_frames, MAX_VIDEO_FRAMES, total_frames)
        adjusted = (requested // group_size) * group_size
        if adjusted < group_size:
            raise ValueError(
                f'Video has/requested only {requested} frames; '
                f'InternVideo2.5 requires at least {group_size} frames.')
        print(f'Requested {args.max_frames} frames; using {adjusted} '
              f'(cap {MAX_VIDEO_FRAMES}, temporal group {group_size}).',
              file=sys.stderr, flush=True)
        pixels, count = read_video(args.input_path, adjusted, 448)

    pixels = pixels.to(device='cuda', dtype=torch.bfloat16)
    prompt = ''.join(f'Frame{i+1}: <image>\n' for i in range(count)) + args.question
    with torch.inference_mode():
        response = model.chat(
            tokenizer=tokenizer, pixel_values=pixels, question=prompt,
            generation_config={'max_new_tokens': args.max_new_tokens, 'do_sample': False},
            num_patches_list=[1] * count)
    print('\n=== InternVideo2.5 Response ===')
    print(response)
    print('=== End Response ===')


if __name__ == '__main__':
    main()
