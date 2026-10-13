#!/usr/bin/env python3
"""Offline InternVideo2.5 inference on NVIDIA Thor.

Usage:
 python3 internvideo2_5_infer.py --video-path /workspace/videos/example.MOV --max-frames 8 --question 'What happens?'
 python3 internvideo2_5_infer.py --input-path /workspace/image.jpg --question 'Describe this image.'
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


def transform_image(size):
    return T.Compose([
        T.Lambda(lambda image: image.convert('RGB')),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def read_video(path, max_frames, size, sample_fps=None):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if count <= 0:
            raise RuntimeError(f'Video contains no readable frames: {path}')
        if sample_fps is not None:
            if fps <= 0:
                raise RuntimeError('Cannot sample by FPS: source FPS unavailable')
            times = np.arange(0, count / fps, 1.0 / sample_fps)
            indices = np.unique(np.clip(np.rint(times * fps).astype(int), 0, count - 1))
        else:
            indices = np.linspace(0, count - 1, min(max_frames, count), dtype=int)
        convert = transform_image(size)
        tensors = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Could not decode frame {index} in {path}')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(convert(Image.fromarray(rgb)))
        print(f'Video FPS: {fps:.2f}; total frames: {count}; sampled: {len(tensors)}', file=sys.stderr, flush=True)
        return torch.stack(tensors), len(tensors)
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description='InternVideo2.5 LAS-VICA experiment inference')
    parser.add_argument('--video-path', '--input-path', dest='input_path', type=Path, required=True)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument('--sample-fps', type=float, default=None)
    sampling.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--question', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--model-path', default=MODEL_PATH)
    parser.add_argument('--tokenizer-path', default=TOKENIZER_PATH)
    args = parser.parse_args()
    if not args.input_path.is_file():
        parser.error(f'Input not found: {args.input_path}')
    if args.sample_fps is not None and args.sample_fps <= 0:
        parser.error('--sample-fps must be positive')
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('--max-frames must be positive')
    if args.max_new_tokens <= 0 or args.image_size <= 0:
        parser.error('max-new-tokens and image-size must be positive')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True,
        local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation='eager').eval().cuda()
    if args.input_path.suffix.lower() in IMAGE_EXTENSIONS:
        with Image.open(args.input_path) as image:
            pixels = transform_image(args.image_size)(image).unsqueeze(0)
        count = 1
    else:
        pixels, count = read_video(args.input_path, args.max_frames or 8,
            args.image_size, sample_fps=args.sample_fps)
    pixels = pixels.to(device='cuda', dtype=torch.bfloat16)
    prompt = ''.join(f'Frame{i+1}: <image>\n' for i in range(count)) + args.question
    with torch.inference_mode():
        response = model.chat(tokenizer=tokenizer, pixel_values=pixels,
            question=prompt, generation_config={'max_new_tokens': args.max_new_tokens,
            'do_sample': False}, num_patches_list=[1] * count)
    print('\n=== InternVideo2.5 Response ===')
    print(response)
    print('=== End Response ===')


if __name__ == '__main__':
    main()
