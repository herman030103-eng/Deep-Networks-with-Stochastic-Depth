#!/usr/bin/env python3
"""
Prepare Tiny ImageNet for tf.keras.utils.image_dataset_from_directory.

This script:
- Detects if you passed the root that contains tiny-imagenet-200 or the tiny-imagenet-200 itself.
- Reorganizes val/images using val_annotations.txt into per-class subfolders:
    val/<class>/*.JPEG

Usage:
    python prepare_tiny_imagenet.py --data_dir path/to/data/tiny-imagenet
or
    python prepare_tiny_imagenet.py --data_dir path/to/data/tiny-imagenet/tiny-imagenet-200
"""
import argparse
import os
import shutil
import sys

def find_tiny_root(data_dir):
    # If data_dir directly contains train/ and val/, use it.
    if os.path.isdir(os.path.join(data_dir, "train")) and os.path.isdir(os.path.join(data_dir, "val")):
        return data_dir
    # If data_dir contains tiny-imagenet-200, use that
    maybe = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.isdir(maybe) and os.path.isdir(os.path.join(maybe, "train")) and os.path.isdir(os.path.join(maybe, "val")):
        return maybe
    # fallback: search one level deeper
    for name in os.listdir(data_dir):
        p = os.path.join(data_dir, name)
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "train")) and os.path.isdir(os.path.join(p, "val")):
            return p
    return None

def reorganize_val(tiny_root):
    val_dir = os.path.join(tiny_root, "val")
    images_dir = os.path.join(val_dir, "images")
    ann_file = os.path.join(val_dir, "val_annotations.txt")
    if not os.path.isdir(val_dir):
        print("ERROR: no val/ in", tiny_root)
        return False
    if not os.path.exists(ann_file):
        print("ERROR: val_annotations.txt not found in", val_dir)
        return False
    if not os.path.isdir(images_dir):
        print("ERROR: val/images not found in", val_dir)
        return False

    # Read annotations
    mapping = {}
    with open(ann_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename, cls = parts[0], parts[1]
                mapping[filename] = cls

    # Create class folders and move files
    moved = 0
    for fname, cls in mapping.items():
        cls_dir = os.path.join(val_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        src = os.path.join(images_dir, fname)
        dst = os.path.join(cls_dir, fname)
        if os.path.exists(src):
            try:
                shutil.move(src, dst)
                moved += 1
            except Exception as e:
                print("Warning: failed moving", src, "->", dst, ":", e)
        else:
            print("Warning: file not found:", src)

    # Try to remove images/ if empty
    try:
        if os.path.isdir(images_dir) and not os.listdir(images_dir):
            os.rmdir(images_dir)
    except Exception as e:
        print("Could not remove images/ folder:", e)

    print(f"Moved {moved} files into class folders under {val_dir}")
    return True

def main(data_dir):
    data_dir = os.path.abspath(data_dir)
    tiny_root = find_tiny_root(data_dir)
    if tiny_root is None:
        print("Could not locate tiny-imagenet-200 structure inside", data_dir)
        sys.exit(2)
    print("Using tiny-imagenet root:", tiny_root)
    ok = reorganize_val(tiny_root)
    if not ok:
        sys.exit(1)
    print("Done. Now val/ contains per-class subfolders suitable for image_dataset_from_directory.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Path to folder that contains tiny-imagenet-200 or is tiny-imagenet-200")
    args = p.parse_args()
    main(args.data_dir)