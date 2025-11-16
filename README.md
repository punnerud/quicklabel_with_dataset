# YOLOX Object Detection - Web-based Annotation & Training

A complete web-based tool for annotating images with bounding boxes and training YOLOX object detection models.

**🔓 Apache 2.0 License** - Free for commercial use!

## Features

- 🖼️ **Web-based Interface**: Draw bounding boxes directly in your browser
- 🎯 **YOLOX Training**: Train YOLOX models (Apache 2.0 license) with your annotations
- 🔮 **Live Predictions**: Test your trained model in the annotation interface
- 📦 **Auto Augmentation**: Mosaic, HSV, and geometric augmentations
- ⚡ **Fast & Simple**: No complex setup, just annotate and train
- 🍎 **Apple Silicon Support**: Automatic MPS acceleration on M1/M2/M3 Macs

## Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Images

Place your images in the `input/` folder:

```bash
mkdir -p input
# Copy your images (.jpg, .jpeg, .png, .bmp) to input/
```

### 3. Annotate Images

Start the web annotation tool:

```bash
python app.py
```

Then open your browser to: **http://localhost:8100**

**Controls:**
- Select a class using the buttons or number keys (1-5)
- Click and drag on the image to draw a bounding box
- Use arrow keys to navigate between images
- Press Delete to remove selected bbox
- Auto-saves all annotations

### 4. Train YOLOX Model

```bash
# Train with default settings (300 epochs, YOLOX-S)
python train.py

# Train with custom settings
python train.py --epochs 200 --batch-size 16 --img-size 640
```

**Model Variants (adjust --depth and --width):**

| Model | depth | width | Parameters | Speed | Use Case |
|-------|-------|-------|------------|-------|----------|
| YOLOX-S | 0.33 | 0.50 | 9.0M | ⚡⚡⚡ | **Recommended for most cases** |
| YOLOX-M | 0.67 | 0.75 | 25.3M | ⚡⚡ | Better accuracy |
| YOLOX-L | 1.00 | 1.00 | 54.2M | ⚡ | High accuracy |
| YOLOX-X | 1.33 | 1.25 | 99.1M | 🐌 | Maximum accuracy |

**Training Options:**
- `--depth`: Model depth multiplier (default: 0.33 for YOLOX-S)
- `--width`: Model width multiplier (default: 0.50 for YOLOX-S)
- `--epochs`: Number of training epochs (default: 300)
- `--batch-size`: Batch size (default: 8, reduce if out of memory)
- `--img-size`: Input image size (default: 640)
- `--device`: Device to use (mps, cuda, cpu - auto-detected by default)

**Example - Train YOLOX-M:**
```bash
python train.py --depth 0.67 --width 0.75 --epochs 200 --batch-size 8
```

### 5. Test Predictions

After training, run the app again:

```bash
python app.py
```

Click the **"Live Detection"** checkbox and adjust the confidence slider to see your model's predictions!

## Project Structure

```
.
├── app.py                      # Web annotation interface
├── train.py                    # YOLOX training script
├── yolox_dataset.py            # Custom dataset adapter
├── yolox_inference.py          # YOLOX inference wrapper
├── requirements.txt            # Python dependencies
├── yolox/                      # YOLOX source code (Apache 2.0)
├── exps/                       # YOLOX experiment configs
├── input/                      # Place your images here
├── output/                     # Trained models
│   └── yolox_custom/
│       ├── best_ckpt.pth      # Best model checkpoint
│       └── latest_ckpt.pth    # Latest model checkpoint
└── data/
    └── annotations/            # Your annotations (JSON)
        └── annotations.json
```

## Classes

Edit the classes in `app.py` (line 30):

```python
CLASSES = ["bryter", "stikkontakt", "elsparkesykkel", "sluk", "kumlokk"]
```

Change to your own classes, for example:

```python
CLASSES = ["person", "car", "bicycle", "dog", "cat"]
```

⚠️ **Important**: If you change classes, retrain your model!

## Tips for Best Results

1. **Annotation Quality**
   - Draw tight boxes around objects
   - Include partial objects (crops at image edges)
   - Be consistent with box boundaries

2. **Dataset Size**
   - Minimum: ~50 images per class
   - Recommended: 200+ images per class
   - More is always better!

3. **Training**
   - Start with YOLOX-S for faster training
   - Use YOLOX-M for production (best balance)
   - Train for at least 200-300 epochs
   - Watch for overfitting (val loss increases)

4. **Augmentation**
   - YOLOX automatically applies:
     - Mosaic augmentation (combines 4 images)
     - Mixup augmentation
     - HSV color jittering
     - Random scaling, translation, and shearing
     - Horizontal flipping

## Troubleshooting

**Out of memory?**
- Reduce `--batch-size` (try 4 or 2)
- Use smaller model (YOLOX-S)
- Reduce `--img-size` (try 416)

**Training too slow?**
- Use YOLOX-S (--depth 0.33 --width 0.50)
- Reduce `--img-size`
- Check that MPS/CUDA is being used

**Poor accuracy?**
- Add more training images
- Train for more epochs
- Try larger model (YOLOX-M or YOLOX-L)
- Check annotation quality

**"No images found"?**
- Make sure images are in `input/` folder
- Check file extensions (.jpg, .jpeg, .png, .bmp)

## Why YOLOX?

### License Comparison

| Model | License | Commercial Use | Open Source Required |
|-------|---------|----------------|---------------------|
| **YOLOX** | **Apache 2.0** | ✅ **Yes, free** | ❌ **No** |
| YOLOv8/v11 | AGPL-3.0 | ⚠️ Requires license | ✅ Yes |
| YOLOv5 | GPL-3.0 | ⚠️ Requires license | ✅ Yes |

**YOLOX with Apache 2.0 license allows you to:**
- ✅ Use in commercial projects for free
- ✅ Keep your code private
- ✅ Modify freely without restrictions
- ✅ No obligation to open-source

### Performance

YOLOX achieves competitive performance with YOLOv5/v8 while being completely free for commercial use:

- YOLOX-S: ~40% AP on COCO (comparable to YOLOv5s)
- YOLOX-M: ~47% AP on COCO (comparable to YOLOv5m)
- Anchor-free design (simpler, cleaner)
- Strong data augmentation built-in

## Advanced Usage

### Resume Training

YOLOX automatically saves checkpoints. To resume:

```bash
# Automatically resumes from latest checkpoint if found
python train.py --resume
```

### Custom Train/Val Split

The dataset is automatically split 80/20 train/val. To modify, edit `yolox_dataset.py` line 65.

### Export to ONNX

After training, you can export to ONNX for deployment:

```bash
python tools/export_onnx.py --output-name yolox_s.onnx --input exps/default/yolox_s.py --ckpt output/yolox_custom/best_ckpt.pth
```

## License

This project uses YOLOX under the Apache 2.0 License.

**YOLOX**: Copyright (c) 2021 Megvii Inc. (Apache 2.0)

Your trained models and annotations are yours - use them freely in any project!

## Credits

Built with:
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) - High-performance anchor-free object detection (Apache 2.0)
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- HTML5 Canvas - Interactive annotation interface

---

**Previous Version**: This project was converted from Ultralytics YOLOv8 (AGPL-3.0) to YOLOX (Apache 2.0) for commercial-friendly licensing.
