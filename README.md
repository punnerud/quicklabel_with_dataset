# YOLOX Object Detection - Web-based Annotation & Training

A complete web-based tool for annotating images with bounding boxes and training YOLOX object detection models. Everything is managed through an intuitive web interface - no code editing required!

**🔓 Apache 2.0 License** - Free for commercial use!

## Features

- 📁 **Multi-Project Management**: Create and manage multiple annotation projects with separate datasets
- 🖼️ **Web-based Interface**: Draw bounding boxes directly in your browser
- 🎯 **GUI-based Training**: Train YOLOX models through the web interface with real-time progress monitoring
- 🔮 **Live Predictions**: Test your trained model in the annotation interface
- 📦 **Auto Augmentation**: Mosaic, HSV, and geometric augmentations
- ⚡ **Fast & Simple**: No complex setup, just create a project, annotate and train
- 🍎 **Apple Silicon Support**: Automatic MPS acceleration on M1/M2/M3 Macs
- 🚀 **Background Training**: Train models in background while continuing to annotate

## Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Application

```bash
python app.py
```

Then open your browser to: **http://localhost:8100**

### 3. Create a Project

1. Click on the **⚙️ Settings** icon (top right)
2. Click **"+ New Project"**
3. Enter project name and define your classes (one per line)
4. Click **"Create Project"**

### 4. Add Images to Your Project

1. In the project settings page, click the **⚙️ gear icon** on your project
2. Select **"Import Images"**
3. Choose your images (.jpg, .jpeg, .png, .bmp)
4. Images will be uploaded to your project

**Alternative**: Manually place images in `projects/[project-id]/input/` folder

### 5. Annotate Images

1. Make sure your project is active (green checkmark)
2. Go back to the main annotation page
3. Draw bounding boxes by:
   - Select a class using the buttons or number keys (1-5)
   - Click and drag on the image to draw a bounding box
   - Use arrow keys to navigate between images
   - Press Delete to remove selected bbox
   - Annotations auto-save

### 6. Train Your Model

1. Go to **Project Settings** (⚙️ icon)
2. Click the **⚙️ gear icon** on your project
3. Select **"🚀 Train Model"**
4. Configure training parameters:
   - **Epochs**: Number of training iterations (default: 300)
   - **Batch Size**: Images per batch (default: 4 for YOLOX-M)
   - **Model Size**: S (faster), M (balanced), or L (accurate)
   - **Device**: MPS for Apple Silicon, CUDA for NVIDIA, or CPU
5. Click **"🚀 Start Training"**
6. Monitor real-time training progress
7. You can continue labeling while training runs in background!

### 7. Test Your Model

After training completes:
1. Return to the annotation page
2. Enable **"Show Predictions"** checkbox
3. Adjust confidence slider to see your model's predictions

## Model Variants

Choose the right model size for your use case:

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| **YOLOX-S** | 9.0M | ⚡⚡⚡ | Good | Fast inference, limited hardware |
| **YOLOX-M** | 25.3M | ⚡⚡ | Better | **Recommended for most cases** |
| **YOLOX-L** | 54.2M | ⚡ | Best | High accuracy requirements |

The model size is automatically detected from your trained model, so you can resume training seamlessly!

## Project Structure

```
.
├── app.py                      # Web annotation & training interface
├── train.py                    # YOLOX training script
├── requirements.txt            # Python dependencies
├── templates/                  # Web UI templates
│   ├── index.html             # Main annotation interface
│   ├── projects.html          # Project management
│   └── train.html             # Training interface
├── projects/                   # Your projects (auto-created)
│   └── [project-id]/
│       ├── input/             # Project images
│       ├── annotations.json   # Project annotations
│       └── output/            # Trained models
│           └── yolox_custom/
│               ├── best_ckpt.pth    # Best model
│               └── latest_ckpt.pth  # Latest model
├── data/
│   └── projects.json          # Project configurations
├── yolox/                      # YOLOX source code (Apache 2.0)
└── exps/                       # YOLOX experiment configs
```

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

### Command-Line Training (Optional)

You can also train from command line if preferred:

```bash
# Train with default settings
python train.py

# Custom settings
python train.py --epochs 200 --batch-size 8 --device mps
```

If multiple projects exist, you'll be prompted to select which one to train.

### Resume Training

Training automatically resumes from the latest checkpoint. If you want to start fresh, change the model size in the training interface (this will trigger a warning).

### Custom Train/Val Split

The dataset is automatically split 80/20 train/val. To modify, edit `yolox_dataset.py` line 65.

### Export to ONNX

After training, you can export to ONNX for deployment:

```bash
python tools/export_onnx.py --output-name yolox_m.onnx --input exps/default/yolox_m.py --ckpt projects/[project-id]/output/yolox_custom/best_ckpt.pth
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
