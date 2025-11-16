#!/usr/bin/env python3
"""
Web-based Image Annotation Tool for YOLO Object Detection
Allows you to draw bounding boxes and label objects.
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np

app = Flask(__name__)

# Paths
BASE_DIR = Path("projects")
PROJECTS_FILE = Path("data/projects.json")
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
ANNOTATIONS_FILE = Path("data/annotations/annotations.json")

# Helper function to get project-specific model paths
def get_model_paths(project_id):
    """Get model paths for a specific project"""
    project_output = BASE_DIR / project_id / "output"
    # Check both possible paths (old structure had nested yolox_custom folder)
    best_paths = [
        project_output / "yolox_custom" / "best_ckpt.pth",
        project_output / "yolox_custom" / "yolox_custom" / "best_ckpt.pth"
    ]
    latest_paths = [
        project_output / "yolox_custom" / "latest_ckpt.pth",
        project_output / "yolox_custom" / "yolox_custom" / "latest_ckpt.pth"
    ]

    best_path = next((p for p in best_paths if p.exists()), best_paths[0])
    latest_path = next((p for p in latest_paths if p.exists()), latest_paths[0])

    return {
        'best': best_path,
        'latest': latest_path
    }

# Create directories
BASE_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
PROJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Class definitions (customize these!)
CLASSES = ["bryter", "stikkontakt", "elsparkesykkel", "sluk", "kumlokk"]

# Legacy support: remove "blank" from old annotations
def clean_legacy_annotations(annotations):
    """Remove 'blank' class from legacy annotations"""
    if 'classes' in annotations and 'blank' in annotations['classes']:
        annotations['classes'] = [c for c in annotations['classes'] if c != 'blank']
    return annotations

# Global model variable
loaded_model = None


def load_projects():
    """Load projects configuration"""
    if PROJECTS_FILE.exists():
        with open(PROJECTS_FILE, 'r') as f:
            return json.load(f)
    return {
        "active_project": None,
        "projects": []
    }


def save_projects(projects_data):
    """Save projects configuration"""
    with open(PROJECTS_FILE, 'w') as f:
        json.dump(projects_data, f, indent=2)


def get_active_project():
    """Get active project configuration"""
    projects_data = load_projects()
    if projects_data['active_project']:
        for project in projects_data['projects']:
            if project['id'] == projects_data['active_project']:
                return project
    return None


def get_project_stats(project):
    """Calculate project statistics"""
    import uuid
    project_dir = BASE_DIR / project['id']
    input_dir = project_dir / "input"
    annotations_file = project_dir / "annotations.json"

    # Count images
    total_images = 0
    if input_dir.exists():
        total_images = len([f for f in input_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])

    # Count annotated images and total annotations
    annotated_images = 0
    total_annotations = 0
    avg_annotations_per_image = 0.0

    if annotations_file.exists():
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
            images_data = annotations.get('images', {})

            for img_name, img_data in images_data.items():
                bboxes = img_data.get('bboxes', [])
                if len(bboxes) > 0:
                    annotated_images += 1
                    total_annotations += len(bboxes)

            # Calculate average annotations per annotated image
            if annotated_images > 0:
                avg_annotations_per_image = total_annotations / annotated_images

    # Calculate size
    size_bytes = 0
    if project_dir.exists():
        for file in project_dir.rglob('*'):
            if file.is_file():
                size_bytes += file.stat().st_size

    return {
        'total_images': total_images,
        'annotated_images': annotated_images,
        'total_annotations': total_annotations,
        'avg_annotations_per_image': round(avg_annotations_per_image, 2),
        'size_bytes': size_bytes
    }


def migrate_legacy_data():
    """Migrate existing data to a new 'Legacy Project' and clean up old structure"""
    import shutil
    import uuid

    # Check if there's legacy data to migrate
    has_legacy_images = INPUT_DIR.exists() and any(f for f in INPUT_DIR.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
    has_legacy_annotations = ANNOTATIONS_FILE.exists()

    if not has_legacy_images and not has_legacy_annotations:
        return None

    # Check if legacy project already exists
    projects_data = load_projects()
    for project in projects_data['projects']:
        if project.get('name') == 'Legacy Project':
            return None  # Already migrated

    # Create legacy project
    project_id = 'legacy'
    legacy_project = {
        'id': project_id,
        'name': 'Legacy Project',
        'labels': CLASSES.copy(),
        'created_at': 'migrated'
    }

    # Create project directory structure
    project_dir = BASE_DIR / project_id
    project_dir.mkdir(exist_ok=True)
    project_input_dir = project_dir / "input"
    project_output_dir = project_dir / "output"
    project_input_dir.mkdir(exist_ok=True)
    project_output_dir.mkdir(exist_ok=True)

    # MOVE images from input/ to projects/legacy/input/
    moved_count = 0
    if INPUT_DIR.exists():
        for img_file in INPUT_DIR.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                shutil.move(str(img_file), str(project_input_dir / img_file.name))
                moved_count += 1

    # MOVE annotations file
    project_annotations_file = project_dir / "annotations.json"
    if has_legacy_annotations:
        # Update paths in annotations to point to new location
        with open(ANNOTATIONS_FILE, 'r') as f:
            annotations = json.load(f)

        # Update image paths
        for img_name, img_data in annotations.get('images', {}).items():
            img_data['path'] = str(project_input_dir / img_name)

        # Save to new location
        with open(project_annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)

        # Remove old annotations file
        ANNOTATIONS_FILE.unlink()
    else:
        with open(project_annotations_file, 'w') as f:
            json.dump({
                "classes": CLASSES.copy(),
                "images": {}
            }, f, indent=2)

    # Move output directory if it exists
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        for item in OUTPUT_DIR.iterdir():
            dest = project_output_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    # Add to projects list and set as active
    projects_data['projects'].append(legacy_project)
    projects_data['active_project'] = project_id
    save_projects(projects_data)

    print(f"✓ Migrated {moved_count} images and annotations to 'Legacy Project'")
    return project_id


def load_yolox_model(project_id=None):
    """Load trained YOLOX model if available for the given project"""
    global loaded_model

    if project_id is None:
        active_project = get_active_project()
        if not active_project:
            return None
        project_id = active_project['id']

    # Get project-specific model paths
    model_paths = get_model_paths(project_id)
    model_path = None
    if model_paths['best'].exists():
        model_path = model_paths['best']
    elif model_paths['latest'].exists():
        model_path = model_paths['latest']

    if model_path is None:
        return None

    try:
        from yolox_inference import YOLOXPredictor
        import torch

        # Get project classes
        active_project = get_active_project()
        if not active_project:
            return None

        loaded_model = YOLOXPredictor(
            model_path=model_path,
            class_names=active_project['labels'],
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            img_size=640
        )
        print(f"✓ YOLOX model loaded from {model_path}")
        return loaded_model
    except Exception as e:
        print(f"Failed to load YOLOX model: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_yolox_inference(image_path, model, conf_threshold=0.25):
    """Run YOLOX inference on an image"""
    if model is None:
        return None

    try:
        # Run inference using YOLOXPredictor
        detections = model.predict(image_path, conf_threshold=conf_threshold)

        # Load image for visualization
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Draw predictions
        predictions = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class']
            conf = det['confidence']

            predictions.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'class': cls_name,
                'confidence': float(conf)
            })

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

            # Draw label with large font
            label = f"{cls_name} {conf:.2f}"
            try:
                # Try to load a large TrueType font (size 100 = ~10x larger than default)
                font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 100)
            except:
                # Fallback to default font if TrueType not available
                font = ImageFont.load_default()
            draw.text((x1, y1 - 120), label, fill='red', font=font)

        # Convert to base64
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return {
            'image': img_base64,
            'predictions': predictions
        }
    except Exception as e:
        print(f"Error running inference: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_annotations():
    """Load existing annotations or create new dict"""
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, 'r') as f:
            data = json.load(f)
            # Clean legacy "blank" class
            data = clean_legacy_annotations(data)
            return data
    return {
        "classes": CLASSES,
        "images": {}
    }


def save_annotations(annotations):
    """Save annotations to JSON file"""
    # Clean legacy classes before saving
    annotations = clean_legacy_annotations(annotations)
    # Ensure we always use current CLASSES
    annotations['classes'] = CLASSES
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(annotations, f, indent=2)


def get_image_list():
    """Get list of images in input directory"""
    return sorted([
        f.name for f in INPUT_DIR.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ])


@app.route('/')
def index():
    """Serve the main annotation page"""
    return render_template('index.html')


@app.route('/projects')
def projects():
    """Serve the projects management page"""
    # Check for active training jobs
    active_training = None
    for job_id, job in training_jobs.items():
        if job['status'] == 'running':
            # Get project name
            project_id = job['project_id']
            projects_data = load_projects()
            project_name = None
            for p in projects_data['projects']:
                if p['id'] == project_id:
                    project_name = p['name']
                    break

            active_training = {
                'job_id': job_id,
                'project_id': project_id,
                'project_name': project_name or project_id
            }
            break  # Only show first active job

    return render_template('projects.html', active_training=active_training)


@app.route('/train/<project_id>')
def train_page(project_id):
    """Serve the training page for a specific project"""
    projects_data = load_projects()
    project = None
    for p in projects_data['projects']:
        if p['id'] == project_id:
            project = p
            break

    if not project:
        return "Project not found", 404

    # Get project stats
    stats = get_project_stats(project)

    # Check for existing trained model
    model_paths = get_model_paths(project_id)
    model_info = None

    if model_paths['best'].exists():
        model_file = model_paths['best']
        model_type = 'best'
    elif model_paths['latest'].exists():
        model_file = model_paths['latest']
        model_type = 'latest'
    else:
        model_file = None

    if model_file:
        import datetime
        import re
        import torch
        stat = model_file.stat()
        size_mb = round(stat.st_size / (1024 * 1024), 2)

        # Try to detect model size from checkpoint config
        model_size = 'm'  # Default to medium
        try:
            ckpt = torch.load(model_file, map_location='cpu')
            config = ckpt.get('model_config', {})
            depth = config.get('depth')
            width = config.get('width')

            # Map depth/width to model size
            # YOLOX-S: depth=0.33, width=0.375
            # YOLOX-M: depth=0.67, width=0.75
            # YOLOX-L: depth=1.0, width=1.0
            if depth and width:
                if depth <= 0.4:
                    model_size = 's'
                elif depth <= 0.8:
                    model_size = 'm'
                else:
                    model_size = 'l'
        except Exception as e:
            logger.warning(f"Could not read model config from checkpoint: {e}")
            # Fallback to file size estimation
            if size_mb < 50:
                model_size = 's'
            elif size_mb < 150:
                model_size = 'm'
            else:
                model_size = 'l'

        model_info = {
            'exists': True,
            'type': model_type,
            'path': str(model_file),
            'size_mb': size_mb,
            'modified': datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'model_size': model_size
        }

        # Try to extract epoch number from filename if it's an epoch checkpoint
        epoch_match = re.search(r'epoch_(\d+)', model_file.name)
        if epoch_match:
            model_info['epoch'] = int(epoch_match.group(1))

        # Try to read training info from log file if it exists
        log_dir = model_file.parent
        log_file = log_dir / 'train_log.txt'
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()

                    # Extract total epochs trained
                    epoch_numbers = []
                    loss_values = []
                    for line in lines:
                        # Look for epoch info
                        if 'epoch:' in line.lower():
                            epoch_match = re.search(r'epoch:\s*(\d+)', line, re.IGNORECASE)
                            if epoch_match:
                                epoch_numbers.append(int(epoch_match.group(1)))

                        # Look for loss values in the format: Avg loss: X.XXX (iou: X.XXX, conf: X.XXX, cls: X.XXX)
                        if 'Avg loss:' in line:
                            loss_match = re.search(r'Avg loss:\s*([\d.]+)', line)
                            if loss_match:
                                loss_values.append(float(loss_match.group(1)))

                    if epoch_numbers:
                        model_info['total_epochs'] = max(epoch_numbers)

                    if loss_values:
                        # Get best (lowest), latest, and average loss
                        model_info['best_loss'] = round(min(loss_values), 3)
                        model_info['latest_loss'] = round(loss_values[-1], 3)
                        model_info['avg_loss'] = round(sum(loss_values) / len(loss_values), 3)

                    # Parse last training timestamp
                    for line in reversed(lines[-100:]):
                        if 'Avg loss:' in line:
                            # Extract timestamp from log line like: "2025-11-16 00:51:21.721 | INFO ..."
                            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            if timestamp_match:
                                model_info['last_training_time'] = timestamp_match.group(1)
                            break
            except Exception as e:
                logger.error(f"Error reading training log: {e}")
                pass

    # Check if there's an active training job for this project
    active_job_id = None
    for job_id, job in training_jobs.items():
        if job['project_id'] == project_id and job['status'] == 'running':
            active_job_id = job_id
            break

    return render_template('train.html', project=project, stats=stats, model_info=model_info, active_job_id=active_job_id)


@app.route('/api/projects')
def get_projects():
    """Get all projects with statistics"""
    projects_data = load_projects()

    # Add statistics to each project
    for project in projects_data['projects']:
        stats = get_project_stats(project)
        project.update(stats)

    return jsonify(projects_data)


@app.route('/api/projects', methods=['POST'])
def create_project():
    """Create a new project"""
    import uuid
    data = request.json
    name = data.get('name', '').strip()
    labels = data.get('labels', [])

    if not name or not labels:
        return jsonify({'status': 'error', 'message': 'Name and labels are required'}), 400

    projects_data = load_projects()

    # Create new project
    project_id = str(uuid.uuid4())[:8]
    new_project = {
        'id': project_id,
        'name': name,
        'labels': labels,
        'created_at': str(Path().cwd())  # placeholder, you can use datetime if needed
    }

    # Create project directory structure
    project_dir = BASE_DIR / project_id
    project_dir.mkdir(exist_ok=True)
    (project_dir / "input").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)

    # Create empty annotations file
    annotations_file = project_dir / "annotations.json"
    with open(annotations_file, 'w') as f:
        json.dump({
            "classes": labels,
            "images": {}
        }, f, indent=2)

    projects_data['projects'].append(new_project)
    save_projects(projects_data)

    return jsonify({'status': 'success', 'project': new_project})


@app.route('/api/projects/<project_id>/upload-images', methods=['POST'])
def upload_project_images(project_id):
    """Upload images to a project"""
    if 'images' not in request.files:
        return jsonify({'status': 'error', 'message': 'No images provided'}), 400

    images = request.files.getlist('images')

    if len(images) == 0:
        return jsonify({'status': 'error', 'message': 'No images provided'}), 400

    # Check project exists
    projects_data = load_projects()
    project = None
    for p in projects_data['projects']:
        if p['id'] == project_id:
            project = p
            break

    if not project:
        return jsonify({'status': 'error', 'message': 'Project not found'}), 404

    project_dir = BASE_DIR / project_id
    input_dir = project_dir / "input"

    uploaded_count = 0
    for image in images:
        if image.filename:
            # Save image
            image.save(input_dir / image.filename)
            uploaded_count += 1

    return jsonify({
        'status': 'success',
        'uploaded': uploaded_count,
        'message': f'Uploaded {uploaded_count} image(s)'
    })


@app.route('/api/projects/<project_id>/images')
def get_project_images(project_id):
    """Get list of images in a project"""
    projects_data = load_projects()
    project = None
    for p in projects_data['projects']:
        if p['id'] == project_id:
            project = p
            break

    if not project:
        return jsonify({'status': 'error', 'message': 'Project not found'}), 404

    project_dir = BASE_DIR / project_id
    input_dir = project_dir / "input"

    if not input_dir.exists():
        return jsonify({'status': 'success', 'images': []})

    images = sorted([
        f.name for f in input_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ])

    return jsonify({'status': 'success', 'images': images})


@app.route('/api/projects/<project_id>/delete-images', methods=['POST'])
def delete_project_images(project_id):
    """Delete images from a project"""
    data = request.json
    images_to_delete = data.get('images', [])

    if not images_to_delete:
        return jsonify({'status': 'error', 'message': 'No images specified'}), 400

    projects_data = load_projects()
    project = None
    for p in projects_data['projects']:
        if p['id'] == project_id:
            project = p
            break

    if not project:
        return jsonify({'status': 'error', 'message': 'Project not found'}), 404

    project_dir = BASE_DIR / project_id
    input_dir = project_dir / "input"
    annotations_file = project_dir / "annotations.json"

    deleted_count = 0

    # Delete image files
    for img_name in images_to_delete:
        img_path = input_dir / img_name
        if img_path.exists():
            img_path.unlink()
            deleted_count += 1

    # Remove from annotations
    if annotations_file.exists():
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        for img_name in images_to_delete:
            if img_name in annotations.get('images', {}):
                del annotations['images'][img_name]

        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)

    return jsonify({
        'status': 'success',
        'deleted': deleted_count,
        'message': f'Deleted {deleted_count} image(s)'
    })


@app.route('/api/projects/active', methods=['POST'])
def set_active_project():
    """Set the active project"""
    data = request.json
    project_id = data.get('project_id')

    if not project_id:
        return jsonify({'status': 'error', 'message': 'Project ID required'}), 400

    projects_data = load_projects()
    projects_data['active_project'] = project_id
    save_projects(projects_data)

    return jsonify({'status': 'success'})


@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project"""
    import shutil

    projects_data = load_projects()

    # Find and remove project
    project = None
    for i, p in enumerate(projects_data['projects']):
        if p['id'] == project_id:
            project = projects_data['projects'].pop(i)
            break

    if not project:
        return jsonify({'status': 'error', 'message': 'Project not found'}), 404

    # Delete project directory
    project_dir = BASE_DIR / project_id
    if project_dir.exists():
        shutil.rmtree(project_dir)

    # If this was the active project, clear it
    if projects_data['active_project'] == project_id:
        projects_data['active_project'] = None

    save_projects(projects_data)

    return jsonify({'status': 'success'})


@app.route('/api/projects/<project_id>/rename-labels', methods=['POST'])
def rename_project_labels(project_id):
    """Rename labels in a project (legacy endpoint - use update-labels instead)"""
    data = request.json
    label_map = data.get('label_map', {})

    if not label_map:
        return jsonify({'status': 'error', 'message': 'No label mappings provided'}), 400

    # Forward to update-labels endpoint
    return update_project_labels(project_id)


@app.route('/api/projects/<project_id>/update-labels', methods=['POST'])
def update_project_labels(project_id):
    """Update labels in a project (rename, delete, add new)"""
    data = request.json
    label_map = data.get('label_map', {})  # {old_name: new_name}
    deleted_labels = data.get('deleted_labels', [])  # [label1, label2]
    new_labels = data.get('new_labels', [])  # [label1, label2]

    projects_data = load_projects()

    # Find project
    project = None
    for p in projects_data['projects']:
        if p['id'] == project_id:
            project = p
            break

    if not project:
        return jsonify({'status': 'error', 'message': 'Project not found'}), 404

    # Update annotations file
    project_dir = BASE_DIR / project_id
    annotations_file = project_dir / "annotations.json"

    if annotations_file.exists():
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        # 1. Rename labels in all bboxes
        for img_name, img_data in annotations.get('images', {}).items():
            updated_bboxes = []
            for bbox in img_data.get('bboxes', []):
                bbox_class = bbox.get('class')

                # Skip deleted labels
                if bbox_class in deleted_labels:
                    continue

                # Rename if needed
                if bbox_class in label_map:
                    bbox['class'] = label_map[bbox_class]

                updated_bboxes.append(bbox)

            img_data['bboxes'] = updated_bboxes

        # 2. Update classes list
        updated_classes = []
        for cls in annotations.get('classes', []):
            # Skip deleted
            if cls in deleted_labels:
                continue
            # Rename
            if cls in label_map:
                updated_classes.append(label_map[cls])
            else:
                updated_classes.append(cls)

        # 3. Add new labels
        for new_label in new_labels:
            if new_label not in updated_classes:
                updated_classes.append(new_label)

        annotations['classes'] = updated_classes

        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)

    # Update project labels
    updated_project_labels = []
    for label in project['labels']:
        # Skip deleted
        if label in deleted_labels:
            continue
        # Rename
        if label in label_map:
            updated_project_labels.append(label_map[label])
        else:
            updated_project_labels.append(label)

    # Add new labels
    for new_label in new_labels:
        if new_label not in updated_project_labels:
            updated_project_labels.append(new_label)

    project['labels'] = updated_project_labels

    save_projects(projects_data)

    return jsonify({'status': 'success'})


@app.route('/api/projects/<project_id>/export-annotations')
def export_project_annotations(project_id):
    """Export project annotations as JSON"""
    project_dir = BASE_DIR / project_id
    annotations_file = project_dir / "annotations.json"

    if not annotations_file.exists():
        return jsonify({'status': 'error', 'message': 'No annotations found'}), 404

    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    return jsonify({'status': 'success', 'annotations': annotations})


@app.route('/api/projects/<project_id>/export-model')
def export_project_model(project_id):
    """Export best model for a project"""
    project_dir = BASE_DIR / project_id
    model_path = project_dir / "output" / "yolox_custom" / "yolox_custom" / "best_ckpt.pth"

    if not model_path.exists():
        return jsonify({
            'status': 'error',
            'message': 'No trained model found for this project. Train a model first using: python train.py'
        })

    # Return download URL
    return jsonify({
        'status': 'success',
        'download_url': f'/download-model/{project_id}'
    })


@app.route('/download-model/<project_id>')
def download_model(project_id):
    """Download best model file"""
    project_dir = BASE_DIR / project_id
    model_path = project_dir / "output" / "yolox_custom" / "yolox_custom" / "best_ckpt.pth"

    if not model_path.exists():
        return "Model not found", 404

    return send_from_directory(
        model_path.parent,
        model_path.name,
        as_attachment=True,
        download_name=f'{project_id}_best_model.pth'
    )


@app.route('/api/config')
def get_config():
    """Get configuration (classes, image list)"""
    active_project = get_active_project()

    if not active_project:
        return jsonify({
            'classes': [],
            'images': [],
            'total_images': 0,
            'model_available': False,
            'project_name': None,
            'error': 'No active project. Please create or select a project.'
        })

    project_dir = BASE_DIR / active_project['id']
    input_dir = project_dir / "input"
    images = sorted([
        f.name for f in input_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ]) if input_dir.exists() else []
    classes = active_project['labels']
    project_name = active_project['name']

    # Check if model exists for this specific project
    model_paths = get_model_paths(active_project['id'])
    model_available = model_paths['best'].exists() or model_paths['latest'].exists()

    return jsonify({
        'classes': classes,
        'images': images,
        'total_images': len(images),
        'model_available': model_available,
        'project_name': project_name,
        'project_id': active_project['id']
    })


@app.route('/api/predict/<image_name>')
def predict_image(image_name):
    """Run YOLO prediction on an image"""
    global loaded_model

    if loaded_model is None:
        loaded_model = load_yolox_model()

    if loaded_model is None:
        return jsonify({'error': 'Model not available. Train a model first.'}), 404

    image_path = INPUT_DIR / image_name
    if not image_path.exists():
        return jsonify({'error': 'Image not found'}), 404

    # Get confidence threshold from query parameter (default 0.25)
    conf_threshold = float(request.args.get('conf', 0.25))

    result = run_yolox_inference(image_path, loaded_model, conf_threshold)

    if result is None:
        return jsonify({'error': 'Failed to run inference'}), 500

    return jsonify(result)


@app.route('/api/annotations')
def get_annotations():
    """Get all annotations"""
    active_project = get_active_project()

    if not active_project:
        return jsonify({"classes": [], "images": {}})

    project_dir = BASE_DIR / active_project['id']
    annotations_file = project_dir / "annotations.json"
    if annotations_file.exists():
        with open(annotations_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"classes": active_project['labels'], "images": {}})


@app.route('/api/annotation/<image_name>')
def get_annotation(image_name):
    """Get annotation for specific image"""
    active_project = get_active_project()

    if not active_project:
        return jsonify({'bboxes': []})

    project_dir = BASE_DIR / active_project['id']
    annotations_file = project_dir / "annotations.json"
    if annotations_file.exists():
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
            if image_name in annotations.get('images', {}):
                return jsonify(annotations['images'][image_name])
    return jsonify({'bboxes': []})


@app.route('/api/annotation/<image_name>', methods=['POST'])
def save_annotation(image_name):
    """Save annotation for specific image"""
    active_project = get_active_project()
    data = request.json
    bboxes = data.get('bboxes', [])

    if not active_project:
        return jsonify({'status': 'error', 'message': 'No active project'}), 400

    project_dir = BASE_DIR / active_project['id']
    annotations_file = project_dir / "annotations.json"
    input_dir = project_dir / "input"

    # Load existing annotations
    if annotations_file.exists():
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
    else:
        annotations = {"classes": active_project['labels'], "images": {}}

    # Save annotation
    annotations['images'][image_name] = {
        'path': str(input_dir / image_name),
        'bboxes': bboxes
    }

    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    return jsonify({'status': 'success'})


@app.route('/api/save-all', methods=['POST'])
def save_all():
    """Save all annotations"""
    annotations = load_annotations()
    save_annotations(annotations)
    return jsonify({
        'status': 'success',
        'message': f'Saved {len(annotations["images"])} annotations'
    })


@app.route('/input/<path:filename>')
def serve_image(filename):
    """Serve images from input directory"""
    active_project = get_active_project()

    if not active_project:
        return "No active project", 404

    project_dir = BASE_DIR / active_project['id']
    input_dir = project_dir / "input"
    return send_from_directory(input_dir, filename)


@app.route('/projects/<project_id>/input/<path:filename>')
def serve_project_image(project_id, filename):
    """Serve images from a specific project's input directory"""
    projects_data = load_projects()
    project = None
    for p in projects_data['projects']:
        if p['id'] == project_id:
            project = p
            break

    if not project:
        return "Project not found", 404

    project_dir = BASE_DIR / project_id
    input_dir = project_dir / "input"

    if not input_dir.exists():
        return "Project input directory not found", 404

    return send_from_directory(input_dir, filename)


def create_html_template():
    """Create the HTML template for the annotation interface"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Annotator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: #2c3e50;
            color: white;
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            font-size: 24px;
        }

        .controls {
            background: #34495e;
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: white;
        }

        .nav-buttons button {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }

        .nav-buttons button:hover {
            background: #2980b9;
        }

        .nav-buttons button:disabled {
            background: #95a5a6;
            cursor: not-allowed;
        }

        .save-btn {
            background: #27ae60 !important;
            padding: 10px 30px !important;
        }

        .save-btn:hover {
            background: #229954 !important;
        }

        .predict-btn {
            background: #e74c3c !important;
        }

        .predict-btn:hover {
            background: #c0392b !important;
        }

        .main-content {
            display: flex;
            height: calc(100vh - 200px);
        }

        .image-panel {
            flex: 1;
            padding: 30px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #ecf0f1;
            position: relative;
        }

        .canvas-container {
            position: relative;
            max-width: 100%;
            max-height: 100%;
            display: inline-block;
        }

        #imageCanvas {
            max-width: 100%;
            max-height: 100%;
            cursor: crosshair;
            border: 2px solid #2c3e50;
        }

        .annotation-panel {
            width: 350px;
            padding: 30px;
            border-left: 1px solid #ddd;
            overflow-y: auto;
        }

        .annotation-panel h2 {
            margin-bottom: 20px;
            color: #2c3e50;
        }

        .class-selector {
            background: #f8f9fa;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 6px;
            border: 2px solid #e0e0e0;
        }

        .class-selector h3 {
            margin-bottom: 10px;
            font-size: 14px;
            color: #666;
        }

        .class-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .class-btn {
            padding: 8px 16px;
            border: 2px solid #3498db;
            background: white;
            color: #3498db;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .class-btn.active {
            background: #3498db;
            color: white;
        }

        .class-btn:hover {
            background: #e3f2fd;
        }

        .class-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 24px;
            height: 24px;
            background: #3498db;
            color: white;
            border-radius: 50%;
            font-weight: bold;
            font-size: 12px;
        }

        .class-btn.active .class-number {
            background: white;
            color: #3498db;
        }

        .bbox-list {
            margin-top: 20px;
        }

        .bbox-item {
            background: #f8f9fa;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 4px;
            border-left: 4px solid #3498db;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .bbox-info {
            flex: 1;
        }

        .bbox-class {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 4px;
        }

        .bbox-coords {
            font-size: 12px;
            color: #666;
        }

        .bbox-delete {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }

        .bbox-delete:hover {
            background: #c0392b;
        }

        .instructions {
            background: #fff3cd;
            padding: 15px;
            margin-top: 20px;
            border-radius: 4px;
            font-size: 14px;
            border-left: 4px solid #ffc107;
        }

        .instructions strong {
            display: block;
            margin-bottom: 8px;
        }

        kbd {
            background: #f4f4f4;
            border: 1px solid #ccc;
            border-radius: 3px;
            box-shadow: 0 1px 0 rgba(0,0,0,0.2);
            color: #333;
            display: inline-block;
            font-family: monospace;
            font-size: 12px;
            line-height: 1;
            padding: 3px 6px;
            white-space: nowrap;
        }

        input[type="range"] {
            -webkit-appearance: none;
            appearance: none;
            height: 8px;
            background: #ddd;
            border-radius: 5px;
            outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            background: #3498db;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        input[type="range"]::-moz-range-thumb {
            width: 20px;
            height: 20px;
            background: #3498db;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            background: #2980b9;
        }

        input[type="range"]::-moz-range-thumb:hover {
            background: #2980b9;
        }

        .stats {
            background: #e8f5e9;
            padding: 10px 15px;
            border-radius: 4px;
            font-size: 14px;
            display: flex;
            align-items: center;
        }

        .prediction-toggle {
            background: #fff;
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            gap: 10px;
            border: 2px solid #ddd;
        }

        .prediction-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: none;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.95);
            z-index: 10;
        }

        .prediction-overlay.visible {
            display: flex;
        }

        .prediction-overlay img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }

        .loading-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 YOLO Annotator <span id="projectName" style="font-size: 16px; font-weight: normal; opacity: 0.8;"></span></h1>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div class="stats" id="stats">Loading...</div>
                <a href="/projects" style="text-decoration: none; font-size: 24px; opacity: 0.8; transition: opacity 0.2s; line-height: 1;" onmouseover="this.style.opacity='1'" onmouseout="this.style.opacity='0.8'" title="Project Settings">⚙️</a>
            </div>
        </div>

        <div class="controls">
            <div class="progress-info">
                <span id="progress">Image 0 / 0</span>
                <span class="filename" id="filename">No image</span>
            </div>
            <div class="nav-buttons">
                <button id="prevBtn" onclick="previousImage()">← Previous (P)</button>
                <button id="nextBtn" onclick="nextImage()">Next (N) →</button>
            </div>
        </div>

        <div class="main-content">
            <div class="image-panel">
                <div class="canvas-container" id="canvasContainer">
                    <canvas id="imageCanvas"></canvas>
                </div>
                <div class="prediction-overlay" id="predictionOverlay">
                    <div class="loading-spinner"></div>
                </div>
            </div>

            <div class="annotation-panel">
                <!-- Live detection toggle -->
                <div class="prediction-toggle" id="liveDetectionContainer" style="display: none; margin-bottom: 15px;">
                    <label style="display: flex; align-items: center; gap: 10px; cursor: pointer; font-weight: 600;">
                        <input type="checkbox" id="liveDetectionCheckbox" onchange="toggleLiveDetection()" style="width: 20px; height: 20px; cursor: pointer;">
                        <span>🤖 Live Detection</span>
                    </label>

                    <!-- Confidence threshold slider -->
                    <div id="confidenceSliderContainer" style="margin-top: 10px; display: none;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <label style="font-size: 13px; font-weight: 600; color: #555;">Confidence Threshold:</label>
                            <span id="confidenceValue" style="font-size: 13px; font-weight: bold; color: #3498db;">25%</span>
                        </div>
                        <input type="range" id="confidenceSlider" min="1" max="99" value="25" step="1"
                               oninput="updateConfidence()" onchange="reloadDetections()"
                               style="width: 100%; cursor: pointer;">
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #999; margin-top: 2px;">
                            <span>1% (all detections)</span>
                            <span>99% (only confident)</span>
                        </div>
                    </div>

                    <div id="detectionStatus" style="font-size: 12px; color: #666; margin-top: 5px; display: none;">
                        Loading model...
                    </div>
                </div>

                <h2>Annotations</h2>

                <!-- Manual prediction button (hidden when live detection is on) -->
                <div class="prediction-toggle" id="predictionToggleContainer" style="display: none;">
                    <button class="predict-btn" onclick="showPredictions()">🔮 Show Predictions</button>
                </div>

                <!-- Class selector -->
                <div class="class-selector">
                    <h3>Select Class:</h3>
                    <div class="class-buttons" id="classButtons"></div>
                </div>

                <!-- Bounding box list -->
                <div class="bbox-list">
                    <h3>Bounding Boxes (<span id="bboxCount">0</span>)</h3>
                    <div id="bboxList"></div>
                </div>

                <div class="instructions">
                    <strong>Keyboard Shortcuts:</strong>
                    • <kbd>1-5</kbd> Select class<br>
                    • <kbd>P</kbd> / <kbd>N</kbd> Previous / Next image<br>
                    • <kbd>Delete</kbd> Delete selected box<br>
                    • <kbd>Esc</kbd> Deselect box<br>
                    <br>
                    <strong>Mouse:</strong>
                    • Click & drag to draw new box<br>
                    • Click box to select<br>
                    • Drag box to move<br>
                    • Drag corners to resize
                </div>
            </div>
        </div>
    </div>

    <script>
        let config = null;
        let images = [];
        let currentIndex = 0;
        let annotations = {};
        let currentBboxes = [];
        let selectedClass = null;
        let isDrawing = false;
        let isDragging = false;
        let isResizing = false;
        let startX, startY;
        let canvas, ctx;
        let currentImage = null;
        let selectedBboxIndex = -1;
        let resizeHandle = null; // 'tl', 'tr', 'bl', 'br' for top-left, top-right, bottom-left, bottom-right
        let dragOffsetX = 0;
        let dragOffsetY = 0;
        let liveDetectionEnabled = false;
        let detectionOverlayImage = null;
        let confidenceThreshold = 0.25;

        const COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];
        const HANDLE_SIZE = 20;

        // Initialize
        async function init() {
            try {
                const configRes = await fetch('/api/config');
                config = await configRes.json();
                images = config.images;

                // Display project name if available
                if (config.project_name) {
                    document.getElementById('projectName').textContent = `(${config.project_name})`;
                }

                const annotationsRes = await fetch('/api/annotations');
                annotations = await annotationsRes.json();

                if (images.length === 0) {
                    alert('No images found in input/ folder. Please add images and refresh.');
                    return;
                }

                // Setup canvas
                canvas = document.getElementById('imageCanvas');
                ctx = canvas.getContext('2d');

                // Canvas event listeners
                canvas.addEventListener('mousedown', startDrawing);
                canvas.addEventListener('mousemove', draw);
                canvas.addEventListener('mouseup', stopDrawing);
                canvas.addEventListener('mouseout', stopDrawing);
                canvas.addEventListener('mousemove', updateCursor);

                createClassButtons();

                // Always show live detection toggle (will show error if model not available)
                document.getElementById('liveDetectionContainer').style.display = 'block';

                // Show prediction button if model is available
                if (config.model_available) {
                    document.getElementById('predictionToggleContainer').style.display = 'block';
                }

                // Find first unannotated image or start at 0
                let startIndex = 0;
                for (let i = 0; i < images.length; i++) {
                    if (!annotations.images[images[i]]) {
                        startIndex = i;
                        break;
                    }
                }

                loadImage(startIndex);
                updateStats();
            } catch (error) {
                console.error('Failed to initialize:', error);
                alert('Failed to load configuration. Make sure the server is running.');
            }
        }

        function createClassButtons() {
            const container = document.getElementById('classButtons');
            container.innerHTML = '';

            config.classes.forEach((className, idx) => {
                const btn = document.createElement('button');
                btn.className = 'class-btn';
                btn.innerHTML = `<span class="class-number">${idx + 1}</span> ${className}`;
                btn.onclick = () => selectClass(className);
                btn.id = `class-btn-${className}`;
                container.appendChild(btn);
            });

            // Select first class by default
            if (config.classes.length > 0) {
                selectClass(config.classes[0]);
            }
        }

        function selectClass(className) {
            selectedClass = className;

            // Update button states
            config.classes.forEach(cls => {
                const btn = document.getElementById(`class-btn-${cls}`);
                if (cls === className) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
        }

        async function loadImage(index) {
            if (index < 0 || index >= images.length) return;

            currentIndex = index;
            const imageName = images[index];

            // Reset selection
            selectedBboxIndex = -1;
            isDragging = false;
            isDrawing = false;
            isResizing = false;

            // Clear current state immediately
            currentBboxes = [];
            currentImage = null;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            updateBboxList();

            // Update UI
            document.getElementById('progress').textContent = `Image ${index + 1} / ${images.length}`;
            document.getElementById('filename').textContent = imageName;

            // Load annotation first
            try {
                const res = await fetch(`/api/annotation/${imageName}`);
                const annotation = await res.json();
                currentBboxes = annotation.bboxes || [];
                updateBboxList();
            } catch (error) {
                console.error('Failed to load annotation:', error);
                currentBboxes = [];
            }

            // Load image and wait for it to complete
            const img = new Image();
            await new Promise((resolve, reject) => {
                img.onload = () => {
                    currentImage = img;
                    canvas.width = img.width;
                    canvas.height = img.height;
                    redrawCanvas();
                    resolve();
                };
                img.onerror = reject;
                img.src = `/input/${imageName}`;
            });

            // Update navigation buttons
            document.getElementById('prevBtn').disabled = index === 0;
            document.getElementById('nextBtn').disabled = index === images.length - 1;

            // Hide predictions
            const overlay = document.getElementById('predictionOverlay');
            overlay.classList.remove('visible');

            // Load detections if live detection is enabled
            if (liveDetectionEnabled) {
                await loadDetections();
            }
        }

        function redrawCanvas() {
            if (!currentImage) return;

            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw image
            ctx.drawImage(currentImage, 0, 0);

            // Draw detection overlay if live detection is enabled
            if (liveDetectionEnabled && detectionOverlayImage) {
                ctx.globalAlpha = 0.7;
                ctx.drawImage(detectionOverlayImage, 0, 0, canvas.width, canvas.height);
                ctx.globalAlpha = 1.0;
            }

            // Draw bboxes
            currentBboxes.forEach((bbox, idx) => {
                const color = COLORS[config.classes.indexOf(bbox.class) % COLORS.length];
                const isSelected = idx === selectedBboxIndex;

                ctx.strokeStyle = color;
                ctx.lineWidth = isSelected ? 16 : 12;
                ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

                // Draw label
                ctx.fillStyle = color;
                ctx.font = 'bold 64px Arial';
                const label = bbox.class;
                const textWidth = ctx.measureText(label).width;
                const textHeight = 80;
                ctx.fillRect(bbox.x, bbox.y - textHeight, textWidth + 40, textHeight);
                ctx.fillStyle = 'white';
                ctx.fillText(label, bbox.x + 20, bbox.y - 20);

                // Draw resize handles if selected
                if (isSelected) {
                    ctx.fillStyle = 'white';
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 3;

                    // Top-left
                    ctx.fillRect(bbox.x - HANDLE_SIZE/2, bbox.y - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);
                    ctx.strokeRect(bbox.x - HANDLE_SIZE/2, bbox.y - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);

                    // Top-right
                    ctx.fillRect(bbox.x + bbox.width - HANDLE_SIZE/2, bbox.y - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);
                    ctx.strokeRect(bbox.x + bbox.width - HANDLE_SIZE/2, bbox.y - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);

                    // Bottom-left
                    ctx.fillRect(bbox.x - HANDLE_SIZE/2, bbox.y + bbox.height - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);
                    ctx.strokeRect(bbox.x - HANDLE_SIZE/2, bbox.y + bbox.height - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);

                    // Bottom-right
                    ctx.fillRect(bbox.x + bbox.width - HANDLE_SIZE/2, bbox.y + bbox.height - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);
                    ctx.strokeRect(bbox.x + bbox.width - HANDLE_SIZE/2, bbox.y + bbox.height - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);
                }
            });
        }

        function getCanvasCoordinates(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;

            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        }

        function getResizeHandle(x, y, bbox) {
            const handles = [
                { name: 'tl', x: bbox.x, y: bbox.y },
                { name: 'tr', x: bbox.x + bbox.width, y: bbox.y },
                { name: 'bl', x: bbox.x, y: bbox.y + bbox.height },
                { name: 'br', x: bbox.x + bbox.width, y: bbox.y + bbox.height }
            ];

            for (let handle of handles) {
                if (Math.abs(x - handle.x) < HANDLE_SIZE && Math.abs(y - handle.y) < HANDLE_SIZE) {
                    return handle.name;
                }
            }
            return null;
        }

        function isInsideBbox(x, y, bbox) {
            return x >= bbox.x && x <= bbox.x + bbox.width &&
                   y >= bbox.y && y <= bbox.y + bbox.height;
        }

        function startDrawing(e) {
            const coords = getCanvasCoordinates(e);

            // Check if clicking on existing bbox (for drag/resize)
            for (let i = currentBboxes.length - 1; i >= 0; i--) {
                const bbox = currentBboxes[i];

                // Check if clicking on resize handle
                const handle = getResizeHandle(coords.x, coords.y, bbox);
                if (handle) {
                    isResizing = true;
                    selectedBboxIndex = i;
                    resizeHandle = handle;
                    startX = coords.x;
                    startY = coords.y;
                    canvas.style.cursor = 'nwse-resize';
                    return;
                }

                // Check if clicking inside bbox (for dragging)
                if (isInsideBbox(coords.x, coords.y, bbox)) {
                    isDragging = true;
                    selectedBboxIndex = i;
                    dragOffsetX = coords.x - bbox.x;
                    dragOffsetY = coords.y - bbox.y;
                    canvas.style.cursor = 'move';
                    redrawCanvas();
                    return;
                }
            }

            // Deselect if clicking outside
            if (selectedBboxIndex !== -1) {
                selectedBboxIndex = -1;
                redrawCanvas();
            }

            // Start drawing new box
            if (!selectedClass) {
                alert('Please select a class first!');
                return;
            }

            isDrawing = true;
            startX = coords.x;
            startY = coords.y;
            canvas.style.cursor = 'crosshair';
        }

        function draw(e) {
            const coords = getCanvasCoordinates(e);

            // Handle dragging
            if (isDragging) {
                const bbox = currentBboxes[selectedBboxIndex];
                bbox.x = coords.x - dragOffsetX;
                bbox.y = coords.y - dragOffsetY;

                // Keep within canvas bounds
                bbox.x = Math.max(0, Math.min(canvas.width - bbox.width, bbox.x));
                bbox.y = Math.max(0, Math.min(canvas.height - bbox.height, bbox.y));

                redrawCanvas();
                return;
            }

            // Handle resizing
            if (isResizing) {
                const bbox = currentBboxes[selectedBboxIndex];
                const dx = coords.x - startX;
                const dy = coords.y - startY;

                if (resizeHandle === 'tl') {
                    bbox.x += dx;
                    bbox.y += dy;
                    bbox.width -= dx;
                    bbox.height -= dy;
                } else if (resizeHandle === 'tr') {
                    bbox.y += dy;
                    bbox.width += dx;
                    bbox.height -= dy;
                } else if (resizeHandle === 'bl') {
                    bbox.x += dx;
                    bbox.width -= dx;
                    bbox.height += dy;
                } else if (resizeHandle === 'br') {
                    bbox.width += dx;
                    bbox.height += dy;
                }

                // Keep minimum size
                if (bbox.width < 20) bbox.width = 20;
                if (bbox.height < 20) bbox.height = 20;

                startX = coords.x;
                startY = coords.y;

                redrawCanvas();
                return;
            }

            // Handle drawing new box
            if (!isDrawing) return;

            const currentX = coords.x;
            const currentY = coords.y;

            redrawCanvas();

            // Draw current box
            const width = currentX - startX;
            const height = currentY - startY;

            const color = COLORS[config.classes.indexOf(selectedClass) % COLORS.length];
            ctx.strokeStyle = color;
            ctx.lineWidth = 12;
            ctx.setLineDash([20, 20]);
            ctx.strokeRect(startX, startY, width, height);
            ctx.setLineDash([]);
        }

        function stopDrawing(e) {
            canvas.style.cursor = 'crosshair';

            // Handle drag/resize completion
            if (isDragging || isResizing) {
                isDragging = false;
                isResizing = false;
                resizeHandle = null;
                saveCurrentAnnotation();
                redrawCanvas();
                updateBboxList();
                return;
            }

            // Handle new box completion
            if (!isDrawing) return;
            isDrawing = false;

            const coords = getCanvasCoordinates(e);
            const endX = coords.x;
            const endY = coords.y;

            const width = endX - startX;
            const height = endY - startY;

            // Minimum box size
            if (Math.abs(width) > 10 && Math.abs(height) > 10) {
                // Normalize coordinates (handle negative width/height)
                const x = Math.min(startX, endX);
                const y = Math.min(startY, endY);
                const w = Math.abs(width);
                const h = Math.abs(height);

                currentBboxes.push({
                    class: selectedClass,
                    x: x,
                    y: y,
                    width: w,
                    height: h
                });

                saveCurrentAnnotation();
                redrawCanvas();
                updateBboxList();
            }
        }

        function updateCursor(e) {
            if (isDrawing || isDragging || isResizing) return;

            const coords = getCanvasCoordinates(e);

            // Check for resize handles
            for (let i = currentBboxes.length - 1; i >= 0; i--) {
                const bbox = currentBboxes[i];
                const handle = getResizeHandle(coords.x, coords.y, bbox);

                if (handle) {
                    if (handle === 'tl' || handle === 'br') {
                        canvas.style.cursor = 'nwse-resize';
                    } else {
                        canvas.style.cursor = 'nesw-resize';
                    }
                    return;
                }

                // Check if inside bbox
                if (isInsideBbox(coords.x, coords.y, bbox)) {
                    canvas.style.cursor = 'move';
                    return;
                }
            }

            canvas.style.cursor = 'crosshair';
        }

        function deleteBbox(index) {
            currentBboxes.splice(index, 1);
            if (selectedBboxIndex === index) {
                selectedBboxIndex = -1;
            } else if (selectedBboxIndex > index) {
                selectedBboxIndex--;
            }
            saveCurrentAnnotation();
            redrawCanvas();
            updateBboxList();
        }

        function updateBboxList() {
            const list = document.getElementById('bboxList');
            const count = document.getElementById('bboxCount');

            count.textContent = currentBboxes.length;
            list.innerHTML = '';

            currentBboxes.forEach((bbox, idx) => {
                const div = document.createElement('div');
                div.className = 'bbox-item';
                div.innerHTML = `
                    <div class="bbox-info">
                        <div class="bbox-class">${bbox.class}</div>
                        <div class="bbox-coords">
                            x:${Math.round(bbox.x)}, y:${Math.round(bbox.y)},
                            w:${Math.round(bbox.width)}, h:${Math.round(bbox.height)}
                        </div>
                    </div>
                    <button class="bbox-delete" onclick="deleteBbox(${idx})">Delete</button>
                `;
                list.appendChild(div);
            });
        }

        async function saveCurrentAnnotation() {
            if (!images[currentIndex]) return;

            const imageName = images[currentIndex];

            try {
                await fetch(`/api/annotation/${imageName}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({bboxes: currentBboxes})
                });
                updateStats();
            } catch (error) {
                console.error('Failed to save annotation:', error);
            }
        }

        async function saveAll() {
            try {
                const res = await fetch('/api/save-all', {method: 'POST'});
                const data = await res.json();
                alert(data.message);
            } catch (error) {
                console.error('Failed to save:', error);
                alert('Failed to save annotations');
            }
        }

        async function updateStats() {
            try {
                const res = await fetch('/api/annotations');
                const data = await res.json();
                const annotated = Object.keys(data.images).length;
                const total = images.length;
                document.getElementById('stats').textContent =
                    `📊 Annotated: ${annotated} / ${total} images`;
            } catch (error) {
                console.error('Failed to update stats:', error);
            }
        }

        function previousImage() {
            if (currentIndex > 0) {
                loadImage(currentIndex - 1);
            }
        }

        function nextImage() {
            if (currentIndex < images.length - 1) {
                loadImage(currentIndex + 1);
            }
        }

        function updateConfidence() {
            const slider = document.getElementById('confidenceSlider');
            const valueDisplay = document.getElementById('confidenceValue');
            confidenceThreshold = parseInt(slider.value) / 100;
            valueDisplay.textContent = slider.value + '%';
        }

        async function reloadDetections() {
            if (liveDetectionEnabled) {
                await loadDetections();
            }
        }

        async function toggleLiveDetection() {
            const checkbox = document.getElementById('liveDetectionCheckbox');
            const statusDiv = document.getElementById('detectionStatus');
            const sliderContainer = document.getElementById('confidenceSliderContainer');

            liveDetectionEnabled = checkbox.checked;

            if (liveDetectionEnabled) {
                if (!config.model_available) {
                    statusDiv.style.display = 'block';
                    statusDiv.textContent = '⚠️ Model not found. Train first: python train.py';
                    statusDiv.style.color = '#e74c3c';
                    checkbox.checked = false;
                    liveDetectionEnabled = false;
                    return;
                }

                sliderContainer.style.display = 'block';
                statusDiv.style.display = 'block';
                statusDiv.textContent = 'Loading detections...';
                statusDiv.style.color = '#666';
                await loadDetections();
            } else {
                sliderContainer.style.display = 'none';
                detectionOverlayImage = null;
                statusDiv.style.display = 'none';
                redrawCanvas();
            }
        }

        async function loadDetections() {
            if (!liveDetectionEnabled || !images[currentIndex]) return;

            const statusDiv = document.getElementById('detectionStatus');

            try {
                const imageName = images[currentIndex];
                const res = await fetch(`/api/predict/${imageName}?conf=${confidenceThreshold}`);
                const data = await res.json();

                if (data.image && liveDetectionEnabled) {
                    // Create image from base64
                    const img = new Image();
                    img.onload = () => {
                        if (liveDetectionEnabled) {
                            detectionOverlayImage = img;
                            redrawCanvas();
                            statusDiv.textContent = `Detected ${data.predictions?.length || 0} objects`;
                            statusDiv.style.color = '#27ae60';
                        }
                    };
                    img.src = `data:image/png;base64,${data.image}`;
                } else {
                    statusDiv.textContent = data.error || 'No detections';
                    statusDiv.style.color = '#e74c3c';
                }
            } catch (error) {
                console.error('Failed to load detections:', error);
                statusDiv.textContent = 'Error loading detections';
                statusDiv.style.color = '#e74c3c';
            }
        }

        async function showPredictions() {
            const overlay = document.getElementById('predictionOverlay');
            overlay.classList.add('visible');
            overlay.innerHTML = '<div class="loading-spinner"></div>';

            try {
                const imageName = images[currentIndex];
                const res = await fetch(`/api/predict/${imageName}`);
                const data = await res.json();

                if (data.image) {
                    overlay.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Predictions">`;

                    // Auto-hide after 5 seconds
                    setTimeout(() => {
                        overlay.classList.remove('visible');
                    }, 5000);
                } else {
                    overlay.innerHTML = '<div style="color: red;">' + (data.error || 'Failed to load predictions') + '</div>';
                }
            } catch (error) {
                console.error('Failed to load predictions:', error);
                overlay.innerHTML = '<div style="color: red;">Error loading predictions</div>';
            }
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ignore if typing in input field
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            const key = e.key.toLowerCase();

            if (e.key === 'ArrowLeft' || key === 'p') {
                e.preventDefault();
                previousImage();
            } else if (e.key === 'ArrowRight' || key === 'n') {
                e.preventDefault();
                nextImage();
            } else if (e.key === 'Escape') {
                // Hide prediction overlay or deselect box
                document.getElementById('predictionOverlay').classList.remove('visible');
                if (selectedBboxIndex !== -1) {
                    selectedBboxIndex = -1;
                    redrawCanvas();
                }
            } else if (e.key >= '1' && e.key <= '9') {
                // Quick class selection
                e.preventDefault();
                const idx = parseInt(e.key) - 1;
                if (idx < config.classes.length) {
                    selectClass(config.classes[idx]);
                }
            } else if (e.key === 'Delete' || e.key === 'Backspace') {
                // Delete selected bbox
                if (selectedBboxIndex !== -1) {
                    e.preventDefault();
                    deleteBbox(selectedBboxIndex);
                }
            }
        });

        // Start the app
        init();
    </script>
</body>
</html>
"""

    with open(templates_dir / "index.html", 'w') as f:
        f.write(html_content)


# Training Management
import subprocess
import threading
import time
import uuid

# Store training jobs
training_jobs = {}

@app.route('/api/train/start', methods=['POST'])
def start_training():
    """Start a training job in the background"""
    data = request.json
    project_id = data.get('project_id')
    epochs = data.get('epochs', 300)
    batch_size = data.get('batch_size', 4)
    model_size = data.get('model_size', 'm')
    device = data.get('device', 'mps')

    # Map model size to depth/width
    model_configs = {
        's': {'depth': 0.33, 'width': 0.375},
        'm': {'depth': 0.67, 'width': 0.75},
        'l': {'depth': 1.0, 'width': 1.0}
    }

    config = model_configs.get(model_size, model_configs['m'])

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Create log file
    log_file = Path(f"training_logs/{job_id}.log")
    log_file.parent.mkdir(exist_ok=True)

    # Build training command
    # Note: Removed --no-resume to allow continuing from previous training
    cmd = [
        'python3', 'train.py',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--depth', str(config['depth']),
        '--width', str(config['width']),
        '--device', device
    ]

    # Store job info
    training_jobs[job_id] = {
        'project_id': project_id,
        'status': 'running',
        'log_file': str(log_file),
        'process': None,
        'start_time': time.time()
    }

    def run_training():
        """Run training in a separate thread"""
        try:
            # Set environment variable for project selection
            env = os.environ.copy()
            env['TRAINING_PROJECT_ID'] = project_id

            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=os.getcwd()
                )
                training_jobs[job_id]['process'] = process
                returncode = process.wait()

                if returncode == 0:
                    training_jobs[job_id]['status'] = 'completed'
                else:
                    training_jobs[job_id]['status'] = 'failed'

        except Exception as e:
            training_jobs[job_id]['status'] = 'failed'
            with open(log_file, 'a') as f:
                f.write(f"\n\nError: {str(e)}\n")

    # Start training thread
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return jsonify({
        'status': 'success',
        'job_id': job_id,
        'message': 'Training started'
    })


@app.route('/api/train/status/<job_id>')
def get_training_status(job_id):
    """Get status of a training job"""
    if job_id not in training_jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

    job = training_jobs[job_id]
    log_content = ''

    # Read log file
    if Path(job['log_file']).exists():
        with open(job['log_file'], 'r') as f:
            log_content = f.read()

    return jsonify({
        'status': 'success',
        'training_status': job['status'],
        'log': log_content,
        'start_time': job['start_time']
    })


@app.route('/api/train/stop/<job_id>', methods=['POST'])
def stop_training(job_id):
    """Stop a training job"""
    if job_id not in training_jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

    job = training_jobs[job_id]

    if job['process']:
        job['process'].terminate()
        job['status'] = 'stopped'

    return jsonify({
        'status': 'success',
        'message': 'Training stopped'
    })


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 YOLO Image Annotation Tool")
    print("=" * 60)

    # Migrate legacy data if exists
    migrated_project = migrate_legacy_data()
    if migrated_project:
        print(f"✓ Migrated existing data to 'Legacy Project'")
        print("=" * 60)

    # Create HTML template
    create_html_template()

    # Check for model for active project
    active_project = get_active_project()
    if active_project:
        model_paths = get_model_paths(active_project['id'])
        if model_paths['best'].exists():
            print(f"   ✓ YOLOX model found for {active_project['name']}")
            print("     Live Detection and Predictions available")
        elif model_paths['latest'].exists():
            print(f"   ✓ YOLOX model found for {active_project['name']} (training checkpoint)")
            print("     Live Detection and Predictions available")
        else:
            print(f"   ℹ No trained model for {active_project['name']} (train first: python3 train.py)")
    else:
        print("   ℹ No active project. Create one to start training.")

    # Show active project info
    print("\n📂 Projects:")
    projects_data = load_projects()
    if projects_data['projects']:
        print(f"   Total projects: {len(projects_data['projects'])}")
        active_project = get_active_project()
        if active_project:
            print(f"   Active: {active_project['name']} ({active_project['id']})")
        else:
            print("   No active project selected")
    else:
        print("   No projects created yet")
    print("   Manage projects at: http://localhost:8100/projects")

    print("\n🌐 Starting web server...")
    print("   Open your browser and go to: http://localhost:8100")
    print("\n   Press Ctrl+C to stop the server")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=8100)
