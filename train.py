#!/usr/bin/env python3
"""
Training script for YOLOX object detection (Apache 2.0 License)
Trains YOLOX models on custom bounding box annotations
"""

import argparse
import os
import random
import warnings
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from loguru import logger

from yolox.core import launch
from yolox.exp import Exp, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices

from yolox_dataset import CustomDataset
from yolox.data import DataLoader, TrainTransform, ValTransform


class CustomExp(Exp):
    """Custom experiment configuration for our dataset"""

    def __init__(self, args):
        super().__init__()

        # Dataset paths
        self.annotations_file = args.annotations

        # Model configuration
        self.num_classes = None  # Will be set from dataset
        self.depth = args.depth
        self.width = args.width

        # Training settings
        self.max_epoch = args.epochs
        self.warmup_epochs = 5
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        # Data augmentation (standard intensity, higher frequency, more scale variation)
        self.mosaic_prob = 1.0  # Use mosaic 100% of the time
        self.mixup_prob = 1.0   # Use mixup 100% of the time
        self.hsv_prob = 1.0     # Adjust HSV (brightness/saturation) 100% of time
        self.flip_prob = 0.5    # Horizontal flip 50%
        self.degrees = 15.0     # Max rotation ±15 degrees
        self.translate = 0.1    # Max translation 10%
        self.mosaic_scale = (0.1, 4.0)  # Wider scale range: 0.1x to 4x
        self.mixup_scale = (0.5, 2.0)   # Mixup scale: 0.5x to 2x
        self.shear = 2.0        # Standard shear
        self.perspective = 0.0  # No perspective transform
        self.enable_mixup = True

        # Input settings
        self.input_size = (args.img_size, args.img_size)
        self.test_size = (args.img_size, args.img_size)
        self.random_size = (14, 26)
        self.multiscale_range = 5

        # Batch settings (0 workers to prevent memory leaks)
        self.data_num_workers = 0  # No workers to prevent multiprocess memory leaks
        self.eval_interval = 10
        self.print_interval = 10

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        """
        Get training dataloader
        """
        from yolox.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        dataset = CustomDataset(
            annotations_file=self.annotations_file,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            split='train'
        )

        # Set num_classes from dataset
        if self.num_classes is None:
            self.num_classes = dataset.num_classes
            logger.info(f"Detected {self.num_classes} classes: {dataset.class_names}")

        # Apply mosaic augmentation
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        # Disable pin_memory for MPS to save memory
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": False}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        """
        Get validation dataloader
        """
        from yolox.data import ValTransform

        valdataset = CustomDataset(
            annotations_file=self.annotations_file,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            split='val'
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": False,  # Disable for MPS to save memory
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """
        Get evaluator for validation
        """
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=0.01,
            nmsthre=0.65,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")

    # Dataset
    parser.add_argument('--annotations', type=str, default='data/annotations/annotations.json',
                        help='Path to annotations file')

    # Model architecture
    parser.add_argument('--depth', type=float, default=0.67,
                        help='Model depth (0.33=s, 0.67=m, 1.0=l, 1.33=x)')
    parser.add_argument('--width', type=float, default=0.75,
                        help='Model width (0.375=s, 0.75=m, 1.0=l, 1.25=x)')

    # Training
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size (4 for YOLOX-M, 8 for YOLOX-S)")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")

    # Device
    parser.add_argument("-d", "--devices", default=None, type=int, help="Device for training")
    parser.add_argument("--device", default="mps", type=str, help="Device to use (cpu, mps, cuda)")

    # Experiment/output
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="Experiment file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="Checkpoint file")
    parser.add_argument("--resume", default=True, action="store_true", help="Resume training from latest checkpoint (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start training from scratch")
    parser.add_argument("-o", "--occupy", dest="occupy", default=False, action="store_true",
                        help="Occupy GPU memory first for training")
    parser.add_argument("-n", "--name", type=str, default="yolox_custom", help="Model name")
    parser.add_argument("--logger", type=str, default="tensorboard", help="Logger type")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")

    # Distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="Distributed backend")
    parser.add_argument("--dist-url", default=None, type=str,
                        help="URL used to set up distributed training")
    parser.add_argument("--num_machines", default=1, type=int, help="Number of machines")
    parser.add_argument("--machine_rank", default=0, type=int, help="Machine rank")

    # FP16
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",
                        help="Adopting mix precision training")

    # Cache
    parser.add_argument("-cache", dest="cache", default=False, action="store_true",
                        help="Cache images for faster training")

    return parser


@logger.catch
def main(exp, args):
    """Main training function"""

    # Set up output directory
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🍎 Using Apple Silicon MPS acceleration")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🎮 Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("💻 Using CPU")

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)
    cudnn.benchmark = True

    # Get data loaders (this also sets num_classes in exp)
    train_loader = exp.get_data_loader(
        batch_size=args.batch_size,
        is_distributed=False,
        no_aug=False,
        cache_img=args.cache
    )

    # Create model after num_classes is set
    model = exp.get_model()
    logger.info(f"Model summary: {model}")
    model.to(device)

    eval_loader = exp.get_eval_loader(
        batch_size=args.batch_size,
        is_distributed=False
    )

    # Training loop with proper loss computation and progress tracking
    from yolox.utils import setup_logger, MeterBuffer
    import torch.optim as optim
    from torch.cuda.amp import GradScaler
    import time

    # Setup output directory
    file_name = os.path.join(output_dir, args.name)
    os.makedirs(file_name, exist_ok=True)
    setup_logger(file_name, distributed_rank=0, filename="train_log.txt", mode="a")

    logger.info(f"Training in directory: {file_name}")

    # Create optimizer with cosine learning rate schedule
    base_lr = exp.basic_lr_per_img * args.batch_size
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=base_lr * 0.05
    )

    # GradScaler for mixed precision (disabled for MPS)
    scaler = GradScaler() if device.type == 'cuda' and args.fp16 else None

    # Training metrics
    meter = MeterBuffer(window_size=10)
    best_ap = 0
    start_epoch = 0

    # Check for existing checkpoint to resume from (auto-resume if exists)
    latest_file = os.path.join(file_name, "latest_ckpt.pth")
    if os.path.exists(latest_file) and args.resume:
        logger.info(f"✓ Resuming from checkpoint: {latest_file}")
        ckpt = torch.load(latest_file, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        if 'best_ap' in ckpt:
            best_ap = ckpt['best_ap']

        # Ensure at least 100 more epochs to train
        remaining_epochs = args.epochs - start_epoch
        if remaining_epochs < 100:
            args.epochs = start_epoch + 100
            logger.info(f"  Extending training to {args.epochs} epochs (minimum 100 more)")

        logger.info(f"  Continuing from epoch {start_epoch + 1}/{args.epochs}, best_ap: {best_ap:.4f}")
    elif os.path.exists(latest_file) and not args.resume:
        logger.info(f"ℹ Found existing checkpoint, but --no-resume set. Starting from epoch 1.")
        logger.info(f"  Training will continue writing to same directory.")

    logger.info("Starting training...")
    logger.info(f"Initial learning rate: {base_lr:.6f}")
    if start_epoch > 0:
        logger.info(f"Resuming from epoch {start_epoch + 1}/{args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()

        # Get actual number of iterations
        max_iter = len(train_loader.dataset) // args.batch_size

        # Suppress warnings during training iterations to keep output clean
        warnings.filterwarnings('ignore')

        for iter_i, (imgs, targets, img_info, img_id) in enumerate(train_loader):
            # Break at end of epoch
            if iter_i >= max_iter:
                break
            iter_start = time.time()

            # Move to device
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Forward pass (YOLOX returns dict of losses when training)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs, targets=targets)
            else:
                outputs = model(imgs, targets=targets)

            # Extract losses from dict
            loss = outputs['total_loss']
            iou_loss = outputs['iou_loss']
            conf_loss = outputs['conf_loss']
            cls_loss = outputs['cls_loss']
            l1_loss = outputs['l1_loss']
            num_fg = outputs['num_fg']

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Extract scalar values BEFORE deleting tensors
            loss_val = loss.item() if hasattr(loss, 'item') else loss
            iou_loss_val = iou_loss.item() if hasattr(iou_loss, 'item') else iou_loss
            conf_loss_val = conf_loss.item() if hasattr(conf_loss, 'item') else conf_loss
            cls_loss_val = cls_loss.item() if hasattr(cls_loss, 'item') else cls_loss
            l1_loss_val = l1_loss.item() if hasattr(l1_loss, 'item') else l1_loss
            num_fg_val = num_fg if not hasattr(num_fg, 'item') else num_fg.item()

            # Delete ALL tensors and variables to free memory
            del outputs, loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg
            del imgs, targets, img_info, img_id

            # Clear gradients to free memory
            optimizer.zero_grad(set_to_none=True)

            # Empty cache every iteration for MPS
            if device.type == 'mps':
                torch.mps.empty_cache()

            # Aggressive memory cleanup for MPS
            if device.type == 'mps':
                # Force immediate cleanup every 5 iterations
                if iter_i % 5 == 0:
                    import gc
                    gc.collect()
                    torch.mps.empty_cache()
                    # Synchronize to ensure operations complete
                    torch.mps.synchronize()

            # Update metrics with scalar values
            iter_time = time.time() - iter_start
            meter.update(
                iter_time=iter_time,
                data_time=0,
                loss=loss_val,
                iou_loss=iou_loss_val,
                conf_loss=conf_loss_val,
                cls_loss=cls_loss_val,
                l1_loss=l1_loss_val,
                lr=optimizer.param_groups[0]['lr'],
                num_fg=num_fg_val
            )

            # Calculate progress
            progress_pct = (iter_i + 1) / max_iter * 100
            eta_seconds = meter['iter_time'].global_avg * (max_iter - iter_i - 1)
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"

            # Create visual progress bar
            bar_width = 30
            filled = int(bar_width * (iter_i + 1) / max_iter)
            bar = '█' * filled + '░' * (bar_width - filled)

            # Build progress message
            progress_msg = (
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"[{bar}] {iter_i + 1}/{max_iter} ({progress_pct:.0f}%) | "
                f"loss: {meter['loss'].latest:.2f} "
                f"(iou: {meter['iou_loss'].latest:.2f}, "
                f"conf: {meter['conf_loss'].latest:.2f}, "
                f"cls: {meter['cls_loss'].latest:.2f}) | "
                f"eta: {eta_str}"
            )

            # Update same line with carriage return
            import sys
            sys.stdout.write(f"\r{progress_msg}")
            sys.stdout.flush()

        # Clear the progress bar line before printing epoch summary
        import sys
        sys.stdout.write('\r' + ' ' * 150 + '\r')  # Clear line with spaces
        sys.stdout.flush()

        # Epoch finished - save epoch stats before resetting meter
        epoch_time = time.time() - epoch_start
        avg_loss = meter['loss'].avg
        avg_iou = meter['iou_loss'].avg
        avg_conf = meter['conf_loss'].avg
        avg_cls = meter['cls_loss'].avg

        # Reset meter to prevent memory accumulation
        meter = MeterBuffer(window_size=10)

        # Aggressive memory cleanup after each epoch
        if device.type == 'mps':
            import gc
            # Multiple passes of garbage collection
            for _ in range(3):
                gc.collect()
            torch.mps.empty_cache()
            torch.mps.synchronize()
            # Small sleep to allow OS to reclaim memory
            import time as time_module
            time_module.sleep(0.1)

        # Log epoch summary
        logger.info(f"Epoch [{epoch + 1}/{args.epochs}] completed in {epoch_time:.1f}s")
        logger.info(
            f"  Avg loss: {avg_loss:.3f} "
            f"(iou: {avg_iou:.3f}, "
            f"conf: {avg_conf:.3f}, "
            f"cls: {avg_cls:.3f})"
        )

        # Update learning rate
        lr_scheduler.step()

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            ckpt_file = os.path.join(file_name, f"epoch_{epoch + 1}_ckpt.pth")
            logger.info(f"Saving checkpoint to {ckpt_file}")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_ap': best_ap,
                'model_config': {
                    'depth': args.depth,
                    'width': args.width,
                    'num_classes': exp.num_classes
                }
            }, ckpt_file)

        # Always save latest
        latest_file = os.path.join(file_name, "latest_ckpt.pth")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_ap': best_ap,
            'model_config': {
                'depth': args.depth,
                'width': args.width,
                'num_classes': exp.num_classes
            }
        }, latest_file)

    # Save final model as best
    best_file = os.path.join(file_name, "best_ckpt.pth")
    torch.save({
        'model': model.state_dict(),
        'model_config': {
            'depth': args.depth,
            'width': args.width,
            'num_classes': exp.num_classes
        }
    }, best_file)

    logger.info(f"✓ Training complete!")
    logger.info(f"  Best model saved to: {best_file}")
    logger.info(f"  Latest checkpoint: {latest_file}")


if __name__ == "__main__":
    # Configure logger format at the very start
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>\n",
        level="INFO"
    )

    args = make_parser().parse_args()

    # Load projects and select project
    import json
    import os
    projects_file = Path("data/projects.json")
    if not projects_file.exists():
        logger.error(f"❌ Projects file not found: {projects_file}")
        logger.error("   Please create a project first using the web interface")
        exit(1)

    with open(projects_file) as f:
        projects_data = json.load(f)

    projects = projects_data.get('projects', [])
    if not projects:
        logger.error("❌ No projects found")
        logger.error("   Please create a project first using the web interface")
        exit(1)

    # Check if project ID is provided via environment variable (from web interface)
    env_project_id = os.environ.get('TRAINING_PROJECT_ID')

    if env_project_id:
        # Use project from environment variable
        selected_project = next((p for p in projects if p['id'] == env_project_id), None)
        if not selected_project:
            logger.error(f"❌ Project with ID {env_project_id} not found")
            exit(1)
        logger.info(f"✅ Training project: {selected_project['name']}")
    elif len(projects) > 1:
        # If multiple projects, let user choose
        logger.info("📁 Available projects:")
        for i, proj in enumerate(projects):
            logger.info(f"   {i+1}. {proj['name']} (ID: {proj['id']})")

        while True:
            try:
                choice = input("\nSelect project number (or press Enter for active project): ").strip()
                if not choice:
                    # Use active project
                    project_id = projects_data.get('active_project')
                    if not project_id:
                        logger.error("No active project set")
                        continue
                    selected_project = next((p for p in projects if p['id'] == project_id), None)
                    if not selected_project:
                        logger.error("Active project not found")
                        continue
                    break
                else:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(projects):
                        selected_project = projects[choice_idx]
                        break
                    else:
                        logger.error(f"Invalid choice. Please enter 1-{len(projects)}")
            except ValueError:
                logger.error("Invalid input. Please enter a number")
        logger.info(f"✅ Selected project: {selected_project['name']}")
    else:
        selected_project = projects[0]
        logger.info(f"✅ Training project: {selected_project['name']}")

    project_id = selected_project['id']

    # Update paths to use project-specific directories
    project_dir = Path("projects") / project_id
    args.annotations = str(project_dir / "annotations.json")
    args.output_dir = str(project_dir / "output")

    # Check annotations
    if not Path(args.annotations).exists():
        logger.error(f"❌ Annotations file not found: {args.annotations}")
        logger.error("   Please annotate some images first using app.py")
        exit(1)

    # Create experiment
    exp = CustomExp(args)

    # Add experiment_name to args for Trainer
    args.experiment_name = args.name

    logger.info("="*60)
    logger.info("🎯 YOLOX Training Pipeline (Apache 2.0 License)")
    logger.info("="*60)

    # Launch training
    if args.devices is None:
        args.devices = 1  # Single device

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
