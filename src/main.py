import torch
import os
import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pathlib import Path

from data.dataloader import VOCDataModule, COCODataModule, CUB200DataModule
from utils.argparser import get_parser, write_config_file
from models.classifier import VGG16ClassifierModel, Resnet50ClassifierModel
from models.explainer_classifier import ExplainerClassifierModel
from models.interpretable_fcnn import InterpretableFCNN
from models.explainer_classifier_rtsal import RTSalExplainerClassifierModel

main_dir = Path(os.path.dirname(os.path.abspath(__file__)))

parser = get_parser()
args = parser.parse_args()
if args.arg_log:
    write_config_file(args)

pl.seed_everything(args.seed)

# Set up Logging
if args.use_tensorboard_logger:
    log_dir = "tb_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = pl.loggers.TensorBoardLogger(log_dir, name="NN Explainer")
else:
    logger = False

# Set up data module
if args.dataset == "VOC":
    data_path = main_dir / args.data_base_path / 'VOC2007'
    data_module = VOCDataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation
    )
    num_classes = 20
elif args.dataset == "COCO":
    data_path = main_dir / args.data_base_path / 'COCO2014'
    data_module = COCODataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation
    )
    num_classes = 91
elif args.dataset == "CUB":
    data_path = main_dir / args.data_base_path / 'CUB200'
    data_module = CUB200DataModule(
        data_path=data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size, use_data_augmentation=args.use_data_augmentation
    )
    num_classes = 200
else:
    raise Exception("Unknown dataset " + args.dataset)

# Set up model
if args.model_to_train == "explainer":
    model = ExplainerClassifierModel(
        num_classes=num_classes, dataset=args.dataset, classifier_type=args.classifier_type, classifier_checkpoint=args.classifier_checkpoint, fix_classifier=args.fix_classifier, learning_rate=args.learning_rate, 
        class_mask_min_area=args.class_mask_min_area, class_mask_max_area=args.class_mask_max_area, entropy_regularizer=args.entropy_regularizer, use_mask_variation_loss=args.use_mask_variation_loss, 
        mask_variation_regularizer=args.mask_variation_regularizer, use_mask_area_loss=args.use_mask_area_loss, mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, 
        mask_total_area_regularizer=args.mask_total_area_regularizer, ncmask_total_area_regularizer=args.ncmask_total_area_regularizer, metrics_threshold=args.metrics_threshold, 
        save_masked_images=args.save_masked_images, save_masks=args.save_masks,
        save_all_class_masks=args.save_all_class_masks, save_path=args.save_path
    )

    if args.explainer_classifier_checkpoint is not None:
        model = model.load_from_checkpoint(
            args.explainer_classifier_checkpoint,
            num_classes=num_classes, dataset=args.dataset, classifier_type=args.classifier_type, classifier_checkpoint=args.classifier_checkpoint, fix_classifier=args.fix_classifier, learning_rate=args.learning_rate, 
            class_mask_min_area=args.class_mask_min_area, class_mask_max_area=args.class_mask_max_area, entropy_regularizer=args.entropy_regularizer, use_mask_variation_loss=args.use_mask_variation_loss, 
            mask_variation_regularizer=args.mask_variation_regularizer, use_mask_area_loss=args.use_mask_area_loss, mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, 
            mask_total_area_regularizer=args.mask_total_area_regularizer, ncmask_total_area_regularizer=args.ncmask_total_area_regularizer, metrics_threshold=args.metrics_threshold, 
            save_masked_images=args.save_masked_images, save_masks=args.save_masks, save_all_class_masks=args.save_all_class_masks, save_path=args.save_path
        )
elif args.model_to_train == "classifier":
    if args.classifier_type == "vgg16":
        model = VGG16ClassifierModel(
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, use_imagenet_pretraining=args.use_imagenet_pretraining, 
            fix_classifier_backbone=args.fix_classifier_backbone, metrics_threshold=args.metrics_threshold
        )
    elif args.classifier_type == "resnet50":
        model = Resnet50ClassifierModel(
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, use_imagenet_pretraining=args.use_imagenet_pretraining, 
            fix_classifier_backbone=args.fix_classifier_backbone, metrics_threshold=args.metrics_threshold
        )
    else:
        raise Exception("Unknown classifier type " + args.classifier_type)

    if args.classifier_checkpoint is not None:
        model = model.load_from_checkpoint(
            args.classifier_checkpoint,
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, use_imagenet_pretraining=args.use_imagenet_pretraining, 
            fix_classifier_backbone=args.fix_classifier_backbone, metrics_threshold=args.metrics_threshold
        )
elif args.model_to_train == "fcnn":
    model = InterpretableFCNN(
        num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, class_mask_min_area=args.class_mask_min_area, class_mask_max_area=args.class_mask_max_area, 
        use_mask_coherency_loss=args.use_mask_coherency_loss, use_mask_variation_loss=args.use_mask_variation_loss, mask_variation_regularizer=args.mask_variation_regularizer, 
        use_mask_area_loss=args.use_mask_area_loss, mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, mask_total_area_regularizer=args.mask_total_area_regularizer, 
        ncmask_total_area_regularizer=args.ncmask_total_area_regularizer, metrics_threshold=args.metrics_threshold,
        save_masked_images=args.save_masked_images, save_masks=args.save_masks, save_all_class_masks=args.save_all_class_masks, save_path=args.save_path
    )
    if args.fcnn_checkpoint is not None:
        model = model.load_from_checkpoint(
            args.fcnn_checkpoint,
            num_classes=num_classes, dataset=args.dataset, learning_rate=args.learning_rate, class_mask_min_area=args.class_mask_min_area, class_mask_max_area=args.class_mask_max_area, 
            use_mask_coherency_loss=args.use_mask_coherency_loss, use_mask_variation_loss=args.use_mask_variation_loss, mask_variation_regularizer=args.mask_variation_regularizer, 
            use_mask_area_loss=args.use_mask_area_loss, mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, mask_total_area_regularizer=args.mask_total_area_regularizer, 
            ncmask_total_area_regularizer=args.ncmask_total_area_regularizer, metrics_threshold=args.metrics_threshold, save_masked_images=args.save_masked_images,
            save_masks=args.save_masks, save_all_class_masks=args.save_all_class_masks, save_path=args.save_path
        )
elif args.model_to_train == "rtsal_explainer":
    model = RTSalExplainerClassifierModel(
        num_classes=num_classes, dataset=args.dataset, classifier_type=args.classifier_type, 
        classifier_checkpoint=args.classifier_checkpoint, fix_classifier=args.fix_classifier,
        learning_rate=args.learning_rate, metrics_threshold=args.metrics_threshold, 
        save_masked_images=args.save_masked_images, save_masks=args.save_masks, 
        save_all_class_masks=args.save_all_class_masks, save_path=args.save_path
    )
    if args.explainer_classifier_checkpoint is not None:
        model = model.load_from_checkpoint(
            args.explainer_classifier_checkpoint,
            num_classes=num_classes, dataset=args.dataset, classifier_type=args.classifier_type, 
            classifier_checkpoint=args.classifier_checkpoint, fix_classifier=args.fix_classifier,
            learning_rate=args.learning_rate, metrics_threshold=args.metrics_threshold, 
            save_masked_images=args.save_masked_images, save_masks=args.save_masks, 
            save_all_class_masks=args.save_all_class_masks, save_path=args.save_path
        )
else:
    raise Exception("Unknown model type " + args.model_to_train)

# Define Early Stopping condition
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=args.early_stop_min_delta,
    patience=args.early_stop_patience,
    verbose=False,
    mode="min",
)

trainer = pl.Trainer(
    logger = logger,
    callbacks = [early_stop_callback],
    gpus = [0] if torch.cuda.is_available() else 0,
    terminate_on_nan = True,
    checkpoint_callback = args.checkpoint_callback,
)

if args.train_model:
    trainer.fit(model=model, datamodule=data_module)
    trainer.test()
else:
    trainer.test(model=model, datamodule=data_module)
