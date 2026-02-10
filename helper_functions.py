import os
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from torch.cuda import amp

from main import classification_transform, segmentation_transform, DATASET_PATH, OilSpillClassificationDataset, \
    OilSpillSegmentationDataset, MobileNetV4, ConvNeXtV2, EfficientNetV3, ResNeXtV3, InternImage, RepLKNet, device, \
    evaluate_classification_model, create_umamba, create_vmunet, create_unetpp, create_multicbam, create_transunet, \
    evaluate_segmentation_model, visualize_segmentation_predictions, train_data, valid_data, OUTPUT_DIR

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9, 
            'reserved': torch.cuda.memory_reserved() / 1e9  
        }
    else:
        return {'allocated': 0, 'reserved': 0}


def reduce_model_size(model, input_shape=(3, 224, 224)):
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy_input = torch.randn(1, *input_shape).to(device)
    try:
        with amp.autocast():
            output = model(dummy_input)

        activation_estimate = 0
        for module in model.modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                activation_estimate += np.prod(module.weight.size()) * 4  

        memory_usage = {
            'parameters': total_params * 4 / 1e6,  
            'activations_estimate': activation_estimate / 1e6,  
            'total_estimate': (total_params * 4 + activation_estimate) / 1e6  
        }
    except Exception as e:
        memory_usage = {
            'parameters': total_params * 4 / 1e6,  
            'error': str(e)
        }

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'memory_usage': memory_usage
    }


def create_test_split():
    global test_data, test_cls_dataset, test_seg_dataset, test_cls_loader, test_seg_loader
    global val_cls_dataset, val_seg_dataset, val_cls_loader, val_seg_loader, valid_data

    from sklearn.model_selection import train_test_split

    print("\nCreating test split from validation data...")
    val_indices = list(range(len(valid_data["images"])))
    val_indices, test_indices = train_test_split(val_indices, test_size=0.5, random_state=42)

    test_data = {
        "images": [valid_data["images"][i] for i in test_indices],
        "annotations": [ann for ann in valid_data["annotations"]
                        if ann["image_id"] in [valid_data["images"][i]["id"] for i in test_indices]],
        "categories": valid_data["categories"]
    }

    valid_data_updated = {
        "images": [valid_data["images"][i] for i in val_indices],
        "annotations": [ann for ann in valid_data["annotations"]
                        if ann["image_id"] in [valid_data["images"][i]["id"] for i in val_indices]],
        "categories": valid_data["categories"]
    }

    test_cls_dataset = OilSpillClassificationDataset(
        test_data, DATASET_PATH, "valid", transform=classification_transform
    )
    test_seg_dataset = OilSpillSegmentationDataset(
        test_data, DATASET_PATH, "valid", transform=segmentation_transform
    )

    test_cls_loader = torch.utils.data.DataLoader(test_cls_dataset, batch_size=8, shuffle=False, num_workers=2)
    test_seg_loader = torch.utils.data.DataLoader(test_seg_dataset, batch_size=4, shuffle=False, num_workers=2)

    val_cls_dataset = OilSpillClassificationDataset(
        valid_data_updated, DATASET_PATH, "valid", transform=classification_transform
    )
    val_seg_dataset = OilSpillSegmentationDataset(
        valid_data_updated, DATASET_PATH, "valid", transform=segmentation_transform
    )

    val_cls_loader = torch.utils.data.DataLoader(val_cls_dataset, batch_size=8, shuffle=False, num_workers=2)
    val_seg_loader = torch.utils.data.DataLoader(val_seg_dataset, batch_size=4, shuffle=False, num_workers=2)

    print(f"- Test set created with {len(test_data['images'])} images")
    print(f"- Updated validation set has {len(valid_data_updated['images'])} images")


def evaluate_best_models(cls_results, seg_results):
    print("\n==== Evaluating Best Models on Test Set ====")

    best_cls_metrics = None
    best_seg_metrics = None

    if cls_results:
        best_cls_model_name = max(cls_results, key=lambda x: cls_results[x]['best_val_acc'])
        print(f"\nBest classification model: {best_cls_model_name}")

        if best_cls_model_name == "InternImage":
            best_cls_model = InternImage()
        elif best_cls_model_name == "MobileNetV4":
            best_cls_model = MobileNetV4()
        elif best_cls_model_name == "RepLKNet":
            best_cls_model = RepLKNet()
        elif best_cls_model_name == "ConvNeXtV2":
            best_cls_model = ConvNeXtV2()
        elif best_cls_model_name == "EfficientNetV3":
            best_cls_model = EfficientNetV3()
        elif best_cls_model_name == "ResNeXtV3":
            best_cls_model = ResNeXtV3()

        best_cls_model.load_state_dict(torch.load(f"{OUTPUT_DIR}/{best_cls_model_name}_best.pth"))
        best_cls_model = best_cls_model.to(device)

        cls_metrics = evaluate_classification_model(best_cls_model, test_cls_loader)
        print(f"Test Accuracy: {cls_metrics['accuracy']:.2f}%")
        print(f"Test Precision: {cls_metrics['precision']:.4f}")
        print(f"Test Recall: {cls_metrics['recall']:.4f}")
        print(f"Test F1 Score: {cls_metrics['f1_score']:.4f}")

        best_cls_metrics = cls_metrics

        del best_cls_model
        clear_gpu_memory()

    if seg_results:
        best_seg_model_name = max(seg_results, key=lambda x: seg_results[x]['best_val_iou'])
        print(f"\nBest segmentation model: {best_seg_model_name}")

        if best_seg_model_name == "UMamba":
            best_seg_model = create_umamba()
        elif best_seg_model_name == "VMUNet":
            best_seg_model = create_vmunet()
        elif best_seg_model_name == "MultiCBAM":
            best_seg_model = create_multicbam()
        elif best_seg_model_name == "UNetPlusPlus":
            best_seg_model = create_unetpp()
        elif best_seg_model_name == "TransUNet":
            best_seg_model = create_transunet()

        best_seg_model.load_state_dict(torch.load(f"{OUTPUT_DIR}/{best_seg_model_name}_best.pth"))
        best_seg_model = best_seg_model.to(device)

        seg_metrics = evaluate_segmentation_model(best_seg_model, test_seg_loader)
        print(f"Test Mean IoU: {seg_metrics['mean_iou']:.4f}")
        print(f"Test Median IoU: {seg_metrics['median_iou']:.4f}")

        best_seg_metrics = seg_metrics

        print("\nVisualizing segmentation predictions...")
        random_indices = np.random.choice(len(test_seg_dataset), min(3, len(test_seg_dataset)), replace=False)
        visualize_segmentation_predictions(best_seg_model, test_seg_dataset, random_indices)

        del best_seg_model
        clear_gpu_memory()

    test_results = {
        'classification': {
            'model': best_cls_model_name if 'best_cls_model_name' in locals() else None,
            'metrics': best_cls_metrics
        },
        'segmentation': {
            'model': best_seg_model_name if 'best_seg_model_name' in locals() else None,
            'metrics': best_seg_metrics
        }
    }

    with open(f"{OUTPUT_DIR}/test_results.json", 'w') as f:
        json.dump(test_results, f, indent=4, default=str)

    return test_results


def generate_final_report(cls_comparison, seg_comparison):
    print("\n==== Generating Final Report ====")

    report = f"""

- Dataset: Oil Spill Detection from Roboflow
- Total images: {len(train_data['images']) + len(valid_data['images']) + (len(test_data['images']) if 'test_data' in globals() and test_data else 0)}
- Training set: {len(train_data['images'])} images
- Validation set: {len(valid_data['images'])} images
- Test set: {len(test_data['images']) if 'test_data' in globals() and test_data else 0} images
- Classes: {[cat['name'] for cat in train_data['categories']]}

{cls_comparison.to_markdown()}

{seg_comparison.to_markdown()}


The best performing classification model was {cls_comparison.iloc[0]['Model']} with an accuracy of {cls_comparison.iloc[0]['Best Validation Accuracy']:.2f}%.

The best performing segmentation model was {seg_comparison.iloc[0]['Model']} with an IoU of {seg_comparison.iloc[0]['Best Validation IoU']:.4f}.

This project used memory-optimized training to reduce GPU memory consumption.
Memory-saving techniques applied:
- Individual model training
- Manual garbage collection between models
- Reduced batch sizes when needed
- Optional mixed precision training
"""

    with open(f"{OUTPUT_DIR}/final_report.md", 'w') as f:
        f.write(report)

    print(f"Final report saved to {OUTPUT_DIR}/final_report.md")


def estimate_model_memory_requirements():
   
    memory_reqs = {
        'classification': {},
        'segmentation': {}
    }

    cls_models = {
        "InternImage": InternImage(),
        "MobileNetV4": MobileNetV4(),
        "RepLKNet": RepLKNet(),
        "ConvNeXtV2": ConvNeXtV2(),
        "EfficientNetV3": EfficientNetV3(),
        "ResNeXtV3": ResNeXtV3()
    }

    for name, model in cls_models.items():
        model.to(device)
        memory_reqs['classification'][name] = reduce_model_size(model)
        model.cpu()
        del model
        clear_gpu_memory()

    seg_models = {
        "UMamba": create_umamba(),
        "VMUNet": create_vmunet(),
        "MultiCBAM": create_multicbam(),
        "UNetPlusPlus": create_unetpp(),
        "TransUNet": create_transunet()
    }

    for name, model in seg_models.items():
        model.to(device)
        memory_reqs['segmentation'][name] = reduce_model_size(model)
        model.cpu()
        del model
        clear_gpu_memory()

    print("\n===== Model Memory Requirements =====")

    print("\nClassification Models:")
    for name, info in memory_reqs['classification'].items():
        print(
            f"{name}: {info['total_parameters'] / 1e6:.2f}M params, ~{info['memory_usage']['parameters']:.2f} MB parameter storage")

    print("\nSegmentation Models:")
    for name, info in memory_reqs['segmentation'].items():
        print(
            f"{name}: {info['total_parameters'] / 1e6:.2f}M params, ~{info['memory_usage']['parameters']:.2f} MB parameter storage")

    return memory_reqs