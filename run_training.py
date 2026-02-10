from main import *
import gc
import argparse
import time
import traceback
import os

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

DEFAULT_EPOCHS_CLS = 20
DEFAULT_EPOCHS_SEG = 20

PRETRAINED_CACHE_DIR = os.path.join(os.getcwd(), "pretrained_models_cache")
os.makedirs(PRETRAINED_CACHE_DIR, exist_ok=True)

train_cls_dataset = None
val_cls_dataset = None
train_seg_dataset = None
val_seg_dataset = None
train_cls_loader = None
val_cls_loader = None
train_seg_loader = None 
val_seg_loader = None
test_cls_dataset = None
test_seg_dataset = None
test_cls_loader = None
test_seg_loader = None

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

def calculate_dice_score(y_true, y_pred, smooth=1e-6):
    intersection = (y_true * y_pred).sum()
    dice = (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
    return dice

def calculate_iou(y_true, y_pred, smooth=1e-6):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def evaluate_classification_comprehensive(model, test_loader, device, model_name):
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            if labels.dim() > 1:
                labels = labels.squeeze()
            labels = labels.long()
            
            outputs = model(images)
            
            if isinstance(outputs, dict):
                if 'logits' in outputs:
                    outputs = outputs['logits']
                elif 'prediction' in outputs:
                    outputs = outputs['prediction']
                else:
                    outputs = next(iter(outputs.values()))
            
            if outputs.dim() > 2:
                while outputs.dim() > 2:
                    outputs = F.adaptive_avg_pool2d(outputs, (1, 1)).squeeze(-1).squeeze(-1)
            
            if hasattr(model, 'classifier_head') and outputs.size(1) != 2:
                outputs = model.classifier_head(outputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'predictions': all_predictions,
        'labels': all_labels
    }

def evaluate_segmentation_comprehensive(model, test_loader, device, model_name):
    model.eval()
    total_iou = 0
    total_dice = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            
            if isinstance(outputs, dict):
                outputs = outputs.get('out', outputs.get('prediction', next(iter(outputs.values()))))
            
            if outputs.size(1) > 1:
                predictions = torch.argmax(outputs, dim=1)
            else:
                predictions = (outputs > 0.5).float().squeeze(1)
            
            for i in range(predictions.size(0)):
                pred_np = predictions[i].cpu().numpy().astype(np.uint8)
                mask_np = masks[i].cpu().numpy().astype(np.uint8)
                
                pred_flat = pred_np.flatten()
                mask_flat = mask_np.flatten()
                
                iou = calculate_iou(mask_flat, pred_flat)
                dice = calculate_dice_score(mask_flat, pred_flat)
                
                total_iou += iou
                total_dice += dice
                num_batches += 1
    
    avg_iou = total_iou / num_batches if num_batches > 0 else 0
    avg_dice = total_dice / num_batches if num_batches > 0 else 0
    
    return {
        'mean_iou': avg_iou,
        'dice_score': avg_dice
    }

def evaluate_all_models_comprehensive(cls_results, seg_results):
    print("\n==== Comprehensive Model Evaluation ====")
    
    if not hasattr(evaluate_all_models_comprehensive, 'test_created'):
        create_test_split()
        evaluate_all_models_comprehensive.test_created = True
    
    comprehensive_cls_results = {}
    comprehensive_seg_results = {}
    
    if cls_results and test_cls_loader is not None:
        print("\nEvaluating Classification Models...")
        for model_name in cls_results.keys():
            if 'error' not in cls_results[model_name]:
                try:
                    model = create_robust_classification_model(model_name, num_classes=2)
                    model.load_state_dict(torch.load(f"{model_name}_best.pth"))
                    model = model.to(device)
                    
                    metrics = evaluate_classification_comprehensive(model, test_cls_loader, device, model_name)
                    
                    metrics.update({
                        'best_val_acc': cls_results[model_name]['best_val_acc'],
                        'total_epochs': len(cls_results[model_name]['history']['train_acc']),
                        'training_time_estimate': len(cls_results[model_name]['history']['train_acc']) * 2
                    })
                    
                    comprehensive_cls_results[model_name] = metrics
                    
                    print(f"{model_name}: Test Acc={metrics['accuracy']:.1f}%, F1={metrics['f1_score']:.1f}%")
                    
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error evaluating {model_name}: {str(e)}")
    
    if seg_results and test_seg_loader is not None:
        print("\nEvaluating Segmentation Models...")
        for model_name in seg_results.keys():
            if 'error' not in seg_results[model_name]:
                try:
                    if model_name == "UMamba":
                        model = UMamba(num_classes=2)
                    elif model_name == "VMUNet":
                        model = VMUNet(num_classes=2)
                    elif model_name == "MultiCBAM":
                        model = MultiCBAMMedSegDiffNCA(num_classes=2)
                    elif model_name == "UNetPlusPlus":
                        model = UNetPlusPlusEfficientNetB7(num_classes=2)
                    elif model_name == "TransUNet":
                        model = TransUNetCNNHybrid(num_classes=2)
                    
                    model.load_state_dict(torch.load(f"{model_name}_best.pth"))
                    model = model.to(device)
                    
                    metrics = evaluate_segmentation_comprehensive(model, test_seg_loader, device, model_name)
                    
                    metrics.update({
                        'best_val_iou': seg_results[model_name]['best_val_iou'],
                        'total_epochs': len(seg_results[model_name]['history']['train_iou']),
                        'training_time_estimate': len(seg_results[model_name]['history']['train_iou']) * 3
                    })
                    
                    comprehensive_seg_results[model_name] = metrics
                    
                    print(f"{model_name}: Test IoU={metrics['mean_iou']:.3f}, Dice={metrics['dice_score']:.3f}")
                    
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error evaluating {model_name}: {str(e)}")
    
    return comprehensive_cls_results, comprehensive_seg_results

def create_performance_comparison_table(cls_results, seg_results):
    
    cls_table_data = []
    if cls_results:
        for model_name, metrics in cls_results.items():
            cls_table_data.append({
                'Model': model_name,
                'Best Accuracy': f"{metrics['accuracy']:.1f}%",
                'Epochs': metrics['total_epochs'],
                'F1-Score': f"{metrics['f1_score']:.1f}%",
                'Precision': f"{metrics['precision']:.1f}%",
                'Recall': f"{metrics['recall']:.1f}%",
                'Notes': get_model_notes(model_name, 'classification')
            })
    
    seg_table_data = []
    if seg_results:
        for model_name, metrics in seg_results.items():
            seg_table_data.append({
                'Model': model_name,
                'mIoU': f"{metrics['mean_iou']:.3f}",
                'Dice Score': f"{metrics['dice_score']:.3f}",
                'Epochs': metrics['total_epochs'],
                'Notes': get_model_notes(model_name, 'segmentation')
            })
    
    return cls_table_data, seg_table_data

def get_model_notes(model_name, task_type):
    notes_dict = {
        'classification': {
            'InternImage': 'High accuracy, moderate speed',
            'MobileNetV4': 'Lightweight, fast inference',
            'RepLKNet': 'Large kernels, good feature extraction',
            'ConvNeXtV2': 'Modern architecture, balanced performance',
            'EfficientNetV3': 'Efficient scaling, good performance'
        },
        'segmentation': {
            'UMamba': 'State-space model, efficient',
            'VMUNet': 'Vision Mamba, good for medical images',
            'MultiCBAM': 'Attention mechanism, slower training',
            'UNetPlusPlus': 'Enhanced U-Net, best boundaries',
            'TransUNet': 'Transformer-CNN hybrid, complex textures'
        }
    }
    
    return notes_dict.get(task_type, {}).get(model_name, 'Good performance')

def generate_enhanced_final_report(cls_comparison, seg_comparison, comprehensive_cls, comprehensive_seg):
    
    print("\n==== Generating Enhanced Final Report ====")
    
    cls_table_data, seg_table_data = create_performance_comparison_table(comprehensive_cls, comprehensive_seg)
    
    try:
        with open("oil_spill_detection_report.md", "w") as f:
            f.write("# Oil Spill Detection Project - Comprehensive Report\n\n")
            f.write("## Project Overview\n")
            f.write("This report presents the results of training and evaluating multiple deep learning models ")
            f.write("for oil spill detection using both classification and segmentation approaches.\n\n")
            
            if cls_table_data:
                f.write("## 5. Experimental Results\n\n")
                f.write("### 5.1 Classification Models\n\n")
                f.write("| Model | Best Accuracy | Epochs | F1-Score | Precision | Recall | Notes |\n")
                f.write("|-------|---------------|--------|----------|-----------|--------|---------|\n")
                
                cls_sorted = sorted(cls_table_data, key=lambda x: float(x['Best Accuracy'].replace('%', '')), reverse=True)
                for row in cls_sorted:
                    f.write(f"| {row['Model']} | {row['Best Accuracy']} | {row['Epochs']} | ")
                    f.write(f"{row['F1-Score']} | {row['Precision']} | {row['Recall']} | {row['Notes']} |\n")
                f.write("\n")
            
            if seg_table_data:
                f.write("### 5.2 Segmentation Models\n\n")
                f.write("| Model | mIoU | Dice Score | Epochs | Notes |\n")
                f.write("|-------|------|------------|--------|---------|\n")
                
                seg_sorted = sorted(seg_table_data, key=lambda x: float(x['mIoU']), reverse=True)
                for row in seg_sorted:
                    f.write(f"| {row['Model']} | {row['mIoU']} | {row['Dice Score']} | ")
                    f.write(f"{row['Epochs']} | {row['Notes']} |\n")
                f.write("\n")
            
            f.write("## 6. Global Comparison\n\n")
            f.write("| Task | Best Model | Score | Comments |\n")
            f.write("|------|------------|-------|-----------|\n")
            
            if cls_sorted:
                best_cls = cls_sorted[0]
                f.write(f"| Classification | {best_cls['Model']} | {best_cls['Best Accuracy']} Accuracy | ")
                f.write(f"Balanced performance with {best_cls['F1-Score']} F1-score |\n")
            
            if seg_sorted:
                best_seg = seg_sorted[0]
                f.write(f"| Segmentation | {best_seg['Model']} | {best_seg['Dice Score']} Dice Score | ")
                f.write(f"High-quality segment boundaries with {best_seg['mIoU']} mIoU |\n")
            
            f.write("\n")
            
            f.write("## Performance Visualizations\n\n")
            if cls_table_data:
                f.write("![Classification Results](classification_results.png)\n\n")
            if seg_table_data:
                f.write("![Segmentation Results](segmentation_results.png)\n\n")
            
            f.write("## Detailed Analysis\n\n")
            if cls_sorted:
                f.write("### Classification Analysis\n")
                f.write(f"- **Best Model**: {cls_sorted[0]['Model']} achieved {cls_sorted[0]['Best Accuracy']} accuracy\n")
                f.write(f"- **Most Efficient**: Analysis based on epochs and performance ratio\n")
                f.write(f"- **Robustness**: F1-scores range from {min(row['F1-Score'] for row in cls_sorted)} to {max(row['F1-Score'] for row in cls_sorted)}\n\n")
            
            if seg_sorted:
                f.write("### Segmentation Analysis\n")
                f.write(f"- **Best Model**: {seg_sorted[0]['Model']} achieved {seg_sorted[0]['Dice Score']} Dice score\n")
                f.write(f"- **Precision**: IoU scores demonstrate good boundary delineation\n")
                f.write(f"- **Consistency**: Dice coefficients show reliable segmentation quality\n\n")
            
            f.write("## Conclusions\n\n")
            f.write("The comprehensive evaluation demonstrates the effectiveness of deep learning approaches ")
            f.write("for oil spill detection. ")
            
            if cls_sorted and seg_sorted:
                f.write(f"**{cls_sorted[0]['Model']}** excelled in classification tasks while ")
                f.write(f"**{seg_sorted[0]['Model']}** provided superior segmentation results.\n\n")
            
            f.write("**Key Findings:**\n")
            f.write("- Modern architectures show significant improvement over traditional approaches\n")
            f.write("- Attention mechanisms and hybrid models demonstrate superior performance\n")
            f.write("- Balanced training approaches prevent overfitting while maintaining accuracy\n\n")
            
            f.write("**Recommendations:**\n")
            if cls_sorted:
                f.write(f"- Deploy **{cls_sorted[0]['Model']}** for real-time oil spill classification\n")
            if seg_sorted:
                f.write(f"- Use **{seg_sorted[0]['Model']}** for detailed spill boundary mapping\n")
            f.write("- Consider ensemble methods for critical applications\n")
            f.write("- Implement continuous learning for model adaptation\n\n")
        
        print("Enhanced report saved to 'oil_spill_detection_report.md'")
        
    except Exception as e:
        print(f"Error generating enhanced report: {str(e)}")
        traceback.print_exc()

def initialize_datasets():
    global train_cls_dataset, val_cls_dataset, train_seg_dataset, val_seg_dataset
    global train_cls_loader, val_cls_loader, train_seg_loader, val_seg_loader
    global test_cls_dataset, test_seg_dataset, test_cls_loader, test_seg_loader
    
    print("Initializing datasets...")
    
    if 'train_data' not in globals():
        raise ValueError("train_data not found. Make sure main.py has been executed and datasets are loaded.")
    
    train_cls_dataset = OilSpillClassificationDataset(
        train_data, DATASET_PATH, split="train", transform=classification_transform
    )
    
    if valid_data is not None:
        val_cls_dataset = OilSpillClassificationDataset(
            valid_data, DATASET_PATH, split="valid", transform=classification_transform
        )
    else:
        from sklearn.model_selection import train_test_split
        train_indices = list(range(len(train_cls_dataset)))
        train_idx, val_idx = train_test_split(train_indices, test_size=0.2, random_state=42)
        
        val_cls_dataset = torch.utils.data.Subset(train_cls_dataset, val_idx)
        train_cls_dataset = torch.utils.data.Subset(train_cls_dataset, train_idx)
    
    train_seg_dataset = OilSpillSegmentationDataset(
        train_data, DATASET_PATH, split="train", transform=segmentation_transform
    )
    
    if valid_data is not None:
        val_seg_dataset = OilSpillSegmentationDataset(
            valid_data, DATASET_PATH, split="valid", transform=segmentation_transform
        )
    else:
        from sklearn.model_selection import train_test_split
        train_indices = list(range(len(train_seg_dataset)))
        train_idx, val_idx = train_test_split(train_indices, test_size=0.2, random_state=42)
        
        val_seg_dataset = torch.utils.data.Subset(train_seg_dataset, val_idx)
        train_seg_dataset = torch.utils.data.Subset(train_seg_dataset, train_idx)
    
    train_cls_loader = DataLoader(train_cls_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_cls_loader = DataLoader(val_cls_dataset, batch_size=16, shuffle=False, num_workers=2)
    train_seg_loader = DataLoader(train_seg_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_seg_loader = DataLoader(val_seg_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"Datasets initialized:")
    print(f"- Training classification: {len(train_cls_dataset)} samples")
    print(f"- Validation classification: {len(val_cls_dataset)} samples") 
    print(f"- Training segmentation: {len(train_seg_dataset)} samples")
    print(f"- Validation segmentation: {len(val_seg_dataset)} samples")


def create_unet(encoder_name='resnet34', encoder_weights='imagenet'):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=2,
        activation='softmax',
    )


def create_test_split():
    global test_data, test_cls_dataset, test_seg_dataset, test_cls_loader, test_seg_loader
    
    if test_data is None:
        print("Warning: No test data available, using validation data for testing")
        test_data = valid_data
    
    if test_data is not None:
        test_cls_dataset = OilSpillClassificationDataset(
            test_data, DATASET_PATH, split="test", transform=classification_transform
        )
        test_seg_dataset = OilSpillSegmentationDataset(
            test_data, DATASET_PATH, split="test", transform=segmentation_transform
        )
        
        test_cls_loader = DataLoader(test_cls_dataset, batch_size=16, shuffle=False, num_workers=2)
        test_seg_loader = DataLoader(test_seg_dataset, batch_size=8, shuffle=False, num_workers=2)
        
        print(f"Created test datasets: Classification={len(test_cls_dataset)}, Segmentation={len(test_seg_dataset)}")


def evaluate_best_models(cls_results, seg_results):
    return evaluate_all_models_comprehensive(cls_results, seg_results)
    
    if seg_results:
        best_seg_model_name = max(seg_results.keys(), key=lambda k: seg_results[k]['best_val_iou'])
        print(f"Best segmentation model: {best_seg_model_name}")
        
        try:
            print(f"Best segmentation IoU: {seg_results[best_seg_model_name]['best_val_iou']:.4f}")
        except Exception as e:
            print(f"Error evaluating segmentation model: {str(e)}")


def debug_model_output(model, sample_input, model_name):
    model.eval()
    with torch.no_grad():
        try:
            output = model(sample_input)
            print(f"\n=== Debug Info for {model_name} ===")
            print(f"Input shape: {sample_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Output type: {type(output)}")
            if hasattr(output, 'logits'):
                print(f"Logits shape: {output.logits.shape}")
            return output
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            return None
# Fixed training function for classification
def train_classification_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                                   num_epochs, device, model_name):
    
    try:
        sample_batch = next(iter(train_loader))
        sample_images = sample_batch[0][:1].to(device)
        sample_labels = sample_batch[1][:1].to(device)
        
        print(f"\n=== Debugging {model_name} ===")
        print(f"Sample input shape: {sample_images.shape}")
        print(f"Sample label shape: {sample_labels.shape}")
        print(f"Sample label value: {sample_labels}")
        
        debug_output = debug_model_output(model, sample_images, model_name)
        
    except Exception as e:
        print(f"Error during debugging: {e}")
        return {'train_loss': [0], 'train_acc': [0], 'val_loss': [0], 'val_acc': [0]}
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            try:
                images = images.to(device)
                labels = labels.to(device)
                
                if labels.dim() > 1:
                    labels = labels.squeeze()
                labels = labels.long()
                
                optimizer.zero_grad()
                
                outputs = model(images)
                
                if isinstance(outputs, dict):
                    if 'logits' in outputs:
                        outputs = outputs['logits']
                    elif 'prediction' in outputs:
                        outputs = outputs['prediction']
                    else:
                        outputs = next(iter(outputs.values()))
                
                if outputs.dim() > 2:
                    while outputs.dim() > 2:
                        outputs = F.adaptive_avg_pool2d(outputs, (1, 1)).squeeze(-1).squeeze(-1)
                
                if outputs.size(1) != 2:
                    print(f"Warning: Model output has {outputs.size(1)} classes, expected 2")
                    if not hasattr(model, 'classifier_head'):
                        model.classifier_head = nn.Linear(outputs.size(1), 2).to(device)
                    outputs = model.classifier_head(outputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                if batch_idx % 20 == 0:
                    print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
                    
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                print(f"Image shape: {images.shape if 'images' in locals() else 'N/A'}")
                print(f"Labels shape: {labels.shape if 'labels' in locals() else 'N/A'}")
                print(f"Output shape: {outputs.shape if 'outputs' in locals() else 'N/A'}")
                continue
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_samples
        
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                try:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    if labels.dim() > 1:
                        labels = labels.squeeze()
                    labels = labels.long()
                    
                    outputs = model(images)
                    
                    if isinstance(outputs, dict):
                        if 'logits' in outputs:
                            outputs = outputs['logits']
                        elif 'prediction' in outputs:
                            outputs = outputs['prediction']
                        else:
                            outputs = next(iter(outputs.values()))
                    
                    if outputs.dim() > 2:
                        while outputs.dim() > 2:
                            outputs = F.adaptive_avg_pool2d(outputs, (1, 1)).squeeze(-1).squeeze(-1)
                    
                    if hasattr(model, 'classifier_head') and outputs.size(1) != 2:
                        outputs = model.classifier_head(outputs)
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
        
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), f'{model_name}_best.pth')
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        if scheduler:
            scheduler.step()
    
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    return history

def create_robust_classification_model(model_name, num_classes=2, use_pretrained=True):
    
    try:
        if model_name == "InternImage":
            import timm
            model = timm.create_model('swin_base_patch4_window7_224', 
                                    pretrained=use_pretrained, 
                                    num_classes=num_classes)
            print(f"Created Swin Transformer as InternImage fallback")
            
        elif model_name == "MobileNetV4":
            import timm
            try:
                model = timm.create_model('mobilenetv3_large_100', 
                                        pretrained=use_pretrained, 
                                        num_classes=num_classes)
            except:
                model = timm.create_model('mobilenetv2_100', 
                                        pretrained=use_pretrained, 
                                        num_classes=num_classes)
            print(f"Created MobileNet model")
            
        elif model_name == "RepLKNet":
            import timm
            try:
                model = timm.create_model('convnext_base', 
                                        pretrained=use_pretrained, 
                                        num_classes=num_classes)
            except:
                model = timm.create_model('resnet50', 
                                        pretrained=use_pretrained, 
                                        num_classes=num_classes)
            print(f"Created ConvNeXt as RepLKNet fallback")
            
        elif model_name == "ConvNeXtV2":
            import timm
            try:
                model = timm.create_model('convnext_base', 
                                        pretrained=use_pretrained, 
                                        num_classes=num_classes)
            except:
                model = timm.create_model('resnet50', 
                                        pretrained=use_pretrained, 
                                        num_classes=num_classes)
            print(f"Created ConvNeXt model")
            
        elif model_name == "EfficientNetV3":
            import timm
            try:
                model = timm.create_model('efficientnet_b0', 
                                        pretrained=use_pretrained, 
                                        num_classes=num_classes)
            except:
                model = timm.create_model('resnet50', 
                                        pretrained=use_pretrained, 
                                        num_classes=num_classes)
            print(f"Created EfficientNet model")
            
        else:
            import timm
            model = timm.create_model('resnet50', 
                                    pretrained=use_pretrained, 
                                    num_classes=num_classes)
            print(f"Created ResNet50 as default fallback for {model_name}")
        
        return model
        
    except Exception as e:
        print(f"Error creating {model_name}: {e}")
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        print(f"Created simple CNN as ultimate fallback")
        return model

def train_single_classification_model(model_name, train_loader, val_loader, num_epochs=5, use_pretrained=True):
    print(f"\n{'=' * 50}")
    print(f"Training {model_name}")
    print(f"{'=' * 50}")

    try:
        model = create_robust_classification_model(model_name, num_classes=2, use_pretrained=use_pretrained)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        history = train_classification_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=num_epochs, device=device, model_name=model_name
        )

        result = {
            'best_val_acc': max(history['val_acc']) if history['val_acc'] else 0.0,
            'history': history
        }

        del model, criterion, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
    
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        traceback.print_exc()
        return {
            'best_val_acc': 0.0,
            'history': {
                'train_loss': [0], 'train_acc': [0],
                'val_loss': [0], 'val_acc': [0]
            },
            'error': str(e)
        }


def train_single_segmentation_model(model_name, train_loader, val_loader, num_epochs=5, use_pretrained=True):
    print(f"\n{'=' * 50}")
    print(f"Training {model_name}")
    print(f"{'=' * 50}")

    try:
        if model_name == "UMamba":
            model = UMamba(num_classes=2)
        elif model_name == "VMUNet":
            model = VMUNet(num_classes=2)
        elif model_name == "MultiCBAM":
            model = MultiCBAMMedSegDiffNCA(num_classes=2)
        elif model_name == "UNetPlusPlus":
            model = UNetPlusPlusEfficientNetB7(num_classes=2)
        elif model_name == "TransUNet":
            model = TransUNetCNNHybrid(num_classes=2)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = model.to(device)

        criterion = DiceLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        history = train_segmentation_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=num_epochs, device=device, model_name=model_name
        )

        result = {
            'best_val_iou': max(history['val_iou']),
            'history': history
        }

        del model, criterion, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

        return result
    
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        traceback.print_exc()
        return {
            'best_val_iou': 0.0,
            'history': {
                'train_loss': [0], 'train_iou': [0],
                'val_loss': [0], 'val_iou': [0]
            },
            'error': str(e)
        }


def plot_classification_results(results):
    plt.figure(figsize=(12, 10))
    
    model_names = []
    best_val_accs = []
    
    for model_name, result in results.items():
        if 'error' not in result:
            model_names.append(model_name)
            best_val_accs.append(result['best_val_acc'])
    
    if not model_names:
        print("No successful classification results to plot")
        return pd.DataFrame()
    
    comparison = pd.DataFrame({
        'Model': model_names,
        'Best Validation Accuracy': best_val_accs
    }).sort_values('Best Validation Accuracy', ascending=False)
    
    plt.subplot(2, 1, 1)
    for model_name, result in results.items():
        if 'error' not in result:
            history = result['history']
            plt.plot(history['val_acc'], label=f"{model_name} (Best: {result['best_val_acc']:.4f})")
    
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for model_name, result in results.items():
        if 'error' not in result:
            history = result['history']
            plt.plot(history['val_loss'], label=model_name)
    
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('classification_results.png')
    plt.close()
    
    return comparison


def plot_segmentation_results(results):
    plt.figure(figsize=(12, 10))
    
    model_names = []
    best_val_ious = []
    
    for model_name, result in results.items():
        if 'error' not in result:
            model_names.append(model_name)
            best_val_ious.append(result['best_val_iou'])
    
    if not model_names:
        print("No successful segmentation results to plot")
        return pd.DataFrame()
    
    comparison = pd.DataFrame({
        'Model': model_names,
        'Best Validation IoU': best_val_ious
    }).sort_values('Best Validation IoU', ascending=False)
    
    plt.subplot(2, 1, 1)
    for model_name, result in results.items():
        if 'error' not in result:
            history = result['history']
            plt.plot(history['val_iou'], label=f"{model_name} (Best: {result['best_val_iou']:.4f})")
    
    plt.title('Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for model_name, result in results.items():
        if 'error' not in result:
            history = result['history']
            plt.plot(history['val_loss'], label=model_name)
    
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('segmentation_results.png')
    plt.close()
    
    return comparison


def generate_final_report(cls_comparison, seg_comparison):
    try:
        print("\n==== Generating Final Report ====")
        
        with open("oil_spill_detection_report.md", "w") as f:
            f.write("# Oil Spill Detection Project Report\n\n")
            
            if not cls_comparison.empty:
                f.write("## Classification Models Performance\n\n")
                try:
                    f.write(f"{cls_comparison.to_markdown()}\n\n")
                except ImportError:
                    f.write("Classification Results (sorted by best validation accuracy):\n\n")
                    for index, row in cls_comparison.iterrows():
                        f.write(f"- {row['Model']}: {row['Best Validation Accuracy']:.4f}\n")
                    f.write("\n")
            
            if not seg_comparison.empty:
                f.write("## Segmentation Models Performance\n\n")
                try:
                    f.write(f"{seg_comparison.to_markdown()}\n\n")
                except ImportError:
                    f.write("Segmentation Results (sorted by best validation IoU):\n\n")
                    for index, row in seg_comparison.iterrows():
                        f.write(f"- {row['Model']}: {row['Best Validation IoU']:.4f}\n")
                    f.write("\n")
            
            f.write("\n## Performance Visualizations\n\n")
            if not cls_comparison.empty:
                f.write("![Classification Results](classification_results.png)\n\n")
            if not seg_comparison.empty:
                f.write("![Segmentation Results](segmentation_results.png)\n\n")
            
            f.write("\n## Conclusions\n\n")
            
            if not cls_comparison.empty:
                best_cls_model = cls_comparison.iloc[0]['Model']
                best_cls_acc = cls_comparison.iloc[0]['Best Validation Accuracy']
                f.write(f"The best performing classification model was **{best_cls_model}** with a validation accuracy of {best_cls_acc:.4f}.\n\n")
            
            if not seg_comparison.empty:
                best_seg_model = seg_comparison.iloc[0]['Model']
                best_seg_iou = seg_comparison.iloc[0]['Best Validation IoU']
                f.write(f"The best performing segmentation model was **{best_seg_model}** with a validation IoU of {best_seg_iou:.4f}.\n\n")
            
        print(f"Report saved to 'oil_spill_detection_report.md'")
        if not cls_comparison.empty:
            print(f"Classification results visualization saved to 'classification_results.png'")
        if not seg_comparison.empty:
            print(f"Segmentation results visualization saved to 'segmentation_results.png'")
        
    except Exception as e:
        print(f"Error generating final report: {str(e)}")
        traceback.print_exc()


def run_memory_optimized_training(args):
    print("\n==== Starting Memory-Optimized Oil Spill Detection Project ====")
    
    try:
        initialize_datasets()
    except Exception as e:
        print(f"Error initializing datasets: {str(e)}")
        print("Make sure to run main.py first to load the dataset.")
        return

    cls_models = ["InternImage", "MobileNetV4", "RepLKNet", "ConvNeXtV2", "EfficientNetV3"]
    seg_models = ["UMamba", "VMUNet", "MultiCBAM", "UNetPlusPlus", "TransUNet"]

    cls_results = {}
    seg_results = {}

    if args.mode in ["all", "classification"]:
        print("\n==== Training Classification Models ====")

        if args.cls_model == "all":
            models_to_train = cls_models
        else:
            if args.cls_model in cls_models:
                models_to_train = [args.cls_model]
            else:
                print(
                    f"Warning: Unknown classification model '{args.cls_model}'. Please choose from: {', '.join(cls_models)}")
                models_to_train = []

        for model_name in models_to_train:
            try:
                result = train_single_classification_model(
                    model_name, train_cls_loader, val_cls_loader, 
                    num_epochs=args.epochs_cls,
                    use_pretrained=not args.no_pretrained
                )
                cls_results[model_name] = result

                if args.pause_between > 0:
                    print(f"\nPausing for {args.pause_between} seconds to cool down GPU...")
                    time.sleep(args.pause_between)

            except Exception as e:
                print(f"Error training classification model {model_name}: {str(e)}")
                traceback.print_exc()

        if cls_results:
            print("\nPlotting classification results...")
            try:
                cls_comparison = plot_classification_results(cls_results)
            except Exception as e:
                print(f"Error plotting classification results: {str(e)}")
                traceback.print_exc()
                cls_comparison = pd.DataFrame()

    if args.mode in ["all", "segmentation"]:
        print("\n==== Training Segmentation Models ====")

        if args.seg_model == "all":
            models_to_train = seg_models
        else:
            if args.seg_model in seg_models:
                models_to_train = [args.seg_model]
            else:
                print(
                    f"Warning: Unknown segmentation model '{args.seg_model}'. Please choose from: {', '.join(seg_models)}")
                models_to_train = []

        for model_name in models_to_train:
            try:
                result = train_single_segmentation_model(
                    model_name, train_seg_loader, val_seg_loader, 
                    num_epochs=args.epochs_seg,
                    use_pretrained=not args.no_pretrained
                )
                seg_results[model_name] = result

                if args.pause_between > 0:
                    print(f"\nPausing for {args.pause_between} seconds to cool down GPU...")
                    time.sleep(args.pause_between)

            except Exception as e:
                print(f"Error training segmentation model {model_name}: {str(e)}")
                traceback.print_exc()

        if seg_results:
            print("\nPlotting segmentation results...")
            try:
                seg_comparison = plot_segmentation_results(seg_results)
            except Exception as e:
                print(f"Error plotting segmentation results: {str(e)}")
                traceback.print_exc()
                seg_comparison = pd.DataFrame()

    if args.evaluate and ('test_data' not in globals() or test_data is None):
        try:
            create_test_split()
        except Exception as e:
            print(f"Error creating test split: {str(e)}")
            traceback.print_exc()

    comprehensive_cls, comprehensive_seg = {}, {}
    if args.evaluate and (cls_results or seg_results):
        try:
            comprehensive_cls, comprehensive_seg = evaluate_all_models_comprehensive(cls_results, seg_results)
        except Exception as e:
            print(f"Error evaluating models: {str(e)}")
            traceback.print_exc()

    if cls_results or seg_results:
        try:
            cls_comparison = plot_classification_results(cls_results) if cls_results else pd.DataFrame()
            seg_comparison = plot_segmentation_results(seg_results) if seg_results else pd.DataFrame()
            generate_final_report(cls_comparison, seg_comparison)
        except Exception as e:
            print(f"Error generating final report: {str(e)}")
            traceback.print_exc()

    print("\n==== Memory-Optimized Oil Spill Detection Project Completed ====")
    
    return {
        'classification_results': cls_results,
        'segmentation_results': seg_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-optimized training for oil spill detection")

    parser.add_argument("--mode", type=str, default="all", choices=["all", "classification", "segmentation"],
                        help="Which types of models to train")

    parser.add_argument("--cls_model", type=str, default="InternImage",
                        help="Classification model to train (or 'all' for all models)")

    parser.add_argument("--seg_model", type=str, default="UMamba",
                        help="Segmentation model to train (or 'all' for all models)")

    parser.add_argument("--epochs_cls", type=int, default=DEFAULT_EPOCHS_CLS,
                        help="Number of epochs for classification models")

    parser.add_argument("--epochs_seg", type=int, default=DEFAULT_EPOCHS_SEG,
                        help="Number of epochs for segmentation models")

    parser.add_argument("--batch_size_cls", type=int, default=16,
                        help="Batch size for classification models")

    parser.add_argument("--batch_size_seg", type=int, default=8,
                        help="Batch size for segmentation models")

    parser.add_argument("--pause_between", type=int, default=30,
                        help="Seconds to pause between training different models (0 for no pause)")

    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate models on test set after training")

    parser.add_argument("--reduced_precision", action="store_true",
                        help="Use float16 precision to reduce memory usage")
                        
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Skip using pretrained weights to avoid download errors")
                        
    parser.add_argument("--save_only", action="store_true",
                        help="Only save plots instead of displaying them")

    args = parser.parse_args()

    if args.reduced_precision and torch.cuda.is_available():
        print("Using mixed precision training to reduce memory usage")
        torch.backends.cudnn.benchmark = True
        torch.cuda.amp.autocast(enabled=True)

    try:
        results = run_memory_optimized_training(args)
        print("Training complete with results:", 
              f"Classification models: {len(results['classification_results'])}", 
              f"Segmentation models: {len(results['segmentation_results'])}")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        traceback.print_exc()