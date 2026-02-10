import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os
import timm
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import gc
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_model_directory():
    possible_paths = [
        Path.home() / "fatha" / "OilSpillProject",
        Path.cwd() / "OilSpillProject",
        Path.cwd() / "fatha" / "OilSpillProject",
        Path("/home") / "fatha" / "OilSpillProject",
        Path("OilSpillProject"),
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            print(f"Found model directory at: {path}")
            return str(path)
    
    model_dir = Path.cwd() / "OilSpillProject"
    model_dir.mkdir(exist_ok=True)
    print(f"Using/Created model directory at: {model_dir}")
    return str(model_dir)

MODEL_DIR = get_model_directory()

class InternImage(nn.Module):
    def __init__(self, num_classes=2):
        super(InternImage, self).__init__()
        try:
            model_names = [
                'internimage_t_1k_224',
                'internimage_s_1k_224', 
                'internimage_b_1k_224',
                'swin_tiny_patch4_window7_224'
            ]
            
            self.backbone = None
            for model_name in model_names:
                try:
                    self.backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
                    print(f"InternImage: Using {model_name}")
                    break
                except:
                    continue
            
            if self.backbone is None:
                self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
            
            self._fix_classifier_head(num_classes)
                
        except Exception as e:
            print(f"InternImage init error: {e}")
            self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
            self._fix_classifier_head(num_classes)
    
    def _fix_classifier_head(self, num_classes):
        try:
            if hasattr(self.backbone, 'head'):
                head = self.backbone.head
                if hasattr(head, 'fc'):
                    in_features = head.fc.in_features
                    self.backbone.head.fc = nn.Linear(in_features, num_classes)
                elif hasattr(head, 'in_features'):
                    in_features = head.in_features
                    self.backbone.head = nn.Linear(in_features, num_classes)
                elif isinstance(head, nn.Linear):
                    in_features = head.in_features
                    self.backbone.head = nn.Linear(in_features, num_classes)
                else:
                    in_features = self._get_head_input_features()
                    self.backbone.head = nn.Linear(in_features, num_classes)
                    
            elif hasattr(self.backbone, 'classifier'):
                classifier = self.backbone.classifier
                if hasattr(classifier, 'in_features'):
                    in_features = classifier.in_features
                    self.backbone.classifier = nn.Linear(in_features, num_classes)
                else:
                    in_features = self._get_head_input_features()
                    self.backbone.classifier = nn.Linear(in_features, num_classes)
            else:
                in_features = self._get_head_input_features()
                self.backbone.head = nn.Linear(in_features, num_classes)
                
        except Exception as e:
            print(f"Error fixing classifier head: {e}")
            self.backbone.head = nn.Linear(768, num_classes)
    
    def _get_head_input_features(self):
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            
            original_head = None
            if hasattr(self.backbone, 'head'):
                original_head = self.backbone.head
                self.backbone.head = nn.Identity()
            elif hasattr(self.backbone, 'classifier'):
                original_head = self.backbone.classifier
                self.backbone.classifier = nn.Identity()
            
            with torch.no_grad():
                features = self.backbone(dummy_input)
                if isinstance(features, tuple):
                    features = features[0]
                in_features = features.shape[-1]
            
            if original_head is not None:
                if hasattr(self.backbone, 'head'):
                    self.backbone.head = original_head
                else:
                    self.backbone.classifier = original_head
                    
            return in_features
            
        except Exception as e:
            print(f"Error getting head input features: {e}")
            return 768
    
    def forward(self, x):
        try:
            output = self.backbone(x)
            if len(output.shape) > 2:
                output = output.view(output.size(0), -1)
                if not hasattr(self, 'final_fc'):
                    self.final_fc = nn.Linear(output.size(1), 2).to(output.device)
                output = self.final_fc(output)
            return output
        except Exception as e:
            print(f"InternImage forward error: {e}")
            batch_size = x.size(0)
            return torch.zeros(batch_size, 2, device=x.device)

class MobileNetV4(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV4, self).__init__()
        try:
            model_names = [
                'mobilenetv4_conv_small.e2400_r224_in1k',
                'mobilenetv3_large_100',
                'mobilenetv2_100'
            ]
            
            self.backbone = None
            for model_name in model_names:
                try:
                    self.backbone = timm.create_model(model_name, pretrained=False)
                    print(f"MobileNetV4: Using {model_name}")
                    break
                except:
                    continue
            
            if self.backbone is None:
                self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=False)
                
            if hasattr(self.backbone, 'classifier'):
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(in_features, num_classes)
            elif hasattr(self.backbone, 'head'):
                in_features = self.backbone.head.in_features
                self.backbone.head = nn.Linear(in_features, num_classes)
                
        except Exception as e:
            print(f"MobileNetV4 init error: {e}")
            self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=False)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class RepLKNet(nn.Module):
    def __init__(self, num_classes=2):
        super(RepLKNet, self).__init__()
        try:
            model_names = [
                'replknet_31b_in1k',
                'replknet_31l_in1k', 
                'resnext50_32x4d',
                'resnet50',
            ]
            
            self.backbone = None
            for model_name in model_names:
                try:
                    self.backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
                    print(f"RepLKNet: Successfully created {model_name}")
                    break
                except Exception as e:
                    print(f"Failed to create {model_name}: {e}")
                    continue
            
            if self.backbone is None:
                self.backbone = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)
                
        except Exception as e:
            print(f"RepLKNet init error: {e}")
            self.backbone = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class ConvNeXtV2(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNeXtV2, self).__init__()
        try:
            model_names = [
                'convnextv2_tiny.fcmae_ft_in22k_in1k',
                'convnext_tiny.in12k_ft_in1k',
                'convnext_tiny'
            ]
            
            self.backbone = None
            for model_name in model_names:
                try:
                    self.backbone = timm.create_model(model_name, pretrained=False)
                    print(f"ConvNeXtV2: Using {model_name}")
                    break
                except:
                    continue
            
            if self.backbone is None:
                self.backbone = timm.create_model('convnext_tiny', pretrained=False)
                
            if hasattr(self.backbone, 'head') and hasattr(self.backbone.head, 'fc'):
                in_features = self.backbone.head.fc.in_features
                self.backbone.head.fc = nn.Linear(in_features, num_classes)
            elif hasattr(self.backbone, 'head'):
                in_features = self.backbone.head.in_features
                self.backbone.head = nn.Linear(in_features, num_classes)
                
        except Exception as e:
            print(f"ConvNeXtV2 init error: {e}")
            self.backbone = timm.create_model('convnext_tiny', pretrained=False)
            in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class EfficientNetV3(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetV3, self).__init__()
        try:
            model_names = [
                'efficientnet_b0.ra_in1k',
                'tf_efficientnetv2_b0.in1k',
                'efficientnet_b0'
            ]
            
            self.backbone = None
            for model_name in model_names:
                try:
                    self.backbone = timm.create_model(model_name, pretrained=False)
                    print(f"EfficientNetV3: Using {model_name}")
                    break
                except:
                    continue
            
            if self.backbone is None:
                self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
                
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
            
        except Exception as e:
            print(f"EfficientNetV3 init error: {e}")
            self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class ResNeXtV3(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNeXtV3, self).__init__()
        try:
            model_names = [
                'resnext50_32x4d',
                'resnet50',
            ]
            
            self.backbone = None
            for model_name in model_names:
                try:
                    self.backbone = timm.create_model(model_name, pretrained=False)
                    print(f"ResNeXtV3: Using {model_name}")
                    break
                except:
                    continue
            
            if self.backbone is None:
                self.backbone = timm.create_model('resnet50', pretrained=False)
                
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        except Exception as e:
            print(f"ResNeXtV3 init error: {e}")
            self.backbone = timm.create_model('resnet50', pretrained=False)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class UMamba(nn.Module):
    def __init__(self, num_classes=2):
        super(UMamba, self).__init__()
        self.model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights=None,
            classes=num_classes,
            activation='softmax'
        )
    
    def forward(self, x):
        return self.model(x)

class VMUNet(nn.Module):
    def __init__(self, num_classes=2):
        super(VMUNet, self).__init__()
        try:
            self.model = smp.Unet(
                encoder_name='mit_b2',
                encoder_weights=None,
                classes=num_classes,
                activation='softmax'
            )
        except:
            self.model = smp.Unet(
                encoder_name='resnet50',
                encoder_weights=None,
                classes=num_classes,
                activation='softmax'
            )
    
    def forward(self, x):
        return self.model(x)

class MultiCBAMMedSegDiffNCA(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiCBAMMedSegDiffNCA, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name='efficientnet-b4',
            encoder_weights=None,
            classes=num_classes,
            activation='softmax'
        )
    
    def forward(self, x):
        return self.model(x)

class UNetPlusPlusEfficientNetB7(nn.Module):
    def __init__(self, num_classes=2):
        super(UNetPlusPlusEfficientNetB7, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name='efficientnet-b7',
            encoder_weights=None,
            classes=num_classes,
            activation='softmax'
        )
    
    def forward(self, x):
        return self.model(x)

class TransUNetCNNHybrid(nn.Module):
    def __init__(self, num_classes=2):
        super(TransUNetCNNHybrid, self).__init__()
        try:
            self.model = smp.DeepLabV3Plus(
                encoder_name='mit_b3',
                encoder_weights=None,
                classes=num_classes,
                activation='softmax'
            )
        except:
            self.model = smp.DeepLabV3Plus(
                encoder_name='resnet101',
                encoder_weights=None,
                classes=num_classes,
                activation='softmax'
            )
    
    def forward(self, x):
        return self.model(x)

def safe_load_state_dict(model, checkpoint_path, device):
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key[7:] if key.startswith('module.') else key
            cleaned_state_dict[clean_key] = value
        
        missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
        
        if len(missing) == 0 and len(unexpected) == 0:
            print("✓ Perfect state dict match!")
        else:
            print(f"⚠ Minor mismatches - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
        return True
                
    except Exception as e:
        print(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return False

def load_classification_models():
    models = {}
    model_classes = {
        "InternImage": InternImage,
        "MobileNetV4": MobileNetV4,
        "RepLKNet": RepLKNet,
        "ConvNeXtV2": ConvNeXtV2,
        "EfficientNetV3": EfficientNetV3,
        "ResNeXtV3": ResNeXtV3
    }
    
    print(f"Looking for models in: {MODEL_DIR}")
    print(f"Available files: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'Directory not found'}")
    
    for model_name, model_class in model_classes.items():
        for suffix in ["_best.pth", "_final.pth", ".pth"]:
            model_path = os.path.join(MODEL_DIR, f"{model_name}{suffix}")
            
            if os.path.exists(model_path):
                print(f"\n=== Loading {model_name} from {model_path} ===")
                try:
                    model = model_class().to(device)
                    if safe_load_state_dict(model, model_path, device):
                        model.eval()
                        models[model_name] = model
                        print(f"✓ Successfully loaded {model_name}")
                        break
                except Exception as e:
                    print(f"✗ Failed to load {model_name}: {e}")
                    continue
    
    return models

def load_segmentation_models():
    models = {}
    model_classes = {
        "UMamba": UMamba,
        "VMUNet": VMUNet,
        "MultiCBAMMedSegDiffNCA": MultiCBAMMedSegDiffNCA,
        "UNetPlusPlusEfficientNetB7": UNetPlusPlusEfficientNetB7,
        "TransUNetCNNHybrid": TransUNetCNNHybrid,
        "MultiCBAM": MultiCBAMMedSegDiffNCA,
        "UNetPlusPlus": UNetPlusPlusEfficientNetB7,
        "TransUNet": TransUNetCNNHybrid,
    }
    
    for model_name, model_class in model_classes.items():
        for suffix in ["_best.pth", "_final.pth", ".pth"]:
            model_path = os.path.join(MODEL_DIR, f"{model_name}{suffix}")
            
            if os.path.exists(model_path):
                print(f"\n=== Loading segmentation {model_name} from {model_path} ===")
                try:
                    model = model_class().to(device)
                    if safe_load_state_dict(model, model_path, device):
                        model.eval()
                        models[model_name] = model
                        print(f"✓ Successfully loaded segmentation {model_name}")
                        break
                except Exception as e:
                    print(f"✗ Failed to load segmentation {model_name}: {e}")
                    continue
    
    return models

print("Loading classification models...")
classification_models = load_classification_models()
print(f"Loaded {len(classification_models)} classification models: {list(classification_models.keys())}")

print("\nLoading segmentation models...")
segmentation_models = load_segmentation_models()
print(f"Loaded {len(segmentation_models)} segmentation models: {list(segmentation_models.keys())}")

def preprocess_image_classification(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def preprocess_image_segmentation(image, size=(224, 224)):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return image_tensor

def classify_image(image, model_name):
    if model_name not in classification_models:
        return f"Model {model_name} not available", 0.0
    
    model = classification_models[model_name]
    image_tensor = preprocess_image_classification(image)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        no_spill_prob = probabilities[0][0].item()
        spill_prob = probabilities[0][1].item()
    
    class_names = ["No Oil Spill", "Oil Spill Detected"]
    result = class_names[predicted_class]
    
    detailed_result = f"{result}\nNo Spill: {no_spill_prob*100:.2f}%\nOil Spill: {spill_prob*100:.2f}%"
    
    return detailed_result, confidence * 100

def segment_image(image, model_name):
    if model_name not in segmentation_models:
        return None, "Model not available"
    
    model = segmentation_models[model_name]
    original_size = image.size[::-1] if isinstance(image, Image.Image) else image.shape[:2]
    
    image_tensor = preprocess_image_segmentation(image)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        pred_mask = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
    
    pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                          (original_size[1], original_size[0]), 
                          interpolation=cv2.INTER_NEAREST)
    
    return pred_mask, "Success"

def create_overlay_visualization(original_image, mask, alpha=0.6):
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    colored_mask = np.zeros_like(original_image)
    colored_mask[mask == 1] = [255, 0, 0]
    
    overlay = cv2.addWeighted(original_image, 1-alpha, colored_mask, alpha, 0)
    return overlay

def process_classification(image, model_name):
    if image is None:
        return "Please upload an image", "", None
    
    if not classification_models:
        return "No classification models available", "", None
    
    if model_name not in classification_models:
        return f"Model {model_name} not loaded", "", None
    
    try:
        result, confidence = classify_image(image, model_name)
        confidence_text = f"Confidence: {confidence:.2f}%"
        return result, confidence_text, image
    
    except Exception as e:
        return f"Error: {str(e)}", "", None

def process_segmentation(image, model_name):
    if image is None:
        return None, None, "Please upload an image"
    
    if not segmentation_models:
        return None, None, "No segmentation models available"
    
    if model_name not in segmentation_models:
        return None, None, f"Model {model_name} not loaded"
    
    try:
        mask, status = segment_image(image, model_name)
        
        if mask is None:
            return None, None, f"Error: {status}"
        
        overlay = create_overlay_visualization(image, mask)
        mask_display = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
        
        total_pixels = mask.shape[0] * mask.shape[1]
        oil_pixels = np.sum(mask == 1)
        oil_percentage = (oil_pixels / total_pixels) * 100
        
        info = f"Oil spill area: {oil_percentage:.2f}% of image\nTotal pixels: {total_pixels:,}\nOil spill pixels: {oil_pixels:,}\nModel: {model_name}"
        
        return mask_display, overlay, info
    
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def create_interface():
    with gr.Blocks(title="Oil Spill Detection System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🛢️ Oil Spill Detection System")
        gr.Markdown("Upload an image to detect and segment oil spills using your trained models.")
        
        status_lines = [f"**Model Directory:** `{MODEL_DIR}`"]
        
        if classification_models:
            status_lines.append(f"**Classification Models ({len(classification_models)}):** {', '.join(classification_models.keys())}")
        else:
            status_lines.append("**Classification Models:** None loaded")
            
        if segmentation_models:
            status_lines.append(f"**Segmentation Models ({len(segmentation_models)}):** {', '.join(segmentation_models.keys())}")
        else:
            status_lines.append("**Segmentation Models:** None loaded")
        
        gr.Markdown("## 📊 Model Status\n" + "\n".join(status_lines))
        
        if not classification_models and not segmentation_models:
            gr.Markdown("## ⚠️ No Models Loaded\nCheck your model directory and ensure .pth files are present.")
        
        with gr.Tab("🔍 Classification"):
            with gr.Row():
                with gr.Column():
                    cls_input = gr.Image(type="pil", label="Upload Image")
                    cls_model = gr.Dropdown(
                        choices=list(classification_models.keys()) if classification_models else ["No models available"],
                        value=list(classification_models.keys())[0] if classification_models else None,
                        label="Select Classification Model"
                    )
                    cls_btn = gr.Button("🔍 Classify", variant="primary", size="lg")
                
                with gr.Column():
                    cls_result = gr.Textbox(label="Classification Result", interactive=False, lines=3)
                    cls_confidence = gr.Textbox(label="Confidence Score", interactive=False)
                    cls_output = gr.Image(label="Input Image")
        
        with gr.Tab("🎯 Segmentation"):
            with gr.Row():
                with gr.Column():
                    seg_input = gr.Image(type="pil", label="Upload Image")
                    seg_model = gr.Dropdown(
                        choices=list(segmentation_models.keys()) if segmentation_models else ["No models available"],
                        value=list(segmentation_models.keys())[0] if segmentation_models else None,
                        label="Select Segmentation Model"
                    )
                    seg_btn = gr.Button("🎯 Segment", variant="primary", size="lg")
                    seg_info = gr.Textbox(label="Segmentation Analysis", interactive=False, lines=4)
                
                with gr.Column():
                    with gr.Row():
                        seg_mask = gr.Image(label="Segmentation Mask")
                        seg_overlay = gr.Image(label="Overlay Visualization")
        
        cls_btn.click(
            fn=process_classification,
            inputs=[cls_input, cls_model],
            outputs=[cls_result, cls_confidence, cls_output]
        )
        
        seg_btn.click(
            fn=process_segmentation,
            inputs=[seg_input, seg_model],
            outputs=[seg_mask, seg_overlay, seg_info]
        )
    
    return demo

if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n{'='*50}")
    print(f"Oil Spill Detection System")
    print(f"{'='*50}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Classification models: {len(classification_models)}")
    print(f"Segmentation models: {len(segmentation_models)}")
    
    if not classification_models and not segmentation_models:
        print("\n⚠️ No models loaded. Please check:")
        print("1. Model directory path is correct")
        print("2. .pth files exist in the directory")
        print("3. File permissions are correct")
        print(f"\nLooking for files like:")
        print("- InternImage_best.pth")
        print("- MobileNetV4_final.pth")
        print("- UMamba_best.pth")
        print("- etc.")
    else:
        print("✅ Models loaded successfully!")
    
    demo = create_interface()
    demo.launch(
        server_port=7860,
        share=False,
        debug=True,
        inbrowser=True
    )