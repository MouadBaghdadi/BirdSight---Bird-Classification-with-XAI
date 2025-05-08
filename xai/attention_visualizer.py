import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import inspect
from transformers import SwinForImageClassification
from transformers.models.swin.modeling_swin import SwinSelfAttention

def visualize_swin_attention(model_path, image_path, num_classes=1011, layer_idx=None, gamma=0.7, show_plot=True, save_path=None):
    model_name = "microsoft/swin-large-patch4-window12-384-in22k"
    model = SwinForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        output_attentions=True  
    )

    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    attention_maps = []
    
    def attention_hook(module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            attention = output[1]
            attention_maps.append(attention.detach().cpu())
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, SwinSelfAttention):
            hooks.append(module.register_forward_hook(attention_hook))
    
    print(f"Running inference on image: {image_path}")
    with torch.no_grad():
        outputs = model(input_tensor, output_attentions=True)
        predicted_class = outputs.logits.argmax(-1).item() + 1
        confidence = torch.softmax(outputs.logits, dim=-1)[0, predicted_class].item()
        print(f"Predicted class: {predicted_class}, confidence: {confidence:.4f}")
    
    for hook in hooks:
        hook.remove()
    
    print(f"Collected {len(attention_maps)} attention maps")
    
    if not attention_maps:
        print("No attention maps were collected!")
        return
    
    attn_to_use = attention_maps[-1] if layer_idx is None else attention_maps[layer_idx]
    print(f"Using attention map with shape: {attn_to_use.shape}")
    
    if attn_to_use.dim() >= 4 and attn_to_use.shape[1] > 1:
        attn_to_use = attn_to_use.mean(1) 
    
    num_windows = attn_to_use.shape[0]
    print(f"Number of windows: {num_windows}")
    
    grid_side = int(np.sqrt(num_windows))
    window_size = int(np.sqrt(attn_to_use.shape[-1]))
    print(f"Grid size: {grid_side}x{grid_side}, Window size: {window_size}x{window_size}")
    
    full_heatmap = np.zeros((grid_side * window_size, grid_side * window_size))
    
    for i in range(num_windows):
        window_row = i // grid_side
        window_col = i % grid_side
        
        
        if attn_to_use.dim() == 3:  
            window_attn = attn_to_use[i].mean(0)
        elif attn_to_use.dim() == 4:  # [batch, num_heads, window_tokens, window_tokens]
            window_attn = attn_to_use[0, i].mean(0)
        else:
            window_attn = attn_to_use[i].mean(0)
        
        window_attn = window_attn.reshape(window_size, window_size).numpy()
        
        row_start = window_row * window_size
        row_end = row_start + window_size
        col_start = window_col * window_size
        col_end = col_start + window_size
        
        center_y, center_x = grid_side*window_size/2, grid_side*window_size/2
        y, x = np.ogrid[:window_size, :window_size]
        y = y + row_start - center_y
        x = x + col_start - center_x
        mask = np.exp(-(x**2 + y**2) / (2 * (grid_side*window_size/4)**2))
        
        window_attn = window_attn * mask
        
        full_heatmap[row_start:row_end, col_start:col_end] = window_attn
    
    full_heatmap = (full_heatmap - full_heatmap.min()) / (full_heatmap.max() - full_heatmap.min() + 1e-8)
    
    full_heatmap = np.power(full_heatmap, gamma)
    
    full_heatmap = cv2.resize(full_heatmap, (384, 384), interpolation=cv2.INTER_CUBIC)
    
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(full_heatmap, cmap='jet')
    plt.title("Attention Heatmap")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    resized_img = np.array(image.resize((384, 384)))
    plt.imshow(resized_img)
    plt.imshow(full_heatmap, alpha=0.5, cmap='jet')
    plt.title(f"Overlay (Class {predicted_class}, {confidence:.2%})")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(image_path), "swin_attention_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close() 
    
    print(f"Visualization complete! Saved as '{save_path}'")
    return full_heatmap

def debug_swin_model():
    print("Démarrage du débogage du modèle Swin...")
    
    # Load model
    model_name = "microsoft/swin-large-patch4-window12-384-in22k"
    model = SwinForImageClassification.from_pretrained(model_name)
    print("Modèle chargé avec succès\n")
    
    print("Structure générale:")
    for name, _ in model.named_children():
        print(f"- {name}: {type(getattr(model, name)).__name__}")
    print()
    
    print("Structure du backbone Swin:")
    for name, _ in model.swin.named_children():
        print(f"- {name}: {type(getattr(model.swin, name)).__name__}")
    print()
    
    print("Modules d'attention dans le modèle:")
    for name, module in model.named_modules():
        if "attention" in name.lower():
            print(f"- {name}: {type(module).__name__}")
    print()
    
    print("Analyse détaillée du premier module d'attention (swin.encoder.layers.0.blocks.0.attention):")
    first_attn = model.swin.encoder.layers[0].blocks[0].attention
    
    print("Attributs:")
    for attr in dir(first_attn):
        if not attr.startswith("_") and not callable(getattr(first_attn, attr)):
            print(f"  - {attr}: {type(getattr(first_attn, attr))}")
    
    print("\nMéthodes:")
    for method in dir(first_attn):
        if not method.startswith("_") and callable(getattr(first_attn, method)):
            print(f"  - {method}()")
    print()
    
    print("5. Vérification des hooks d'attention:")
    for name, module in model.named_modules():
        if "attention" in name.lower():
            has_attn_method = hasattr(module, "get_attn_weights")
            has_attn_attr = hasattr(module, "attn")
            
            try:
                forward_is_tuple = False
                if hasattr(module, "forward"):
                    sig = inspect.signature(module.forward)
                    if len(sig.parameters) > 0:
                        forward_is_tuple = True  
            except:
                forward_is_tuple = False
                
            print(f"Module {name}:")
            print(f"  - Has get_attn_weights(): {has_attn_method}")
            print(f"  - Has attn attribute: {has_attn_attr}")
            print(f"  - Forward outputs tuple: {forward_is_tuple}")
    
# if __name__ == "__main__":
#     # debug_swin_model()
    
#     save_path = None
#     model_path = "models/model_valacc_89_20250506_004455.pt" 
#     image_path = "62368568953146948a0bd483808f7000.jpg"  
#     visualize_swin_attention(model_path, image_path)