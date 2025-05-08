from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
import traceback
import time
from xai.swin_classifier import SwinClassifier
import torch.nn as nn
import matplotlib 
matplotlib.use('Agg')


app = Flask(__name__, static_folder='static')

def create_folders():
    os.makedirs(os.path.join(app.static_folder, 'uploads'), exist_ok=True)

@app.before_request
def before_request():
    if not hasattr(app, '_got_first_request'):
        create_folders()
        app._got_first_request = True

model_path = os.path.join(os.path.dirname(__file__), 'models', 'corrected_model.onnx')
session = ort.InferenceSession(model_path)

num_classes = 1011

model = SwinClassifier(num_classes=1011)
checkpoint = torch.load("models/model_valacc_89_20250506_004455.pt", map_location="cpu")

model.load_state_dict(checkpoint, strict=False)
model.eval()


def load_class_names():
    class_names = {}
    with open(os.path.join(os.path.dirname(__file__), 'static', 'data', 'classes.txt'), 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    index, name = parts
                    class_names[int(index)] = name
    return class_names

def load_bird_audio_data():
    with open(os.path.join(os.path.dirname(__file__), 'static', 'data', 'bird_audio.json'), 'r') as f:
        return json.load(f)

class_names = load_class_names()
bird_audio_data = load_bird_audio_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((384, 384))

        img_array = np.array(img).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        img_array = (img_array - mean) / std
        
        img_array = img_array.transpose(2, 0, 1)
        
        img_array = np.expand_dims(img_array, axis=0)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: img_array})
        
        output = result[0][0]
        predicted_class = np.argmax(output)
        
  
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / exp_output.sum()
        confidence = float(probabilities[predicted_class] * 100)
        
        class_index = predicted_class + 1  
        full_class_name = class_names.get(class_index, f"Class {class_index}")
        

        species_name = full_class_name
        life_stage = ""
        
        import re
        matches = re.match(r'^(.*?)(?:\s*\(([^)]+)\))?$', full_class_name)
        if matches:
            species_name = matches.group(1).strip()
            if matches.group(2):
                life_stage = matches.group(2).strip()
        print(f"Species name: {species_name}, Life stage: {life_stage}")
        base_audio_name = species_name.lower().replace(' ', '_').replace('-', '_')
        print(f"Base audio name: {base_audio_name}")
        audio_path_mp3 = f"/static/audio/{base_audio_name}.mp3"
        audio_path_wav = f"/static/audio/{base_audio_name}.wav"
        
        mp3_exists = os.path.exists(os.path.join(app.static_folder, 'audio', f"{base_audio_name}.mp3"))
        wav_exists = os.path.exists(os.path.join(app.static_folder, 'audio', f"{base_audio_name}.wav"))
        
        audio_path = None
        if mp3_exists:
            audio_path = audio_path_mp3
        elif wav_exists:
            audio_path = audio_path_wav
                
        return jsonify({
            'class_name': full_class_name,
            'species_name': species_name,
            'life_stage': life_stage,
            'confidence': confidence,
            'audio_path': audio_path,
            'class_index': int(class_index)
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/heatmap', methods=['POST'])
def generate_heatmap():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        temp_dir = os.path.join(app.static_folder, 'uploads')
        os.makedirs(temp_dir, exist_ok=True)
        
        timestamp = int(time.time())
        temp_filename = f"temp_{timestamp}_{file.filename}"
        temp_path = os.path.join(temp_dir, temp_filename)
        file.save(temp_path)
        
        print(f"Saved uploaded image to {temp_path}")
        
        model_path = os.path.join(os.path.dirname(__file__), "models/model_valacc_89_20250506_004455.pt")
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return jsonify({'error': f'Model not found at {model_path}'}), 500
        
        print(f"Using model from {model_path}")
        
        heatmap_filename = f"heatmap_{timestamp}.png"
        heatmap_path = os.path.join(temp_dir, heatmap_filename)
        
        try:
            from xai.attention_visualizer import visualize_swin_attention
            
            print("Attempting to generate heatmap...")
            visualize_swin_attention(
                model_path=model_path,
                image_path=temp_path,
                save_path=heatmap_path,
                show_plot=False
            )
            print(f"Heatmap generated at {heatmap_path}")
            
        except Exception as viz_error:
            print(f"Visualization error: {str(viz_error)}")
            print(traceback.format_exc())
            
            try:
                img = Image.open(temp_path)
                img = img.resize((384, 384))
                heatmap = Image.new('RGB', img.size, (255, 0, 0))
                heatmap.putalpha(100) 
                img.paste(heatmap, (0, 0), heatmap)
                img.save(heatmap_path)
                print(f"Created placeholder heatmap at {heatmap_path}")
            except:
                return jsonify({'error': f'Failed to generate heatmap: {str(viz_error)}'}), 500
        
        if not os.path.exists(heatmap_path):
            return jsonify({'error': 'Failed to generate heatmap image'}), 500
        
        heatmap_url = f'/static/uploads/{heatmap_filename}'
        original_url = f'/static/uploads/{temp_filename}'
        
        print(f"Returning heatmap URL: {heatmap_url}")
        return jsonify({
            'heatmap_url': heatmap_url,
            'original_url': original_url,
            'success': True
        })
        
    except Exception as e:
        print(f"General error in generate_heatmap: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500    

if __name__ == '__main__':
    os.makedirs(os.path.join(app.static_folder, 'data'), exist_ok=True)
    os.makedirs(os.path.join(app.static_folder, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(app.static_folder, 'uploads'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    
    app.run(debug=True)
