#!/usr/bin/env python3
"""
Lung Cancer Classification Web App
Upload images and get predictions with confidence scores
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw
import io
import base64
import os
import numpy as np
import cv2

def is_lung_like(image):
    img = np.array(image)

    # Check if image is too colorful (CT scans are usually grayscale)
    if img.ndim == 3:
        color_diff = np.std(img[:, :, 0] - img[:, :, 1])
        if color_diff > 30:
            return False

    # Check brightness
    if img.mean() > 200:
        return False

    return True


from run_lung_cancer_model import CSPDarkNet53

from auth import init_db, create_user, verify_user, get_user_by_id

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(int(user_id))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_PATH = 'cspdarknet53_best.pth'
CLASS_NAMES = ['Benign cases', 'Malignant cases', 'Normal cases']

model = None

def load_model():
    """Load the trained model"""
    global model
    if model is None:
        model = CSPDarkNet53(num_classes=len(CLASS_NAMES)).to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f" Model loaded from {MODEL_PATH}")
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def generate_gradcam(model, input_tensor, target_layer):
    """Generate Grad-CAM heatmap"""
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    

    output = model(input_tensor)
    pred_class = output.argmax(dim=1)
    

    model.zero_grad()
    output[0, pred_class].backward()
    
    
    handle_forward.remove()
    handle_backward.remove()
    
    
    gradients_val = gradients[0].cpu().data.numpy()[0]
    activations_val = activations[0].cpu().data.numpy()[0]
    
    weights = np.mean(gradients_val, axis=(1, 2))
    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * activations_val[i]
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    
    return cam

def detect_bounding_boxes(heatmap, threshold=0.7):
    """Detect bounding boxes from heatmap with improved accuracy"""
    
    heatmap_smooth = cv2.GaussianBlur(heatmap, (5, 5), 0)
    
    
    binary = (heatmap_smooth > threshold).astype(np.uint8) * 255
    
    
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    
    box_scores = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  
            x, y, w, h = cv2.boundingRect(contour)
            
            
            mask = np.zeros_like(heatmap)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            avg_intensity = np.sum(heatmap * mask) / area
            
            
            score = area * avg_intensity
            box_scores.append((score, (x, y, w, h)))
    
    if not box_scores:
        return []
    
    
    box_scores.sort(reverse=True)
    top_box = box_scores[0][1]
    
    
    x, y, w, h = top_box
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(224 - x, w + 2 * padding)
    h = min(224 - y, h + 2 * padding)
    
    return [(x, y, w, h)]

def draw_boxes_on_image(image, boxes, predicted_class):
    """Draw bounding boxes on image"""
    if not boxes:
        return image
    
    draw = ImageDraw.Draw(image)
    
    
    if 'Malignant' in predicted_class:
        box_color = 'red'
        label = 'Cancer Detected'
    elif 'Benign' in predicted_class:
        box_color = 'yellow'
        label = 'Benign Lesion'
    else:
        return image  
    
    for (x, y, w, h) in boxes:
        
        for i in range(4):
            draw.rectangle([x-i, y-i, x+w+i, y+h+i], outline=box_color)
        
    
        label_bbox = draw.textbbox((x, y-20), label)
        draw.rectangle(label_bbox, fill=box_color)
        draw.text((x, y-20), label, fill='black')
    
    return image

def predict_image(image):
    """Make prediction on uploaded image with bounding boxes"""
    model = load_model()
    
    
    original_image = image.convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, 1)
    
    confidence_value = confidence.item()

    THRESHOLD = 0.7

    if confidence_value < THRESHOLD:
        return {
            "predicted_class": "Unknown / Not a lung CT image",
            "confidence": confidence_value * 100,
            "all_probabilities": {},
            "annotated_image": None
        }
    
    all_probs = probs[0].cpu().numpy()
    predicted_class = CLASS_NAMES[pred_class.item()]
    
    
    annotated_image = None
    if 'Malignant' in predicted_class or 'Benign' in predicted_class:
        
        target_layer = model.stage5.concat_conv.conv
        
        
        heatmap = generate_gradcam(model, input_tensor, target_layer)
        
        
        boxes = detect_bounding_boxes(heatmap, threshold=0.75)
        
        
        annotated_image = original_image.copy()
        annotated_image = annotated_image.resize((224, 224))
        annotated_image = draw_boxes_on_image(annotated_image, boxes, predicted_class)
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence.item() * 100,
        'all_probabilities': {
            CLASS_NAMES[i]: float(all_probs[i] * 100) 
            for i in range(len(CLASS_NAMES))
        },
        'annotated_image': annotated_image
    }

@app.route('/')
@login_required
def index():
    """Render main page"""
    return render_template('index.html', username=current_user.username)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handle user registration"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            return render_template('signup.html', error='Passwords do not match')
        
        success, message = create_user(username, email, password)
        if success:
            return redirect(url_for('login', success='Account created successfully! Please sign in.'))
        else:
            return render_template('signup.html', error=message)
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    success_msg = request.args.get('success')
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = verify_user(username, password)
        if user:
            login_user(user)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html', success=success_msg)

@app.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type. Please upload PNG or JPG'}), 400
    
    try:
        
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Validate image type
        if not is_lung_like(image):
            return jsonify({
                "error": "Uploaded image does not appear to be a lung CT scan"
            }), 400
        
        
        result = predict_image(image)
        

        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        result['image'] = f"data:image/jpeg;base64,{image_base64}"
        

        if result['annotated_image']:
            buffered = io.BytesIO()
            result['annotated_image'].save(buffered, format="JPEG")
            annotated_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            result['annotated_image'] = f"data:image/jpeg;base64,{annotated_base64}"
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    
    init_db()
    

    if not os.path.exists(MODEL_PATH):
        print(f" Error: Model file '{MODEL_PATH}' not found!")
        exit(1)
    
    
    load_model()
    
    print("\n" + "=" * 60)
    print(" Lung Cancer Classification Web App")
    print("=" * 60)
    print(f" Model loaded successfully")
    print(f" Classes: {CLASS_NAMES}")
    print(f" Device: {device}")
    print("\n Starting server...")
    print(" Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)