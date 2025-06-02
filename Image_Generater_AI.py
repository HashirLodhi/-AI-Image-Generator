import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import time
import os
from flask import Flask, request, render_template_string, send_file, jsonify
import base64
import io

app = Flask(__name__)

# Enhanced HTML template with premium styling
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PICORA - AI Image Generator</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #7B68EE;
            --primary-dark: #5F4BDB;
            --secondary: #00CED1;
            --accent: #FF6B6B;
            --dark: #2D3748;
            --darker: #1A202C;
            --light: #F7FAFC;
            --lighter: #FFFFFF;
            --gray: #EDF2F7;
            --success: #48BB78;
            --danger: #F56565;
            --warning: #F6AD55;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            background-color: var(--light);
            color: var(--dark);
            line-height: 1.6;
        }
        
        .container {
            max-width: 1000px;
            margin: 2rem auto;
            padding: 3rem;
            background: var(--lighter);
            border-radius: 16px;
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.08);
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(0, 0, 0, 0.03);
        }
        
        .container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 8px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
        }
        
        .header {
            text-align: center;
            margin-bottom: 2.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        .logo {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
            letter-spacing: -1px;
        }
        
        .tagline {
            color: #718096;
            font-size: 1.1rem;
            font-weight: 400;
            max-width: 600px;
            margin: 0 auto;
        }
        
        .form-group {
            margin-bottom: 1.8rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.6rem;
            font-weight: 500;
            color: var(--darker);
            font-size: 0.95rem;
        }
        
        input[type="text"],
        input[type="number"],
        select {
            width: 100%;
            padding: 1rem 1.2rem;
            border: 2px solid var(--gray);
            border-radius: 10px;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: var(--lighter);
            color: var(--darker);
        }
        
        input[type="text"]:focus,
        input[type="number"]:focus,
        select:focus {
            border-color: var(--primary);
            outline: none;
            box-shadow: 0 0 0 4px rgba(123, 104, 238, 0.15);
        }
        
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            padding: 1.1rem 2.2rem;
            border-radius: 12px;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            text-align: center;
            text-decoration: none;
            box-shadow: 0 4px 15px rgba(123, 104, 238, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(123, 104, 238, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn-block {
            display: block;
            width: 100%;
        }
        
        .output {
            margin-top: 3rem;
            text-align: center;
            animation: fadeIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .output img {
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            border: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        .output img:hover {
            transform: scale(1.02);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        }
        
        .download-btn {
            display: inline-block;
            background: linear-gradient(135deg, var(--success), #38A169);
            color: white;
            padding: 1rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            text-decoration: none;
            margin-top: 2rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
        }
        
        .download-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(72, 187, 120, 0.4);
        }
        
        .progress-container {
            margin: 2rem 0;
            padding: 1.8rem;
            background: var(--gray);
            border-radius: 12px;
            box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        
        .progress-bar {
            height: 14px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border-radius: 8px;
            width: 0%;
            transition: width 0.5s ease;
            margin-bottom: 0.8rem;
            position: relative;
            overflow: hidden;
        }
        
        .progress-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, 
                           transparent, 
                           rgba(255, 255, 255, 0.5), 
                           transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .status-message {
            font-size: 0.95rem;
            color: #4A5568;
            font-weight: 500;
        }
        
        .error-message {
            color: var(--danger);
            background: rgba(245, 101, 101, 0.1);
            padding: 1.2rem;
            border-radius: 12px;
            margin-top: 2rem;
            border-left: 5px solid var(--danger);
            font-weight: 500;
        }
        
        /* Enhanced loading overlay */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(247, 250, 252, 0.95);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            backdrop-filter: blur(8px);
            opacity: 0;
            visibility: hidden;
            transition: all 0.4s ease;
        }
        
        .loading-overlay.active {
            opacity: 1;
            visibility: visible;
        }
        
        .loading-spinner {
            width: 100px;
            height: 100px;
            border: 10px solid rgba(123, 104, 238, 0.1);
            border-top: 10px solid var(--primary);
            border-bottom: 10px solid var(--secondary);
            border-radius: 50%;
            animation: spin 1.5s linear infinite;
            margin-bottom: 25px;
            position: relative;
        }
        
        .loading-spinner::after {
            content: '';
            position: absolute;
            top: -10px;
            left: -10px;
            right: -10px;
            bottom: -10px;
            border: 10px solid transparent;
            border-radius: 50%;
            border-top-color: rgba(123, 104, 238, 0.2);
            border-bottom-color: rgba(0, 206, 209, 0.2);
            animation: spinReverse 2s linear infinite;
        }
        
        .loading-text {
            font-size: 1.3rem;
            color: var(--darker);
            font-weight: 600;
            margin-top: 25px;
            text-align: center;
            max-width: 300px;
            line-height: 1.5;
        }
        
        .loading-progress {
            width: 300px;
            height: 8px;
            background: rgba(123, 104, 238, 0.1);
            border-radius: 4px;
            margin-top: 30px;
            overflow: hidden;
            position: relative;
        }
        
        .loading-progress-bar {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            transition: width 0.4s ease;
            position: relative;
        }
        
        .loading-progress-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, 
                           transparent, 
                           rgba(255, 255, 255, 0.7), 
                           transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes spinReverse {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(-360deg); }
        }
        
        footer {
            text-align: center;
            margin-top: 4rem;
            color: #718096;
            font-size: 0.9rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        /* Form grid layout */
        .form-row {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .form-col {
            flex: 1;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .container {
                padding: 2rem 1.5rem;
                margin: 1rem;
                border-radius: 12px;
            }
            
            .logo {
                font-size: 2.2rem;
            }
            
            .tagline {
                font-size: 1rem;
            }
            
            .form-row {
                flex-direction: column;
                gap: 1rem;
            }
        }
        
        /* Floating particles background effect */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }
        
        .particle {
            position: absolute;
            background: rgba(123, 104, 238, 0.1);
            border-radius: 50%;
            animation: float linear infinite;
        }
        
        @keyframes float {
            0% { transform: translateY(0) rotate(0deg); }
            100% { transform: translateY(-100vh) rotate(360deg); }
        }
        
        /* Tooltip styles */
        .tooltip {
            position: relative;
            display: inline-block;
            margin-left: 5px;
            cursor: help;
        }
        
        .tooltip .tooltip-text {
            visibility: hidden;
            width: 200px;
            background-color: var(--darker);
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8rem;
            font-weight: normal;
        }
        
        .tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
    </style>
</head>
<body>
    <!-- Floating particles background -->
    <div class="particles" id="particles"></div>
    
    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-spinner"></div>
        <div class="loading-text">Picora is crafting your visual masterpiece...</div>
        <div class="loading-progress">
            <div class="loading-progress-bar" id="loadingProgressBar"></div>
        </div>
    </div>

    <div class="container">
        <div class="header">
            <div class="logo">PICORA</div>
            <div class="tagline">Transform your imagination into stunning AI-generated visuals with just a few words</div>
        </div>
        
        <form method="POST" id="generateForm">
            <div class="form-group">
                <label for="prompt">Describe your vision
                    <span class="tooltip">?
                        <span class="tooltip-text">Be descriptive! Include details about style, colors, composition, etc.</span>
                    </span>
                </label>
                <input type="text" id="prompt" name="prompt" required 
                       placeholder="A cyberpunk cityscape at night with neon lights reflecting on wet streets..." 
                       value="{{ prompt|default('') }}">
            </div>
            
            <div class="form-row">
                <div class="form-col">
                    <div class="form-group">
                        <label for="width">Width (px)</label>
                        <input type="number" id="width" name="width" 
                               value="{{ width|default(512) }}" min="256" max="1024">
                    </div>
                </div>
                <div class="form-col">
                    <div class="form-group">
                        <label for="height">Height (px)</label>
                        <input type="number" id="height" name="height" 
                               value="{{ height|default(512) }}" min="256" max="1024">
                    </div>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-col">
                    <div class="form-group">
                        <label for="steps">Generation Steps
                            <span class="tooltip">?
                                <span class="tooltip-text">More steps = higher quality (but slower generation)</span>
                            </span>
                        </label>
                        <input type="number" id="steps" name="steps" 
                               value="{{ steps|default(20) }}" min="5" max="50">
                    </div>
                </div>
                <div class="form-col">
                    <div class="form-group">
                        <label for="guidance_scale">Creativity
                            <span class="tooltip">?
                                <span class="tooltip-text">Higher values follow your prompt more strictly</span>
                            </span>
                        </label>
                        <input type="number" id="guidance_scale" name="guidance_scale" 
                               step="0.1" value="{{ guidance_scale|default(7.5) }}" 
                               min="1" max="20">
                    </div>
                </div>
            </div>
            
            <button type="submit" class="btn btn-block">
                <span class="btn-text">Generate Image</span>
            </button>
        </form>

        {% if progress %}
        <div class="progress-container">
            <div class="progress-bar" id="progress-bar" style="width: {{ progress }}%"></div>
            <div class="status-message" id="status-message">{{ status_message }}</div>
        </div>
        {% endif %}

        {% if image %}
        <div class="output">
            <h2 style="margin-bottom: 1.5rem; color: var(--darker);">Your AI-Generated Artwork</h2>
            <img src="data:image/png;base64,{{ image }}" alt="AI Generated Image">
            <a href="/download/{{ image_name }}" class="download-btn">
                Download High-Quality PNG
            </a>
        </div>
        {% endif %}

        {% if error %}
        <div class="error-message">
            <strong>Error:</strong> {{ error }}
        </div>
        {% endif %}
        
        <footer>
            &copy; 2025 Picora AI. All generated images are royalty-free for personal and commercial use.
        </footer>
    </div>

    <script>
        // Create floating particles
        function createParticles() {
            const particlesContainer = document.getElementById('particles');
            const particleCount = 15;
            
            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.classList.add('particle');
                
                // Random size between 5 and 20px
                const size = Math.random() * 15 + 5;
                particle.style.width = `${size}px`;
                particle.style.height = `${size}px`;
                
                // Random position
                particle.style.left = `${Math.random() * 100}%`;
                particle.style.top = `${Math.random() * 100}%`;
                
                // Random opacity
                particle.style.opacity = Math.random() * 0.3 + 0.1;
                
                // Random animation duration
                const duration = Math.random() * 30 + 20;
                particle.style.animationDuration = `${duration}s`;
                
                // Random delay
                particle.style.animationDelay = `${Math.random() * 10}s`;
                
                particlesContainer.appendChild(particle);
            }
        }
        
        // Initialize particles
        createParticles();
        
        // Form submission handler
        document.getElementById('generateForm').addEventListener('submit', function() {
            const overlay = document.getElementById('loadingOverlay');
            const progressBar = document.getElementById('loadingProgressBar');
            
            // Show loading overlay
            overlay.classList.add('active');
            
            // Simulate progress (we'll update this with actual progress later)
            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 5;
                if (progress >= 90) {
                    clearInterval(interval);
                }
                progressBar.style.width = `${progress}%`;
            }, 300);
            
            return true;
        });
    </script>
</body>
</html>
'''

class ProgressTracker:
    def __init__(self, total_steps):
        self.current_step = 0
        self.total_steps = total_steps
    
    def get_generation_progress(self, step, timestep, latents):
        self.current_step = step + 1
    
    def get_generation_progress_percent(self):
        return min(100, int((self.current_step / self.total_steps) * 100))

# Initialize model once at startup
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    safety_checker=None
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cpu")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            prompt = request.form['prompt']
            width = int(request.form.get('width', 512))
            height = int(request.form.get('height', 512))
            num_inference_steps = int(request.form.get('steps', 20))
            guidance_scale = float(request.form.get('guidance_scale', 7.5))
            
            tracker = ProgressTracker(num_inference_steps)
            
            with torch.no_grad():
                image = pipe(
                    prompt, 
                    width=width, 
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    callback=tracker.get_generation_progress,
                    callback_steps=1
                ).images[0]
            
            img_io = io.BytesIO()
            image.save(img_io, 'PNG')
            img_io.seek(0)
            img_data = img_io.getvalue()
            
            timestamp = int(time.time())
            image_name = f"picora_{timestamp}.png"
            
            if not hasattr(app, 'generated_images'):
                app.generated_images = {}
            app.generated_images[image_name] = img_data
            
            image_base64 = base64.b64encode(img_data).decode('utf-8')
            
            return render_template_string(HTML_TEMPLATE,
                image=image_base64,
                image_name=image_name,
                **request.form
            )
            
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, 
                error=f"An error occurred while generating your image: {str(e)}",
                **request.form
            )
    
    return render_template_string(HTML_TEMPLATE)

@app.route('/download/<image_name>')
def download(image_name):
    if hasattr(app, 'generated_images') and image_name in app.generated_images:
        img_data = app.generated_images[image_name]
        return send_file(
            io.BytesIO(img_data),
            mimetype='image/png',
            as_attachment=True,
            download_name=image_name
        )
    return "Image not found", 404

if __name__ == '__main__':
    app.run(debug=True)