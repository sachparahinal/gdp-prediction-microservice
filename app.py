from flask import Flask, request, jsonify
import torch
import torch.nn as nn

app = Flask(__name__)

# Model definition (same as training)
class Polynomial3(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(()))
        self.b = nn.Parameter(torch.randn(()))
        self.c = nn.Parameter(torch.randn(()))
        self.d = nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

# Load model
model = Polynomial3()
checkpoint = torch.load('gdp_polynomial_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Normalization parameters
X_MIN = 1.0
X_MAX = 30.0

def normalize_rank(rank):
    """Normalize country rank to [-1, 1] range"""
    return 2 * (rank - X_MIN) / (X_MAX - X_MIN) - 1

@app.route('/')
def home():
    return jsonify({
        "service": "GDP Prediction Microservice",
        "description": "Predicts GDP per capita based on country economic rank",
        "endpoints": {
            "/": "Service information (this page)",
            "/predict": "POST - Predict GDP for a given rank",
            "/health": "GET - Health check"
        },
        "example": {
            "url": "/predict",
            "method": "POST",
            "body": {"rank": 5},
            "response": {"rank": 5, "predicted_gdp": 92.6, "unit": "thousands USD"}
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'rank' not in data:
            return jsonify({"error": "Missing 'rank' in request body"}), 400
        
        rank = float(data['rank'])
        
        if rank < 1 or rank > 30:
            return jsonify({"error": "Rank must be between 1 and 30"}), 400
        
        # Normalize and predict
        x_normalized = normalize_rank(rank)
        x_tensor = torch.tensor([[x_normalized]], dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(x_tensor).item()
        
        return jsonify({
            "rank": int(rank),
            "predicted_gdp": round(prediction, 2),
            "unit": "thousands USD"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model": "loaded"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
