from flask import Flask, request, jsonify, send_from_directory, session, render_template, redirect, url_for
from flask_cors import CORS
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import requests
import logging
import math


# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='assets', static_url_path='/static/assets')
app.secret_key = 'your-secret-key'  
CORS(app)  


os.makedirs('models', exist_ok=True)


FEATURE_COLUMNS = ['Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']


users = {'user1': 'password123'}


def verify_model(model, model_type):
    try:
        if model_type == "engine":
            logger.debug("Verifying engine model...")
            test_input = np.array([[3.3, 6.65, 2.33, 77.64, 78.43]], dtype=np.float32)  
            if scaler is None:
                raise ValueError("Scaler is not loaded")
            logger.debug(f"Test input shape: {test_input.shape}")
            test_scaled = scaler.transform(test_input)
            test_reshaped = np.expand_dims(test_scaled, axis=-1)
            logger.debug(f"Reshaped input shape: {test_reshaped.shape}")
            prediction = model.predict([test_reshaped, test_reshaped, test_reshaped], verbose=0)
            logger.debug(f"Prediction shape: {prediction.shape}, value: {prediction[0][0]}")
            return isinstance(prediction, np.ndarray) and prediction.size > 0
        return False
    except Exception as e:
        logger.error(f"Model verification failed for {model_type} model: {e}")
        return False


def load_engine_model():
    model_path = os.path.join('models', 'engine_health_model.keras')
    try:
        if os.path.exists(model_path):
            logger.debug(f"Loading model from {model_path}")
            model = keras.models.load_model(model_path)
            
            # Test prediction with dummy data
            test_input = np.array([[3.3, 6.65, 2.33, 77.64, 78.43]], dtype=np.float32)
            try:
                test_pred = model.predict(test_input, verbose=0)
                logger.debug(f"Test prediction successful: {test_pred}")
            except Exception as e:
                logger.error(f"Test prediction failed: {e}")
                raise
                
            logger.info("Engine health prediction model loaded and tested successfully")
            return model
        else:
            logger.error(f"Engine health model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        logger.error(f"Error loading engine model: {e}")
        raise


try:
    scaler_path = os.path.join('models', 'advanced_scaler.pkl')
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found at {scaler_path}")
        scaler = None
    else:
        logger.debug(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        # Test scaler with dummy data
        test_input = np.array([[3.3, 6.65, 2.33, 77.64, 78.43]], dtype=np.float32)
        try:
            test_scaled = scaler.transform(test_input)
            logger.debug(f"Test scaling successful: {test_scaled}")
        except Exception as e:
            logger.error(f"Test scaling failed: {e}")
            raise
            
        logger.info("Scaler loaded and tested successfully")
except Exception as e:
    logger.error(f"Error loading scaler: {e}")
    scaler = None


try:
    engine_model = load_engine_model()
except Exception as e:
    logger.error(f"Could not load engine model: {e}")
    engine_model = None


all_cars = [
    {
        "id": "1",
        "make": "Toyota",
        "model": "Camry",
        "year": 2019,
        "mileage": 35000,
        "engine": "Petrol",
        "color": "Silver",
        "notes": "Regular maintenance up to date. Last serviced on May 2, 2025.",
        "status": "Good"
    },
    {
        "id": "2",
        "make": "Honda",
        "model": "Civic",
        "year": 2020,
        "mileage": 22000,
        "engine": "Petrol",
        "color": "Blue",
        "notes": "Minor scratch on rear bumper. Tire pressure needs checking.",
        "status": "Good"
    },
    {
        "id": "3",
        "make": "Ford",
        "model": "Explorer",
        "year": 2018,
        "mileage": 48000,
        "engine": "Diesel",
        "color": "Black",
        "notes": "Check engine light came on last week. Scheduled for service.",
        "status": "Warning"
    }
]


def get_chatbot_response(message):
    message = message.lower().strip()
    if "engine" in message or "health" in message:
        return "I can help you check your engine's health! Please visit the 'Check Engine Health' section and enter the required parameters."
    elif "workshop" in message or "repair" in message:
        return "Looking for a workshop? Go to the 'Workshops Nearby' section to find trusted auto repair shops in your area."
    elif "product" in message or "parts" in message:
        return "You can browse automotive parts in the 'Shop Products' section. Search for the parts you need!"
    elif "car" in message or "vehicle" in message:
        return "Want to add a vehicle or check your cars? Use the dashboard to manage your vehicles."
    elif "hi" in message or "hello" in message:
        return "Hello! I'm CarBot, your vehicle assistant. How can I assist you today?"
    elif "logout" in message or "sign out" in message:
        return "You can log out by clicking the 'LOGOUT' link in the navigation bar."
    else:
        return "I'm not sure how to help with that. Try asking about engine health, workshops, products, or your vehicles!"


@app.route('/<path:path>')
def serve_static(path):
    if path.startswith('assets/'):
        return send_from_directory('assets', 'edit1.png')
    return send_from_directory('.', path)

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('major.html')

@app.route('/engine-health')
def engine_health():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('engine_health.html')

@app.route('/workshops')
def workshops():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('workshops.html')

@app.route('/products')
def products():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('products.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        with open('products.json', 'r') as file:
            data = json.load(file)
        return jsonify(data), 200
    except FileNotFoundError:
        return jsonify({"error": "Products data not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/workshops', methods=['GET'])
def get_workshops():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude are required'}), 400
    overpass_query = f"""
    [out:json];
    node(around:5000,{lat},{lon})["shop"="car_repair"];
    out body;
    """
    url = 'https://overpass-api.de/api/interpreter'
    params = {'data': overpass_query}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        workshops = []
        for node in data.get('elements', []):
            tags = node.get('tags', {})
            address_parts = []
            if tags.get('addr:housenumber'):
                address_parts.append(tags['addr:housenumber'])
            if tags.get('addr:street'):
                address_parts.append(tags['addr:street'])
            if tags.get('addr:city'):
                address_parts.append(tags['addr:city'])
            if tags.get('addr:state'):
                address_parts.append(tags['addr:state'])
            if tags.get('addr:postcode'):
                address_parts.append(tags['addr:postcode'])
            address = ', '.join(address_parts) if address_parts else 'Address not fully specified in OpenStreetMap'
            workshops.append({
                'name': tags.get('name', 'Auto Repair Shop'),
                'address': address,
                'lat': node['lat'],
                'lon': node['lon']
            })
        return jsonify(workshops)
    except requests.RequestException as e:
        logger.error(f"Error fetching workshops: {e}")
        return jsonify({'error': 'Failed to fetch workshops'}), 500

@app.route('/add-vehicle', methods=['POST'])
def add_vehicle():
    data = request.json
    car_name_parts = data.get('carName', '').split()
    make = car_name_parts[0] if len(car_name_parts) > 0 else "Unknown"
    model = ' '.join(car_name_parts[1:]) if len(car_name_parts) > 1 else data.get('carName', 'Unknown')
    new_car = {
        "id": str(len(all_cars) + 1),
        "make": make,
        "model": model,
        "year": 2021,
        'mileage': int(data.get('mileage', 0)),
        "engine": data.get('engine', 'Unknown'),
        "color": "Silver",
        "notes": "Added from dashboard",
        "status": "Good"
    }
    all_cars.append(new_car)
    logger.info(f"Vehicle added: {new_car}")
    return jsonify({"success": True, "message": "Vehicle added successfully"})

@app.route('/get-cars', methods=['GET'])
def get_cars():
    return jsonify({"cars": all_cars})

@app.route('/add-car', methods=['POST'])
def add_car():
    data = request.json
    new_car = data
    new_car["id"] = str(len(all_cars) + 1)
    all_cars.append(new_car)
    logger.info(f"Car added: {new_car}")
    return jsonify({
        "success": True,
        "message": "Car added successfully",
        "car": new_car
    })

@app.route('/model-info', methods=['GET'])
def get_model_info():
    try:
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        model_info['models']['engine_health_model']['status'] = 'loaded' if engine_model is not None else 'unavailable'
        return jsonify(model_info)
    except Exception as e:
        logger.error(f"Error fetching model info: {e}")
        return jsonify({
            "error": "Failed to retrieve model information",
            "message": str(e)
        }), 500

# Ensure value ranges exist
value_ranges = {
    "mean_values": [3.3, 6.65, 2.33, 77.64, 78.43],  # Mean values for each parameter
    "std_values": [0.5, 0.8, 0.4, 5.0, 5.0]  # Standard deviation values
}

# Save value ranges if file doesn't exist
if not os.path.exists('models/advanced_value_ranges.json'):
    os.makedirs('models', exist_ok=True)
    with open('models/advanced_value_ranges.json', 'w') as f:
        json.dump(value_ranges, f)
    logger.info("Created default value ranges file")

@app.route('/predict-engine-health', methods=['POST'])
def predict_engine_health():
    global engine_model, scaler
    
    try:
        logger.debug("Received prediction request")
        
        # Check if model and scaler are available
        if engine_model is None or scaler is None:
            logger.error("Model or scaler not initialized")
            # Try to reinitialize
            if not initialize_model_and_scaler():
                return jsonify({
                    "status": "Error",
                    "message": "Engine health prediction model or scaler is not available",
                    "error": "Model initialization failed"
                }), 503
        
        data = request.json
        logger.debug(f"Request data: {data}")
        
        # Get and validate input values
        input_values = []
        try:
            for feature in FEATURE_COLUMNS:
                if feature not in data:
                    raise ValueError(f"Missing parameter: {feature}")
                value = float(data[feature])
                if not np.isfinite(value):
                    raise ValueError(f"Invalid value for {feature}: {value}")
                input_values.append(value)
            logger.debug(f"Processed input values: {input_values}")
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return jsonify({
                "status": "Error",
                "message": str(e),
                "error": "Invalid input"
            }), 400

        # Create input array and make prediction
        try:
            input_features = np.array([input_values], dtype=np.float32)
            logger.debug(f"Input features shape: {input_features.shape}")
            
            # Scale input
            input_scaled = scaler.transform(input_features)
            logger.debug(f"Scaled input: {input_scaled}")
            
            # Make prediction
            health_score = float(engine_model.predict(input_scaled, verbose=0)[0][0])
            logger.debug(f"Health score: {health_score}")

            # Parameter importance weights
            parameter_weights = {
                'Lub oil pressure': 1.3,
                'Fuel pressure': 1.2,
                'Coolant pressure': 1.1,
                'lub oil temp': 1.0,
                'Coolant temp': 1.0
            }

            # Calculate weighted deviations
            weighted_deviations = []
            total_weighted_deviation = 0
            max_weighted_deviation = 0
            critical_weighted_deviations = 0

            for i, (value, mean, std) in enumerate(zip(input_values, value_ranges['mean_values'], value_ranges['std_values'])):
                parameter = FEATURE_COLUMNS[i]
                weight = parameter_weights[parameter]
                
                # Calculate base z-score
                z_score = abs((value - mean) / std)
                # Apply weight
                weighted_z_score = z_score * weight
                
                total_weighted_deviation += weighted_z_score
                max_weighted_deviation = max(max_weighted_deviation, weighted_z_score)
                
                # Calculate percentage difference from mean
                percent_diff = ((value - mean) / mean) * 100
                
                # Enhanced significance determination with weighted thresholds
                if weighted_z_score > 2.2 or abs(percent_diff) > 45:
                    significance = 'Critical'
                    critical_weighted_deviations += 1
                elif weighted_z_score > 1.8 or abs(percent_diff) > 35:
                    significance = 'High'
                elif weighted_z_score > 1.4 or abs(percent_diff) > 25:
                    significance = 'Medium'
                else:
                    significance = 'Low'
                    
                weighted_deviations.append({
                    'parameter': parameter,
                    'value': value,
                    'mean': mean,
                    'base_deviation': z_score,
                    'weighted_deviation': weighted_z_score,
                    'weight': weight,
                    'percent_diff': percent_diff,
                    'significance': significance
                })

            # Sort deviations by weighted severity
            weighted_deviations.sort(key=lambda x: (
                0 if x['significance'] == 'Critical' else
                1 if x['significance'] == 'High' else
                2 if x['significance'] == 'Medium' else 3,
                -x['weighted_deviation']
            ))

            # Enhanced risk level determination with sub-levels
            if max_weighted_deviation <= 1.4 and critical_weighted_deviations == 0:
                risk_level = "Low"
                risk_sublevel = "Normal"
            elif max_weighted_deviation <= 1.6 and critical_weighted_deviations == 0:
                risk_level = "Low"
                risk_sublevel = "Elevated"
            elif max_weighted_deviation <= 1.8 and critical_weighted_deviations <= 1:
                risk_level = "Medium"
                risk_sublevel = "Caution"
            elif max_weighted_deviation <= 2.0 and critical_weighted_deviations <= 1:
                risk_level = "Medium"
                risk_sublevel = "Warning"
            else:
                risk_level = "High"
                risk_sublevel = "Critical" if critical_weighted_deviations >= 2 else "Severe"

            # Calculate deviation factor with exponential scaling
            deviation_factor = 0
            if max_weighted_deviation > 1.8:
                deviation_factor = min(0.25, 0.1 * (math.exp(max_weighted_deviation - 1.8) - 1))
            elif critical_weighted_deviations >= 2:
                deviation_factor = 0.2

            adjusted_threshold = 0.55 - deviation_factor

            # Enhanced status determination with transition zones
            is_transition_zone = 1.7 <= max_weighted_deviation <= 1.9
            has_concerning_params = sum(1 for d in weighted_deviations if d['significance'] in ['High', 'Critical']) >= 2

            if (health_score <= adjusted_threshold and max_weighted_deviation < 1.8) or (max_weighted_deviation < 1.6 and critical_weighted_deviations == 0):
                status = "Healthy"
                if is_transition_zone:
                    message = "Engine is operating in acceptable range but showing signs of stress. Monitor closely."
                else:
                    message = "Engine is running within normal parameters. No immediate issues detected."
                    if max_weighted_deviation > 1.4:
                        message += " Some parameters are elevated but within acceptable ranges."

                # Enhanced confidence calculation for healthy status using logarithmic scaling
                base_confidence = 100
                health_penalty = ((health_score / adjusted_threshold) * 35)  # Reduced from 40
                deviation_penalty = math.log(1 + max_weighted_deviation) * 15  # Logarithmic scaling
                confidence = base_confidence - health_penalty - deviation_penalty
                
                # Adjust confidence in transition zone
                if is_transition_zone:
                    confidence = max(confidence * 0.85, 50)  # Reduce confidence in transition zone
            else:
                status = "Unhealthy"
                if critical_weighted_deviations >= 2:
                    message = "Multiple critical issues detected. Immediate maintenance required."
                elif has_concerning_params:
                    message = "Multiple concerning parameters detected. Maintenance recommended soon."
                else:
                    message = "Parameters outside normal range. Schedule maintenance check."

                # Enhanced confidence calculation for unhealthy status using exponential scaling
                if critical_weighted_deviations > 0 or max_weighted_deviation > 1.8:
                    base_confidence = 55 + (critical_weighted_deviations * 8)
                    severity_factor = min(25, math.exp(max_weighted_deviation - 1.6) * 8)
                    confidence = min(base_confidence + severity_factor, 100)
                else:
                    confidence = 50 + (math.log(1 + max_weighted_deviation) * 12)

            # Apply dynamic confidence floor based on parameter severity
            min_confidence = 45 + (5 * sum(1 for d in weighted_deviations if d['significance'] in ['High', 'Critical']))
            confidence = max(min(confidence, 100), min_confidence)

            # Enhanced response data
            response_data = {
                "status": status,
                "confidence": round(confidence, 1),
                "risk_level": risk_level,
                "risk_sublevel": risk_sublevel,
                "message": message,
                "deviations": [{
                    'parameter': d['parameter'],
                    'value': d['value'],
                    'mean': d['mean'],
                    'deviation': d['weighted_deviation'],
                    'percent_diff': d['percent_diff'],
                    'significance': d['significance']
                } for d in weighted_deviations],
                "model_info": "Advanced engine health prediction model with weighted parameters and dynamic thresholds",
                "input_parameters": dict(zip(FEATURE_COLUMNS, input_values)),
                "model_version": "4.0.0",
                "analysis_details": {
                    "max_weighted_deviation": round(max_weighted_deviation, 3),
                    "critical_deviations": critical_weighted_deviations,
                    "confidence_basis": {
                        "base_threshold": 0.55,
                        "adjusted_threshold": round(adjusted_threshold, 3),
                        "deviation_factor": round(deviation_factor, 3)
                    }
                }
            }
            logger.debug(f"Sending response: {response_data}")
            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Prediction processing error: {e}", exc_info=True)
            return jsonify({
                "status": "Error",
                "message": "Error processing prediction",
                "error": str(e)
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}", exc_info=True)
        return jsonify({
            "status": "Error",
            "message": "Failed to process prediction request",
            "error": str(e)
        }), 500


@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.json
        user_message = data.get('message', '')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        response = get_chatbot_response(user_message)
        return jsonify({"reply": response}), 200
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({"error": str(e)}), 500

def initialize_model_and_scaler():
    global engine_model, scaler
    
    logger.info("Starting model and scaler initialization...")
    
    # Load scaler first
    try:
        scaler_path = os.path.join('models', 'scaler.pkl')
        logger.debug(f"Attempting to load scaler from: {scaler_path}")
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
            
            # Test scaler
            test_input = np.array([[3.3, 6.65, 2.33, 77.64, 78.43]], dtype=np.float32)
            test_scaled = scaler.transform(test_input)
            logger.info("Scaler test successful")
        else:
            logger.error(f"Scaler file not found at: {scaler_path}")
            return False
    except Exception as e:
        logger.error(f"Failed to load scaler: {str(e)}")
        return False

    # Load model
    try:
        model_path = os.path.join('models', 'engine_health_model.keras')
        logger.debug(f"Attempting to load model from: {model_path}")
        
        if os.path.exists(model_path):
            engine_model = keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
            
            # Test model
            test_input = np.array([[3.3, 6.65, 2.33, 77.64, 78.43]], dtype=np.float32)
            test_scaled = scaler.transform(test_input)
            test_pred = engine_model.predict(test_scaled, verbose=0)
            logger.info(f"Model test prediction successful: {test_pred}")
        else:
            logger.error(f"Model file not found at: {model_path}")
            return False
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

    logger.info("Model and scaler initialization completed successfully")
    return True


if not initialize_model_and_scaler():
    logger.error("Failed to initialize model and scaler")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)