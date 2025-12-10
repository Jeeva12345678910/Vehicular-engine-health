import requests
import json

# Test scenarios
test_scenarios = [
    {
        "name": "Normal Operating Conditions",
        "data": {
            "Lub oil pressure": 3.3,
            "Fuel pressure": 6.65,
            "Coolant pressure": 2.33,
            "lub oil temp": 77.64,
            "Coolant temp": 78.43
        }
    },
    {
        "name": "High Pressure Scenario",
        "data": {
            "Lub oil pressure": 4.5,
            "Fuel pressure": 8.0,
            "Coolant pressure": 3.0,
            "lub oil temp": 77.0,
            "Coolant temp": 80.0
        }
    },
    {
        "name": "Low Pressure Scenario",
        "data": {
            "Lub oil pressure": 2.0,
            "Fuel pressure": 5.0,
            "Coolant pressure": 1.5,
            "lub oil temp": 77.0,
            "Coolant temp": 80.0
        }
    },
    {
        "name": "High Temperature Scenario",
        "data": {
            "Lub oil pressure": 3.3,
            "Fuel pressure": 6.65,
            "Coolant pressure": 2.33,
            "lub oil temp": 85.0,
            "Coolant temp": 88.0
        }
    }
]

# Test each scenario
for scenario in test_scenarios:
    print("\n" + "="*50)
    print(f"Testing: {scenario['name']}")
    print("="*50)
    
    try:
        response = requests.post('http://localhost:5000/predict-engine-health', 
                               json=scenario['data'])
        
        print("\nRequest data:", json.dumps(scenario['data'], indent=2))
        print("\nStatus code:", response.status_code)
        
        if response.ok:
            result = response.json()
            print("\nResponse:")
            print(f"Status: {result['status']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Message: {result['message']}")
            print("\nParameter Deviations:")
            for dev in result['deviations']:
                print(f"- {dev['parameter']}:")
                print(f"  Value: {dev['value']:.2f}")
                print(f"  Mean: {dev['mean']:.2f}")
                print(f"  Deviation: {dev['deviation']:.2f}")
                print(f"  Significance: {dev['significance']}")
        else:
            print("\nError Response:", json.dumps(response.json(), indent=2))
            
    except Exception as e:
        print("Error:", e) 