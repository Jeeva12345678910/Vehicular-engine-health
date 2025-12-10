import requests
import json

# Test cases with different scenarios
test_cases = [
    {
        "name": "Mean Values Test",
        "data": {
            "Lub oil pressure": 3.30,
            "Fuel pressure": 6.65,
            "Coolant pressure": 2.33,
            "lub oil temp": 77.64,
            "Coolant temp": 78.43
        }
    },
    {
        "name": "Lower Bound Test",
        "data": {
            "Lub oil pressure": 2.28,  # mean - 1 std
            "Fuel pressure": 3.89,     # mean - 1 std
            "Coolant pressure": 1.30,  # mean - 1 std
            "lub oil temp": 74.53,     # mean - 1 std
            "Coolant temp": 72.22      # mean - 1 std
        }
    },
    {
        "name": "Upper Bound Test",
        "data": {
            "Lub oil pressure": 4.32,  # mean + 1 std
            "Fuel pressure": 9.42,     # mean + 1 std
            "Coolant pressure": 3.37,  # mean + 1 std
            "lub oil temp": 80.75,     # mean + 1 std
            "Coolant temp": 84.63      # mean + 1 std
        }
    }
]

# Test each case
for test_case in test_cases:
    print("\n" + "="*50)
    print(f"Testing: {test_case['name']}")
    print("="*50)
    
    try:
        response = requests.post('http://localhost:5000/predict-engine-health', 
                               json=test_case['data'])
        
        print("\nRequest data:", json.dumps(test_case['data'], indent=2))
        print("\nStatus code:", response.status_code)
        
        if response.ok:
            result = response.json()
            print("\nResponse:")
            print(f"Status: {result['status']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Message: {result['message']}")
        else:
            print("\nError Response:", json.dumps(response.json(), indent=2))
            
    except Exception as e:
        print("Error:", e) 