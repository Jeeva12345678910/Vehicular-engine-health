import urllib3
import json
from urllib3.exceptions import HTTPError

http = urllib3.PoolManager()

test_data = {
    "Lub oil pressure": 2.493591821,
    "Fuel pressure": 11.79092738,
    "Coolant pressure": 3.178980794,
    "lub oil temp": 84.14416293,
    "Coolant temp": 81.6321865
}

try:
    encoded_data = json.dumps(test_data).encode('utf-8')
    response = http.request(
        'POST',
        'http://localhost:5000/predict-engine-health',
        body=encoded_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print("\nRequest data:", json.dumps(test_data, indent=2))
    print("\nStatus code:", response.status)
    
    if response.status == 200:
        result = json.loads(response.data.decode('utf-8'))
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
            if 'percent_diff' in dev:
                print(f"  Percent Difference: {dev['percent_diff']:.1f}%")
        print("\nThreshold Info:")
        print(f"Base threshold: {result['threshold_info']['base']}")
        print(f"Adjusted threshold: {result['threshold_info']['adjusted']:.3f}")
        print(f"Deviation factor: {result['threshold_info']['deviation_factor']:.3f}")
        print(f"Max deviation: {result['threshold_info']['max_deviation']:.3f}")
        print(f"Critical deviations: {result['threshold_info']['critical_deviations']}")
    else:
        print("\nError Response:", response.data.decode('utf-8'))
        
except Exception as e:
    print("Error:", e) 