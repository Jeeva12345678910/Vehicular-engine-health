import numpy as np

# Test scenarios
scenarios = [
    {
        "name": "Your Current Values",
        "values": {
            "Lub oil pressure": 2.493591821,
            "Fuel pressure": 11.79092738,
            "Coolant pressure": 3.178980794,
            "lub oil temp": 84.14416293,
            "Coolant temp": 81.6321865
        }
    },
    {
        "name": "Normal Operating Values",
        "values": {
            "Lub oil pressure": 3.30,
            "Fuel pressure": 6.66,
            "Coolant pressure": 2.34,
            "lub oil temp": 77.64,
            "Coolant temp": 78.43
        }
    },
    {
        "name": "Critical High Temperature",
        "values": {
            "Lub oil pressure": 3.30,
            "Fuel pressure": 6.66,
            "Coolant pressure": 2.34,
            "lub oil temp": 95.0,  # Very high
            "Coolant temp": 98.0   # Very high
        }
    },
    {
        "name": "Low Pressure Warning",
        "values": {
            "Lub oil pressure": 1.5,  # Very low
            "Fuel pressure": 3.0,     # Very low
            "Coolant pressure": 1.0,  # Very low
            "lub oil temp": 77.64,
            "Coolant temp": 78.43
        }
    }
]

mean_values = {
    "Lub oil pressure": 3.30,
    "Fuel pressure": 6.66,
    "Coolant pressure": 2.34,
    "lub oil temp": 77.64,
    "Coolant temp": 78.43
}

std_values = {
    "Lub oil pressure": 1.02,
    "Fuel pressure": 2.77,
    "Coolant pressure": 1.03,
    "lub oil temp": 3.11,
    "Coolant temp": 6.20
}

def analyze_scenario(values, name):
    print("\n" + "=" * 70)
    print(f"Scenario: {name}")
    print("=" * 70)

    max_deviation = 0
    critical_count = 0
    deviations = []

    for param in values:
        value = values[param]
        mean = mean_values[param]
        std = std_values[param]
        
        z_score = abs((value - mean) / std)
        percent_diff = ((value - mean) / mean) * 100
        
        if z_score > 2.5 or abs(percent_diff) > 50:
            significance = "HIGH"
        elif z_score > 1.5 or abs(percent_diff) > 30:
            significance = "MEDIUM"
        else:
            significance = "LOW"
            
        if z_score > 2.5:
            critical_count += 1
        max_deviation = max(max_deviation, z_score)
        
        deviations.append({
            'parameter': param,
            'value': value,
            'mean': mean,
            'z_score': z_score,
            'percent_diff': percent_diff,
            'significance': significance
        })
    
    # Sort deviations by significance and z-score
    deviations.sort(key=lambda x: (-len(x['significance']), -x['z_score']))
    
    # Print parameter analysis
    print("\nParameter Analysis:")
    for dev in deviations:
        print(f"\n{dev['parameter']}:")
        print(f"  Current Value: {dev['value']:.2f}")
        print(f"  Mean Value: {dev['mean']:.2f}")
        print(f"  Z-Score: {dev['z_score']:.2f}σ")
        print(f"  Percent Difference: {dev['percent_diff']:+.1f}%")
        print(f"  Significance: {dev['significance']}")

    # Determine status
    if max_deviation < 2.5 and critical_count == 0:
        status = "HEALTHY"
        if max_deviation > 1.5:
            message = "Engine is running within normal parameters, though some values are elevated but acceptable."
        else:
            message = "Engine is running well within normal parameters."
    else:
        status = "UNHEALTHY"
        message = "Critical deviations detected. Maintenance recommended."

    # Determine risk level
    if max_deviation <= 1.5 and critical_count == 0:
        risk = "LOW"
    elif max_deviation <= 2.5 and critical_count <= 1:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    print("\nOverall Assessment:")
    print(f"Maximum Deviation: {max_deviation:.2f}σ")
    print(f"Critical Deviations (>2.5σ): {critical_count}")
    print(f"Status: {status}")
    print(f"Risk Level: {risk}")
    print(f"Message: {message}")

# Run analysis for each scenario
for scenario in scenarios:
    analyze_scenario(scenario['values'], scenario['name']) 