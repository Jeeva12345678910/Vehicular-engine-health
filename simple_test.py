import numpy as np

test_values = {
    "Lub oil pressure": 2.493591821,
    "Fuel pressure": 11.79092738,
    "Coolant pressure": 3.178980794,
    "lub oil temp": 84.14416293,
    "Coolant temp": 81.6321865
}

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

print("\nParameter Analysis:")
print("=" * 50)

max_deviation = 0
critical_count = 0

for param in test_values:
    value = test_values[param]
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
    
    print(f"\n{param}:")
    print(f"  Current Value: {value:.2f}")
    print(f"  Mean Value: {mean:.2f}")
    print(f"  Standard Deviation: {std:.2f}")
    print(f"  Z-Score: {z_score:.2f}σ")
    print(f"  Percent Difference: {percent_diff:+.1f}%")
    print(f"  Significance: {significance}")

print("\n" + "=" * 50)
print("\nOverall Assessment:")
print(f"Maximum Deviation: {max_deviation:.2f}σ")
print(f"Critical Deviations (>2.5σ): {critical_count}")

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

print(f"\nStatus: {status}")
print(f"Message: {message}")

# Determine risk level
if max_deviation <= 1.5 and critical_count == 0:
    risk = "LOW"
elif max_deviation <= 2.5 and critical_count <= 1:
    risk = "MEDIUM"
else:
    risk = "HIGH"

print(f"Risk Level: {risk}") 