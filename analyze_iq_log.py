import re

# === Path to your IQ correction log file ===
log_path = 'logs/iq_correction_log_2025-05-20.txt'  # Adjust if needed

pattern = re.compile(
    r'\[(\d{2}:\d{2}:\d{2})\] I = (-?\d+), Q = (-?\d+), DC power = ([\deE\+\-\.]+)'
)

entries = []

with open(log_path, 'r') as f:
    for line in f:
        match = pattern.match(line.strip())
        if match:
            timestamp, i_val, q_val, power = match.groups()
            entries.append((float(power), int(i_val), int(q_val), timestamp))
        else:
            print(f"⚠️ Skipping malformed line: {line.strip()}")

# Sort by DC power (lowest first)
entries.sort(key=lambda x: x[0])

# Show top 20
print("\n🔍 20 Best IQ Correction Values by lowest DC power:\n")
for idx, (power, i_val, q_val, timestamp) in enumerate(entries[:20], 1):
    print(f"{idx:2d}) [{timestamp}]  I = {i_val:5d}, Q = {q_val:5d}, DC Power = {power:.2e}")