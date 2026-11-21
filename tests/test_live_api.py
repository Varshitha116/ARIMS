#!/usr/bin/env python3
"""Test all ARIMS API endpoints against a running server."""
import httpx
import json
import sys

BASE = 'http://127.0.0.1:8000'
client = httpx.Client(timeout=15)
results = []


def test(method, path, json_body=None, label=None):
    url = BASE + path
    tag = label or f'{method} {path}'
    try:
        if method == 'GET':
            r = client.get(url)
        else:
            r = client.post(url, json=json_body)
        status = 'PASS' if r.status_code == 200 else f'FAIL({r.status_code})'
        body = r.text[:150].replace('\n', ' ')
        results.append((tag, status, body))
    except Exception as e:
        results.append((tag, 'ERROR', str(e)[:80]))


# === 1. Health endpoints ===
test('GET', '/')
test('GET', '/health')

# === 2. Dashboard ===
test('GET', '/dashboard', label='GET /dashboard (HTML)')

# === 3. Detection endpoints ===
test('GET', '/api/detections')

# === 4. Agent endpoints ===
test('GET', '/api/agents/status')
test('GET', '/api/agents/logs')

# === 5. Simulator endpoints ===
test('GET', '/api/road-network')
test('POST', '/api/simulate', {
    'segment_id': 'TEST-001', 'initial_pci': 75.0, 'material': 'asphalt',
    'climate': 'tropical', 'traffic': 'high', 'years': 5, 'num_simulations': 50
})
test('POST', '/api/simulate/compare', {
    'segment_id': 'CMP-001', 'initial_pci': 80.0, 'years': 5, 'num_simulations': 50
})

# === 6. Maintenance endpoints ===
test('GET', '/api/maintenance/priority-queue')
test('POST', '/api/maintenance/prioritize', [
    {'segment_id': 'NH-44-KM-120', 'severity_score': 0.9, 'defect_types': ['D40_Pothole'],
     'traffic': 'very_high', 'risk_score': 0.8, 'estimated_cost': 50000, 'is_highway': True},
    {'segment_id': 'MH-SH-12', 'severity_score': 0.3, 'defect_types': ['D00_Longitudinal_Crack'],
     'traffic': 'low', 'risk_score': 0.2, 'estimated_cost': 8000, 'is_highway': False},
])
test('GET', '/api/maintenance/schedule')
test('GET', '/api/maintenance/db-health')

# === 7. NEW: Budget Agent ===
test('POST', '/api/maintenance/budget-allocate', {
    'total_budget': 500000,
    'jobs': [
        {'segment_id': 'NH-44-KM-120', 'severity_score': 0.9, 'defect_types': ['D40_Pothole'],
         'traffic': 'very_high', 'risk_score': 0.8, 'estimated_cost': 50000, 'is_highway': True},
        {'segment_id': 'MH-SH-12', 'severity_score': 0.4, 'defect_types': ['D00_Longitudinal_Crack'],
         'traffic': 'medium', 'risk_score': 0.3, 'estimated_cost': 15000},
        {'segment_id': 'KA-NH-48', 'severity_score': 0.7, 'defect_types': ['D20_Alligator_Crack'],
         'traffic': 'high', 'risk_score': 0.5, 'estimated_cost': 30000},
    ]
}, label='POST /api/maintenance/budget-allocate (Budget Agent)')

# === 8. NEW: Weather Agent (Monsoon) ===
test('POST', '/api/maintenance/weather-check', {
    'month': 7,
    'region': 'western_ghats',
    'jobs': [
        {'segment_id': 'KA-NH-48', 'severity_score': 0.5, 'defect_types': ['D20_Alligator_Crack'],
         'traffic': 'high', 'risk_score': 0.4, 'estimated_cost': 25000},
    ]
}, label='POST /api/maintenance/weather-check (Monsoon July)')

# === 9. NEW: Weather Agent (Post-Monsoon) ===
test('POST', '/api/maintenance/weather-check', {
    'month': 11,
    'region': 'inland',
    'jobs': [
        {'segment_id': 'AP-SH-5', 'severity_score': 0.6, 'defect_types': ['D40_Pothole'],
         'traffic': 'medium', 'risk_score': 0.5, 'estimated_cost': 20000},
    ]
}, label='POST /api/maintenance/weather-check (Post-Monsoon Nov)')

# === 10. Datasets ===
test('GET', '/api/datasets')

# === Print Results ===
print()
header = 'ARIMS API Endpoint Test Results'
print('=' * 90)
print(f'{header:^90}')
print('=' * 90)
passed = sum(1 for _, s, _ in results if s == 'PASS')
total = len(results)
for tag, status, body in results:
    icon = '[OK]' if status == 'PASS' else '[!!]'
    print(f'  {icon} {status:8s}  {tag}')
    print(f'           -> {body}')
    print()
print('=' * 90)
print(f'  TOTAL: {passed}/{total} endpoints passed')
print('=' * 90)

sys.exit(0 if passed == total else 1)
