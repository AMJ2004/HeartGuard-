import sys
sys.path.insert(0, 'c:/Users/AISHWARYA/Downloads/heart diet final yr')

from app import app

client = app.test_client()

routes = [
    '/',
    '/home',
    '/index',
    '/about',
    '/stress',
    '/fitness',
    '/sleep',
    '/quit_smoking',
    '/assessment',
    '/bmi',
    '/diet',
]

print("Route Test Results:")
for route in routes:
    resp = client.get(route)
    status = "OK" if resp.status_code == 200 else f"FAIL({resp.status_code})"
    print(f"  {route}: {status}")
    if resp.status_code == 500:
        print(f"    Traceback: {resp.data.decode('utf-8')}")

# Test result POST flow
print("\nEnd-to-end Test (POST /result -> analysis.html):")
with client.session_transaction() as sess:
    sess['bmi'] = 24.5
resp = client.post('/result', data={
    'age': 45, 'gender': 1, 'sysBP': 120, 'diaBP': 80,
    'glucose': 100, 'totChol': 200
})
print(f"  POST /result: {'OK' if resp.status_code == 200 else f'FAIL({resp.status_code})'}")
if resp.status_code == 500:
    print(f"    Traceback: {resp.data.decode('utf-8')[:800]}")


