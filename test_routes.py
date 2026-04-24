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


