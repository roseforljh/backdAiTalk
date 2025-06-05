import httpx
import sys

target_url = "https://www.googleapis.com/discovery/v1/apis"

try:
    response = httpx.get(target_url, timeout=20.0)
    
    if response.status_code == 200:
        pass
    else:
        pass

except httpx.TimeoutException as e:
    pass
except httpx.RequestError as e:
    pass
except Exception as e:
    pass