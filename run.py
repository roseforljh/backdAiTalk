from mangum import Mangum
from eztalk_proxy.main import app

# Mangum a-dapter will handle the request and response conversion
# between Cloudflare Workers and our FastAPI app.
handler = Mangum(app)