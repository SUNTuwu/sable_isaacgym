"""Exercise real dashboard authentication/API without displaying secrets."""
import http.cookiejar,json,urllib.parse,urllib.request
from pathlib import Path
jar=http.cookiejar.CookieJar();client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
password=Path('/root/gpufree-data/config/sable-legged-gym/dashboard.password').read_text().strip()
body=urllib.parse.urlencode({'password':password}).encode();password=None
response=client.open(urllib.request.Request('http://127.0.0.1:11111/login',data=body),timeout=10);body=None
assert response.status==200
for route in ['status','learning-curve']:
    response=client.open('http://127.0.0.1:11111/api/'+route,timeout=10);data=json.loads(response.read())
    print('HTTP_OK',route,'iteration',data.get('run',{}).get('iteration') if isinstance(data,dict) else len(data))
