import sys,tempfile,threading,http.cookiejar,urllib.request,urllib.parse,urllib.error,json,secrets
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dashboard,operator_feedback
from http.server import ThreadingHTTPServer
with tempfile.TemporaryDirectory(dir='/root/gpufree-data/tmp',prefix='sable-feedback-') as folder:
 path=Path(folder)/'messages.jsonl'
 dashboard.messages=lambda:operator_feedback.messages(path)
 dashboard.append_message=lambda role,text,reply_to=None,request_id=None:operator_feedback.append_message(role,text,reply_to,request_id,path)
 server=ThreadingHTTPServer(('127.0.0.1',0),dashboard.Handler);server.password=secrets.token_urlsafe(24)
 thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base='http://127.0.0.1:'+str(server.server_port)
 def expect_error(request,code,client=urllib.request):
  try:client.urlopen(request) if client is urllib.request else client.open(request)
  except urllib.error.HTTPError as e:assert e.code==code;return
  raise AssertionError('Expected HTTP '+str(code))
 expect_error(base+'/api/feedback',401)
 client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
 client.open(urllib.request.Request(base+'/login',data=urllib.parse.urlencode({'password':server.password}).encode()))
 payload={'text':'测试建议\n<script>alert(1)</script> 你好','request_id':'same-request'}
 def post(origin=None):
  headers={'Content-Type':'application/json'}
  if origin:headers['Origin']=origin
  return urllib.request.Request(base+'/api/feedback',data=json.dumps(payload).encode(),headers=headers)
 first=json.loads(client.open(post()).read());second=json.loads(client.open(post()).read());assert first['id']==second['id']
 expect_error(post('http://unrelated.invalid'),403,client)
 reply=operator_feedback.append_message('assistant','测试回复',reply_to=first['id'],path=path)
 rows=json.loads(client.open(base+'/api/feedback').read())['messages'];assert len(rows)==2 and rows[1]['reply_to']==first['id'] and rows[0]['text']==payload['text']
 assert operator_feedback.messages(path)==rows
 server.shutdown();server.server_close();thread.join()
 print('AUTH_PERSISTENCE_RETRY_REPLY_AND_CROSS_ORIGIN_CHECKS_PASSED')
