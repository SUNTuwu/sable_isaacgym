"""Existing feedback endpoint/authentication, adapted to current Isaac Gym evidence."""
import argparse,hashlib,hmac,json,secrets,time,re
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs
from run_io import read_json
from operator_feedback import messages,append_message,STORE
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')
TOKEN_KEY=secrets.token_bytes(32)

def current():
    value=read_json(ROOT/'runs/current.json',{})
    run=Path(value.get('run_dir',ROOT/'runs/pending')).resolve()
    if run.parent!=ROOT/'runs':raise ValueError('Invalid run')
    return run

def lines(path):
    out=[]
    if path.exists():
        for row in path.read_text().splitlines():
            try:out.append(json.loads(row))
            except ValueError:pass
    return out

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*_):pass
    def send(self,code,body,kind='text/html; charset=utf-8'):
        body=body.encode() if isinstance(body,str) else body
        self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY');self.end_headers();self.wfile.write(body)
    def authenticated(self):
        try:
            cookies=dict(s.strip().split('=',1) for s in self.headers.get('Cookie','').split(';') if '=' in s)
            expiry,sig=cookies.get('sable_session','').split('.')
            return int(expiry)>time.time() and hmac.compare_digest(sig,hmac.new(TOKEN_KEY,expiry.encode(),hashlib.sha256).hexdigest())
        except (ValueError,TypeError):return False
    def video(self):
        info=read_json(ROOT/'results/legged_gym/side_video_preview.json',{})
        path=Path(info.get('path','/missing')).resolve()
        base=(ROOT/'results/legged_gym/videos').resolve()
        if base not in path.parents or path.suffix!='.mp4' or not path.is_file():return self.send(404,'视频尚未生成')
        size=path.stat().st_size;start=0;end=size-1;partial=False
        value=self.headers.get('Range')
        if value:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
            if not match or not any(match.groups()):return self.send(416,'Invalid range')
            a,b=match.groups()
            if a:start=int(a);end=min(int(b),end) if b else end
            else:start=max(0,size-int(b))
            if start>end or start>=size:return self.send(416,'Invalid range')
            partial=True
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type','video/mp4');self.send_header('Content-Length',str(end-start+1))
        self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        if partial:self.send_header('Content-Range','bytes %d-%d/%d'%(start,end,size))
        self.end_headers()
        try:
            with path.open('rb') as stream:
                stream.seek(start);remaining=end-start+1
                while remaining:
                    block=stream.read(min(65536,remaining))
                    if not block:break
                    self.wfile.write(block);remaining-=len(block)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_GET(self):
        path=urlsplit(self.path).path
        if not self.authenticated():
            if path!='/':return self.send(401,'登录后访问')
            return self.send(200,'<!doctype html><meta charset="utf-8"><title>训练反馈登录</title><body style="font-family:system-ui;background:#101719;color:white;padding:10vw"><h1>鼬科机器人训练监控</h1><form method="POST" action="/login"><label>原反馈网站访问密码 <input type="password" name="password" required autofocus></label> <button>登录</button></form>')
        run=current()
        if path=='/':return self.send(200,Path(__file__).with_name('dashboard.html').read_bytes())
        if path=='/video/side.mp4':return self.video()
        if path=='/api/status':
            state=read_json(run/'status.json',{'status':'preflight'})
            state['stale']=time.time()-state.get('updated_unix',0)>180
            result={'run':state,'checkpoint':read_json(run/'checkpoint.json'),'evaluation':read_json(run/'latest_evaluation.json'),
                'best':read_json(run/'best_evaluation.json'),'milestones':lines(run/'milestones.jsonl')[-30:],
                'progress':read_json(run/'progress_5min.json'),'progress_history':lines(run/'progress_5min.jsonl')[-12:],
                'backup':read_json(run/'backup_receipt.json'),
                'training_hold':read_json(ROOT/'results/cmd16_hist6/TRAINING_PAUSED.json',{}),'server_time_unix':time.time()}
        elif path=='/api/contact-forces':
            pointer=read_json(ROOT/'results/cmd16_hist6/contact_force_latest.json',{})
            report=Path(pointer.get('path','/missing')).resolve()
            base=(ROOT/'results/cmd16_hist6').resolve()
            result={'status':'unavailable'}
            if base in report.parents and report.is_file() and report.suffix=='.json':
                raw=report.read_bytes()
                if hashlib.sha256(raw).hexdigest()==pointer.get('sha256'):
                    evidence=json.loads(raw)
                    result={'status':'ready','analysis':evidence.get('contact_force_analysis'),
                        'checkpoint':evidence.get('checkpoint'),'checkpoint_sha256':evidence.get('checkpoint_sha256'),
                        'iteration':evidence.get('iteration'),'created_unix':evidence.get('created_unix'),
                        'horizon_s':evidence.get('horizon_s'),'seed':evidence.get('seed'),
                        'source_path':str(report),'report_sha256':pointer['sha256'],'training_modified':False}
        elif path=='/api/feedback':result={'messages':messages()[-500:],'storage_path':str(STORE),'monitor':read_json(ROOT/'results/legged_gym/operator_monitor_status.json',{}),'server_time_unix':time.time()}
        elif path=='/api/learning-curve':result=lines(run/'metrics.jsonl')[-5000:]
        elif path=='/api/video':result=read_json(ROOT/'results/legged_gym/side_video_preview.json',{'status':'preparing'})
        else:return self.send(404,'Not found')
        return self.send(200,json.dumps(result,ensure_ascii=False,allow_nan=False),'application/json; charset=utf-8')
    def do_POST(self):
        if self.path=='/api/feedback':
            if not self.authenticated():return self.send(401,'登录后留言')
            origin=self.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=self.headers.get('Host'):return self.send(403,'不接受跨站留言')
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send(415,'请使用JSON提交')
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=24000:return self.send(400,'留言过长')
                payload=json.loads(self.rfile.read(size).decode('utf8'))
                request_id=payload.get('request_id')
                if request_id is not None and (not isinstance(request_id,str) or len(request_id)>80):raise ValueError('无效请求标识')
                row=append_message('user',payload.get('text'),request_id=request_id)
            except (ValueError,TypeError,AttributeError) as error:return self.send(400,str(error))
            return self.send(201,json.dumps(row,ensure_ascii=False),'application/json; charset=utf-8')
        if self.path!='/login':return self.send(404,'Not found')
        size=int(self.headers.get('Content-Length','0'))
        if not 0<size<=1024:return self.send(400,'Invalid request')
        value=parse_qs(self.rfile.read(size).decode('utf8',errors='replace')).get('password',[''])[0]
        if not hmac.compare_digest(value,self.server.password):return self.send(403,'密码不正确')
        expiry=str(int(time.time()+12*3600));token=expiry+'.'+hmac.new(TOKEN_KEY,expiry.encode(),hashlib.sha256).hexdigest()
        self.send_response(303);self.send_header('Location','/');self.send_header('Set-Cookie','sable_session='+token+'; HttpOnly; SameSite=Strict; Path=/; Max-Age=43200');self.send_header('Content-Length','0');self.end_headers()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=11111);p.add_argument('--password-file',type=Path,default=Path('/root/gpufree-data/config/sable-legged-gym/dashboard.password'));a=p.parse_args()
    assert str(a.password_file.resolve()).startswith('/root/gpufree-data/config/')
    server=ThreadingHTTPServer(('0.0.0.0',a.port),Handler);server.password=a.password_file.read_text().strip()
    assert server.password;print('Feedback ready on port',a.port,flush=True);server.serve_forever()
