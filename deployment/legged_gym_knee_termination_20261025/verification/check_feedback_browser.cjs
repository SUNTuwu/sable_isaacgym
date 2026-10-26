const fs=require('fs'),path=require('path'),{spawn}=require('child_process');
const root='/root/gpufree-data/projects/sable-isaacgym';
const output=path.join(root,'results/legged_gym/feedback_ui');
fs.mkdirSync(output,{recursive:true});
const profile=fs.mkdtempSync('/root/gpufree-data/tmp/sable-browser-');
const browser=spawn('/usr/local/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 let ws;
 try{
  const portFile=path.join(profile,'DevToolsActivePort');
  for(let i=0;i<100&&!fs.existsSync(portFile);i++)await sleep(100);
  const port=fs.readFileSync(portFile,'utf8').split('\n')[0];
  const tab=await(await fetch('http://127.0.0.1:'+port+'/json/new?about:blank',{method:'PUT'})).json();
  ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j});
  let seq=0;const waiting=new Map(),errors=[];
  ws.onmessage=event=>{const m=JSON.parse(event.data);if(m.id){const [resolve,reject]=waiting.get(m.id);waiting.delete(m.id);m.error?reject(Error(JSON.stringify(m.error))):resolve(m.result)}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text)};
  const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;waiting.set(id,[resolve,reject]);ws.send(JSON.stringify({id,method,params}))});
  await call('Runtime.enable');await call('Page.enable');
  const password=fs.readFileSync('/root/gpufree-data/config/sable-legged-gym/dashboard.password','utf8').trim();
  const login=await fetch('http://127.0.0.1:11111/login',{method:'POST',body:new URLSearchParams({password}),redirect:'manual'});
  if(login.status!==303)throw Error('Login failed');
  const cookie=login.headers.get('set-cookie').split(';')[0].split('=');
  await call('Network.setCookie',{name:cookie[0],value:cookie.slice(1).join('='),url:'http://127.0.0.1:11111',httpOnly:true,sameSite:'Strict'});
  await call('Emulation.setDeviceMetricsOverride',{width:1365,height:1000,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url:'http://127.0.0.1:11111/'});await sleep(2500);
  const result=await call('Runtime.evaluate',{expression:"JSON.stringify({title:document.title,form:!!document.getElementById('feedback-form'),messages:document.querySelectorAll('.feedback-item').length,monitor:document.getElementById('feedback-monitor')?.textContent,status:document.getElementById('state')?.textContent,overflow:document.documentElement.scrollWidth>innerWidth})",returnByValue:true});
  const report=JSON.parse(result.result.value);if(!report.form||!report.messages||errors.length)throw Error('Page rendering failed: '+JSON.stringify({report,errors}));
  const shot=await call('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(output,'desktop.png'),Buffer.from(shot.data,'base64'));
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await sleep(300);
  const mobile=await call('Runtime.evaluate',{expression:'document.documentElement.scrollWidth <= innerWidth',returnByValue:true});
  const shot2=await call('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(output,'mobile.png'),Buffer.from(shot2.data,'base64'));
  const receipt={passed:true,created_unix:Date.now()/1000,desktop:report,mobile_no_horizontal_overflow:mobile.result.value,errors};
  fs.writeFileSync(path.join(output,'receipt.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt));
 }finally{if(ws)ws.close();browser.kill('SIGTERM')}
})().catch(e=>{console.error(e.message);process.exitCode=1});
