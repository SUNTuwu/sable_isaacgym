"""Bound only the explicitly identified teacher-bank process by wall deadline."""
import argparse,datetime,json,os,signal,subprocess,time
from pathlib import Path

def atomic(path,data):
    temp=path.with_name(path.name+'.watchdog.tmp')
    temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def identity(pid):
    base=Path('/proc')/str(pid)
    try:
        cmd=(base/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        ticks=(base/'stat').read_text().rsplit(') ',1)[1].split()[19]
        return cmd,ticks
    except (FileNotFoundError,ProcessLookupError):return None

def resources(bank):
    result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
    values=result.stdout.strip().split(',')
    if len(values)!=4:return
    row=dict(timestamp=time.time(),gpu_utilization_percent=float(values[0]),gpu_memory_used_mib=float(values[1]),
             gpu_memory_total_mib=float(values[2]),gpu_power_w=float(values[3]))
    with (bank/'resource_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    atomic(bank/'resource_latest.json',row)

def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True,type=Path)
    p.add_argument('--pid',required=True,type=int);p.add_argument('--deadline',required=True)
    p.add_argument('--final-grace-s',type=float,default=180.)
    a=p.parse_args();bank=a.bank.resolve();assert bank.is_dir() and bank.name.startswith('teacher_bank_')
    deadline=datetime.datetime.fromisoformat(a.deadline.replace('Z','+00:00')).timestamp()
    original=identity(a.pid)
    assert original and str(bank) in original[0],'PID command does not identify this bank'
    def alive():return identity(a.pid)==original
    state=dict(pid=os.getpid(),target_pid=a.pid,target_start_ticks=original[1],bank=str(bank),deadline=a.deadline,
               started_unix=time.time(),scope='Only signal the exact original PID/starttime with bank path in command')
    path=bank/'deadline_watchdog.json';atomic(path,dict(state,status='monitoring'))
    next_resource=0.
    while alive() and time.time()<deadline:
        if time.time()>=next_resource:
            try:resources(bank)
            except Exception as e:state['resource_error']=str(e)
            next_resource=time.time()+30
        time.sleep(min(5,max(.1,deadline-time.time())))
    if not alive():
        atomic(path,dict(state,status='target_exited',finished_unix=time.time()));return
    (bank/'STOP').write_text('Deadline reached; save and finish this teacher bank.\n')
    atomic(path,dict(state,status='stop_requested',requested_unix=time.time()))
    end=time.time()+a.final_grace_s
    while alive() and time.time()<end:time.sleep(2)
    if alive():
        os.kill(a.pid,signal.SIGTERM);atomic(path,dict(state,status='term_sent',signal_unix=time.time()))
        end=time.time()+30
        while alive() and time.time()<end:time.sleep(2)
    if alive():
        os.kill(a.pid,signal.SIGKILL);atomic(path,dict(state,status='kill_sent',signal_unix=time.time()))
    else:atomic(path,dict(state,status='deadline_stopped',finished_unix=time.time()))

if __name__=='__main__':main()
