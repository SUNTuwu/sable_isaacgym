"""Shared authenticated operator conversation, durable across training runs."""
import argparse
import fcntl
import json
from pathlib import Path
import time
import uuid

ROOT = Path('/root/gpufree-data/projects/sable-isaacgym')
STORE = ROOT / 'results/legged_gym/operator_feedback.jsonl'


def messages(path=STORE):
    if not path.exists():
        return []
    with path.open('r', encoding='utf8') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        return [json.loads(line) for line in stream if line.strip()]


def append_message(role, text, reply_to=None, request_id=None, path=STORE):
    if role not in ['user', 'assistant'] or not isinstance(text, str) or not 1 <= len(text.strip()) <= 4000:
        raise ValueError('留言需要 1–4000 个字符')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+', encoding='utf8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        prior = [json.loads(line) for line in stream if line.strip()]
        if request_id:
            for row in prior:
                if row.get('request_id') == request_id and row['role'] == role:
                    return row
        if reply_to and not any(row['id'] == reply_to and row['role'] == 'user' for row in prior):
            raise ValueError('找不到被回复的留言')
        row = {'id': uuid.uuid4().hex, 'created_unix': time.time(), 'role': role,
               'text': text.strip(), 'reply_to': reply_to, 'request_id': request_id}
        stream.seek(0, 2)
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
        return row


def pending():
    rows = messages()
    answered = {r.get('reply_to') for r in rows if r['role'] == 'assistant'}
    waiting=[r for r in rows if r['role'] == 'user' and r['id'] not in answered]
    from run_io import atomic_json
    atomic_json(ROOT/'results/legged_gym/operator_monitor_status.json',{'last_checked_unix':time.time(),'pending_count':len(waiting),'active_until':'2026-10-25T19:00:00+08:00'})
    return waiting


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pending', action='store_true')
    parser.add_argument('--reply-to')
    parser.add_argument('--text')
    args = parser.parse_args()
    if args.text:
        print(json.dumps(append_message('assistant', args.text, args.reply_to), ensure_ascii=False))
    else:
        print(json.dumps(pending() if args.pending else messages()[-100:], ensure_ascii=False, indent=2))
