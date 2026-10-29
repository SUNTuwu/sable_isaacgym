"""Deterministic final-checkpoint evaluation in a fresh physical environment."""
from yaw_sine_teacher import *
def main():
    run=Path(__import__('sys').argv[1]);meta=json.loads((run/'config.json').read_text())
    cfg=student_config(Path(__file__).parent/'bootstrap',stage='S3',num_envs=16)
    cfg.env.episode_length_s=EPISODE;cfg.domain_rand.push_robots=False;cfg.migration.freeze_waist=True
    for name in vars(cfg.rewards.scales):setattr(cfg.rewards.scales,name,0.)
    cfg.rewards.scales.alive=1.
    import migration_env
    migration_env.MigrationRobot=Robot;e=create_migration_env(cfg,noise=False)
    bank=torch.load(run/'reset_poses.pt',map_location=e.device)
    e.pose_bank=bank['q'];e.pose_toes=bank['toe'];e.pose_height=bank['height']
    torch.manual_seed(7903);e.reset()
    receipt=json.loads((run/'checkpoint.json').read_text());cp=torch.load(receipt['path'],map_location=e.device)
    policy=Policy(meta['observation_size']).to(e.device);policy.load_state_dict(cp['model_state_dict']);policy.eval()
    alive=torch.ones(16,device=e.device,dtype=torch.bool);records=[]
    with torch.no_grad():
        for _ in range(math.ceil(EPISODE/e.dt)):
            action=policy.actor(e.observation());action[:,-1]=0;e.step(action)
            alive&=~e.metrics['fall'];records.append({k:v.cpu().numpy() for k,v in e.metrics.items()})
    data={k:np.stack([v[k] for v in records]) for k in records[0]}
    error=data['error'];valid=data['age']>=SETTLE
    rmse=float(np.rad2deg(np.sqrt((error[valid]**2).mean()))) if valid.any() else float('inf')
    peak=float(np.rad2deg(np.abs(error[valid]).max())) if valid.any() else float('inf')
    counts=data['count'].max(0)
    report=dict(checkpoint=receipt['path'],checkpoint_iteration=cp['iteration'],final=True,angle_rmse_deg=rmse,peak_error_deg=peak,
        survival_fraction=float(alive.float().mean()),max_liftoffs_per_foot=counts.max(0).tolist(),
        rate_rmse_deg_s=float(np.rad2deg(np.sqrt((data['rate_error'][valid]**2).mean()))) if valid.any() else None,
        max_root_drift_m=float(data['root_drift'].max()),mean_foot_anchor_error_m=float(data['foot_error'].mean()),
        accepted=bool(alive.all()) and bool((counts<=1).all()) and rmse<=5 and peak<=10,
        criteria='16 fresh deterministic envs, two full sine cycles, all survive; RMSE<=5deg, peak<=10deg, each foot <=1 liftoff over entire episode')
    atomic(run/'evaluation.json',report);np.savez_compressed(run/'latest_replay.npz',**data)
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
