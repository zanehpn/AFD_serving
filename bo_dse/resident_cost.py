"""Continuous reserved-GPU cost accounting during serialized resident searches."""
from pathlib import Path
import time
import official_worker as w
from resident_runner import window_cost

class Session:
    def __init__(self,config,campaigns):
        self.config=config;self.campaigns=campaigns;self.enabled=bool(config.get('resident_session_directory'));self.meter=None;self.last=0
    def cpu_seconds(self):
        return sum(w.read(Path(a['directory'])/'state.json').get('optimizer_wall_seconds',0.) for a in self.campaigns)
    def __enter__(self):
        if self.enabled:
            self.directory=Path(self.config['resident_session_directory']);self.directory.mkdir(parents=True,exist_ok=True)
            import official
            if list(self.directory.glob('service-*.json')):official.worker(self.config,'cleanup',self.directory)
            self.meter=w.Monitor(self.config['gpus']);self.meter.start();self.last=0;self.cpu=self.cpu_seconds()
        return self
    def charge(self,result,trial):
        if not self.enabled:return result
        self.meter.sample();end=len(self.meter.rows)-1;rows=list(self.meter.rows[self.last:end+1]);self.last=end
        path=Path(trial)/'resident-cost-window.jsonl';path.write_text(''.join(__import__('json').dumps(r)+'\n' for r in rows))
        cost=window_cost(rows,len(self.config['gpus']));full_wall=cost['wall_seconds'];cpu=self.cpu_seconds();cpu_delta=max(0,cpu-self.cpu);self.cpu=cpu
        # campaign.costs separately adds optimizer wall time; keep GPUhours and
        # energy for the full reserved interval, but avoid double-counting wall.
        cost['wall_seconds']=max(0,full_wall-cpu_delta)
        result.update(worker_cost=result['cost'],cost=cost,cost_protocol='resident_reserved_gpu_intervals_v2',resident_interval_wall_seconds=full_wall,resident_optimizer_wall_seconds=cpu_delta,
                      cost_energy_complete=bool(len(rows)>1 and not self.meter.errors and all(b['timestamp_ns']-a['timestamp_ns']<=1e9 for a,b in zip(rows,rows[1:]))))
        result['artifacts'].append(w.artifact(path));w.write(Path(trial)/'result.json',result);return result
    def __exit__(self,exc_type,exc,tb):
        if not self.enabled:return False
        import official
        cleanup_error=None
        try:official.worker(self.config,'cleanup',self.directory)
        except BaseException as error:cleanup_error=repr(error)
        finally:
            if self.meter:
                self.meter.stop(self.directory)
                w.write(self.directory/'session-cost.json',{'total':window_cost(self.meter.rows,len(self.config['gpus'])),
                    'unassigned_finalization_and_shutdown':window_cost(self.meter.rows[self.last:],len(self.config['gpus'])),
                    'sample_errors':self.meter.errors,'cleanup_error':cleanup_error,
                    'note':'Per-trial reserved costs include idle and switching. Finalization/shutdown overhead retained separately.'})
        if cleanup_error:raise RuntimeError('Resident session cleanup failed: '+cleanup_error)
        return False
