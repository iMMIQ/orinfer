"""Host-only continuous read-only machine sampler (one sample/sec)."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def read(p):
    try:return Path(p).read_text().strip()
    except (OSError,TypeError,ValueError)as e:return {'unavailable':str(e)}

def sample(phase):
    temp={}
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        temp[str(read(zone/'type'))]=read(zone/'temp')
    rails=[]
    for hwmon in Path('/sys/bus/i2c/drivers/ina3221').glob('*/hwmon/hwmon*'):
        for channel in range(1,4):
            label=read(hwmon/f'in{channel}_label');v=read(hwmon/f'in{channel}_input');a=read(hwmon/f'curr{channel}_input')
            rails.append(dict(label=label,voltage_mV=v,current_mA=a,power_mW=int(v)*int(a)/1000 if isinstance(v,str)and v.isdigit()and isinstance(a,str)and a.isdigit()else None))
    return dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=read(phase),gpu_freq_hz={k:read('/sys/class/devfreq/17000000.gpu/'+k)for k in('cur_freq','min_freq','max_freq')},thermal_millicelsius=temp,power_rails=rails,meminfo=read('/proc/meminfo'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--phase',type=Path,required=True);ap.add_argument('--stop',type=Path,required=True);args=ap.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    start=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),power_mode=subprocess.run(['/usr/sbin/nvpmodel','-q'],capture_output=True,text=True).stdout,background_containers=subprocess.run(['docker','ps','--format','{{.Names}} {{.Image}}'],capture_output=True,text=True).stdout.splitlines())
    args.output.with_suffix('.start.json').write_text(json.dumps(start,indent=2))
    with args.output.open('a',buffering=1)as f:
        while not args.stop.exists():
            f.write(json.dumps(sample(args.phase))+'\n');time.sleep(1)
if __name__=='__main__':main()
