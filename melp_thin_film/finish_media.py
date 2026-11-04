"""Encode a complete frame sequence and verify the resulting H.264 video."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw
MEDIA_ENV=os.environ.copy()
# Use system FFmpeg libraries rather than the Torch launcher's library paths.
MEDIA_ENV.pop('LD_LIBRARY_PATH',None)


def run(command):
    result=subprocess.run(command,check=True,capture_output=True,env=MEDIA_ENV)
    return result.stdout


def main():
    p=argparse.ArgumentParser();p.add_argument('--simulation',type=Path,required=True)
    p.add_argument('--frames',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads((a.simulation/'config.json').read_text());n=cfg['frames'];fps=cfg['fps']
    files=sorted(a.frames.glob('*.png'));assert [int(f.stem) for f in files]==list(range(n))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_name(a.output.stem+'.encoding.mp4')
    run(['ffmpeg','-v','error','-y','-framerate',str(fps),'-start_number','0','-i',str(a.frames/'%05d.png'),
         '-c:v','libx264','-threads','4','-preset','slow','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)])
    info=json.loads(run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(temporary)]))
    stream=info['streams'][0];assert int(stream['nb_read_frames'])==n
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
    result=subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(temporary),'-f','null','-'],capture_output=True,check=True,env=MEDIA_ENV)
    assert not result.stderr,result.stderr.decode()
    temporary.replace(a.output)
    raw=run(['ffmpeg','-v','error','-i',str(a.output),'-vf','scale=160:90','-pix_fmt','rgb24','-f','rawvideo','-'])
    frames=np.frombuffer(raw,np.uint8).reshape(n,90,160,3)
    changes=np.abs(np.diff(frames.astype(np.float32),axis=0)).mean((1,2,3))
    # A time-ordered storyboard is extracted from the encoded video itself.
    indices=np.linspace(0,n-1,12).round().astype(int)
    raw=run(['ffmpeg','-v','error','-i',str(a.output),'-vf','scale=384:216','-pix_fmt','rgb24','-f','rawvideo','-'])
    frames=np.frombuffer(raw,np.uint8).reshape(n,216,384,3)
    sheet=Image.new('RGB',(384*4,240*3),(22,28,34));draw=ImageDraw.Draw(sheet)
    for j,i in enumerate(indices):
        x,y=(j%4)*384,(j//4)*240;sheet.paste(Image.fromarray(frames[i]),(x,y));draw.text((x+9,y+221),f'{i/fps:.2f} s  /  frame {i}',fill=(225,232,235))
    sheet.save(a.output.with_suffix('.storyboard.jpg'),quality=94)
    Image.fromarray(frames[indices[7]]).resize((768,432)).save(a.output.with_suffix('.jpg'),quality=94)
    poster=run(['ffmpeg','-v','error','-i',str(a.output),'-vf',f'select=eq(n\\,{indices[7]})',
               '-frames:v','1','-c:v','png','-threads','2','-f','image2pipe','-'])
    a.output.with_suffix('.png').write_bytes(poster)
    report=dict(video=str(a.output.resolve()),simulation=str(a.simulation.resolve()),frames=n,fps=fps,
        width=stream['width'],height=stream['height'],duration=float(info['format']['duration']),
        full_decode_passed=True,sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),bytes=a.output.stat().st_size,
        frame_change_mean=float(changes.mean()),frame_change_max=float(changes.max()),
        largest_frame_changes=(np.argsort(changes)[-8:]+1).tolist(),storyboard_indices=indices.tolist())
    a.output.with_suffix('.verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
