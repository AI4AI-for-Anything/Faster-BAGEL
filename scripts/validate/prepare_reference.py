"""Create the measured reference from this repository's pristine upstream branch."""
import argparse,io,subprocess,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
r=Path(__file__).resolve().parents[2];out=Path(a.output).resolve();assert not out.exists()
out.mkdir(parents=True)
blob=subprocess.check_output(['git','archive','upstream'],cwd=r)
with tarfile.open(fileobj=io.BytesIO(blob)) as t:t.extractall(out,filter='data')
subprocess.run(['git','apply',str(r/'scripts/validate/reference.patch')],cwd=out,check=True)
(out/'data/dataset_info.py').write_bytes((r/'data/dataset_info.py').read_bytes())
print('Prepared reference source at',out)
