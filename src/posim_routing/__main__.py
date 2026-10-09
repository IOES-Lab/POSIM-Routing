"""Write a coarse maritime route as GeoJSON."""
import argparse,json
from pathlib import Path
from .global_route import route

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',nargs=2,type=float,required=True,metavar=('LON','LAT'))
    parser.add_argument('--end',nargs=2,type=float,required=True,metavar=('LON','LAT'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=route(args.start,args.end)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f"{result['properties']['length_m']/1000:.1f} km")

if __name__=='__main__':main()
