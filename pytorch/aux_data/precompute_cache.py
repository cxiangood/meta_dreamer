#!/usr/bin/env python3
"""
Pre-compute lane/junction polygon cache for all locations.
Save as .pkl files, no sumolib needed on HPC for aux label generation.

Run locally:
    python aux_data/precompute_cache.py --map-dir mirro_data_map --output aux_data/cache
"""
import os, sys, pickle, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from shapely.geometry import Polygon
except ImportError:
    print("ERROR: pip install shapely")
    sys.exit(1)

from add_aux_labels import load_lane_polygons_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-dir', required=True)
    parser.add_argument('--output', default='aux_data/cache')
    parser.add_argument('--locs', default='0,1,2,3,4,5,6')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for loc_id in [int(x) for x in args.locs.split(',')]:
        try:
            polygons = load_lane_polygons_meta(loc_id, args.map_dir)
            out_path = os.path.join(args.output, f'polygons_loc{loc_id}.pkl')
            # Shapely polygons need to be serializable
            poly_data = []
            for poly, is_ramp in polygons:
                poly_data.append({'coords': list(poly.exterior.coords), 'is_ramp': is_ramp})
            with open(out_path, 'wb') as f:
                pickle.dump(poly_data, f)
            print(f"  loc {loc_id}: {len(poly_data)} polygons saved to {out_path}")
        except Exception as e:
            print(f"  loc {loc_id}: ERROR - {e}")

    print("Done.")


if __name__ == "__main__":
    main()
