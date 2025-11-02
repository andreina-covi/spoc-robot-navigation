import math
import argparse
import json
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Online evaluation")
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--jsonl_out", default="train.jsonl", type=str)
    # parser.add_argument("--max_eps_len", default=-1, type=int)
    # parser.add_argument("--det_type", default="gt", help="gt or detic", choices=["gt", "detic"])
    # parser.add_argument("--total_num_videos", type=int, default=8200)

    args = parser.parse_args()
    return args

# === 2. Helper functions ===
def cardinal_from_yaw(yaw):
    dirs = ["north","north-east","east","south-east","south","south-west","west","north-west"]
    idx = int(((yaw % 360) + 22.5) // 45) % 8
    return dirs[idx]

def split_name(name):
    return name.split("|")

def bearing_from_agent(ax, az, ox, oz, ayaw_deg):
    vx, vz = ox - ax, oz - az
    obj_ang = math.degrees(math.atan2(vx, vz))
    rel = (obj_ang - ayaw_deg + 540) % 360 - 180
    if -22.5 <= rel <= 22.5: return "front"
    if 22.5 < rel <= 67.5: return "front-right"
    if 67.5 < rel <= 112.5: return "right"
    if 112.5 < rel <= 157.5: return "back-right"
    if rel > 157.5 or rel <= -157.5: return "back"
    if -157.5 < rel <= -112.5: return "back-left"
    if -112.5 < rel <= -67.5: return "left"
    return "front-left"

def get_records(csv_path):
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        action = row.get("ag-action","unknown")
        ax, ay, az = row["ag-pos-x"], row["ag-pos-y"], row["ag-pos-z"]
        yaw = row["ag-rot-y"]
        direction = cardinal_from_yaw(yaw)
        
        obj = row.get("obj-name", "unknown")
        ox, oy, oz = row["obj-pos-x"], row["obj-pos-y"], row["obj-pos-z"]
        dist = row["obj-distance"]
        bearing = bearing_from_agent(ax, az, ox, oz, yaw)
        
        user_text = (
            f"ag-action={action}; ag-pos=({ax:.2f},{ay:.2f},{az:.2f}); "
            f"ag-rot-y={yaw:.1f}; obj={obj} @ ({ox:.2f},{oy:.2f},{oz:.2f}); dist={dist:.2f}"
        )
        
        assistant_text = (
            f"The agent performs '{action}' at position ({ax:.2f}, {ay:.2f}, {az:.2f}), "
            f"facing {direction} ({yaw:.0f}°). It sees a {obj} approximately {dist:.2f} m away "
            f"to the {bearing}."
        )

        example = {
            "messages": [
                {"role": "system", "content": "You are a technical narrator that describes agent navigation telemetry."},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text}
            ]
        }
        records.append(example)
    return records

def write_json(json_path, records):
    with open(json_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(args):
    csv_path = args.csv_path
    jsonl_out = args.jsonl_out
    records = get_records(csv_path)
    write_json(jsonl_out, records)

# print(f"✅ Saved {len(records)} examples to {jsonl_out}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
