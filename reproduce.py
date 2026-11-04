#!/usr/bin/env python3
"""รันประเมิน MFCC ให้ครบ 5 ชุด x 4 สเกล แล้วเทียบกับผลที่เปเปอร์รายงานและผลใน log ของรอบ Journal เดิม

ใช้ test.py ตัวจริงเป็นตัวรัน ไม่เขียน metric ใหม่ซ้ำ · บันทึกภาพผลทุกภาพไว้บนเครื่อง GPU
ผลสรุป: <out>/summary.csv, <out>/SUMMARY.md, <out>/summary.json
"""
import argparse, csv, json, os, subprocess, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PKG = Path(__file__).resolve().parent   # รากของ repo นี้ ไม่ผูกกับเครื่องใด
TZ = timezone(timedelta(hours=7))
DATASETS = ["Set5", "Set14", "B100", "Urban100", "Manga109"]
SCALES = [2, 3, 4, 8]

# PSNR ที่วัดได้ในรอบ Journal เดิม — อ่านจาก 2023-ResultImage/rc20r2rg1nup/log.txt
JOURNAL_LOG_PSNR = {
    2: {"Set5": 38.162, "Set14": 33.857, "B100": 32.279, "Urban100": 32.659, "Manga109": 39.113},
    3: {"Set5": 34.672, "Set14": 30.513, "B100": 29.213, "Urban100": 28.640, "Manga109": 34.153},
    4: {"Set5": 32.418, "Set14": 28.734, "B100": 27.668, "Urban100": 26.475, "Manga109": 30.981},
    8: {"Set5": 27.072, "Set14": 25.005, "B100": 24.830, "Urban100": 22.538, "Manga109": 24.630},
}
# ค่าที่พิมพ์ในตาราง 2 ของบทความ (อ่านจาก PDF หน้า 8)
PAPER = {
    2: {"Set5": (38.16, .9611), "Set14": (33.85, .9195), "B100": (32.28, .9010),
        "Urban100": (32.65, .9331), "Manga109": (39.11, .9780)},
    3: {"Set5": (34.67, .9294), "Set14": (30.51, .8456), "B100": (29.22, .8080),
        "Urban100": (28.64, .8615), "Manga109": (34.15, .9478)},
    4: {"Set5": (32.42, .8973), "Set14": (28.73, .7849), "B100": (27.67, .7399),
        "Urban100": (26.48, .7977), "Manga109": (30.98, .9131)},
    8: {"Set5": (27.07, .7762), "Set14": (25.01, .6412), "B100": (24.84, .5980),
        "Urban100": (22.54, .6196), "Manga109": (24.63, .7791)},
}


def find_dirs(root, dataset, scale):
    base = Path(root) / dataset
    hr = base / "HR"
    lr = None
    for name in (f"X{scale}", f"x{scale}"):
        candidate = base / "LR_bicubic" / name
        if candidate.is_dir():
            lr = candidate
            break
    return (lr, hr) if lr and hr.is_dir() else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, help="โฟลเดอร์ที่มี <ชุด>/HR และ <ชุด>/LR_bicubic/X<scale>")
    ap.add_argument("--out", required=True, help="โฟลเดอร์ผลลัพธ์ใหม่")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--scales", nargs="*", type=int, default=SCALES)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []

    for scale in a.scales:
        for dataset in a.datasets:
            lr, hr = find_dirs(a.benchmark, dataset, scale)
            if lr is None:
                skipped.append({"dataset": dataset, "scale": scale, "reason": "ไม่พบ HR หรือ LR_bicubic"})
                print(f"ข้าม {dataset} x{scale}: ไม่พบโฟลเดอร์", flush=True)
                continue
            run_dir = out / "MFCC" / f"{dataset}_x{scale}"
            if run_dir.exists():
                skipped.append({"dataset": dataset, "scale": scale, "reason": "มีผลรอบนี้อยู่แล้ว ไม่เขียนทับ"})
                print(f"ข้าม {dataset} x{scale}: มีอยู่แล้ว", flush=True)
                continue
            cmd = [a.python, "test.py", "--scale", str(scale), "--device", a.device,
                   "--lr-dir", str(lr), "--hr-dir", str(hr), "--ssim",
                   "--run-dir", str(run_dir)]
            if a.save_images:
                cmd.append("--save-images")
            print(f"▶ {dataset} x{scale}", flush=True)
            proc = subprocess.run(cmd, cwd=PKG, capture_output=True, text=True)
            if proc.returncode != 0:
                skipped.append({"dataset": dataset, "scale": scale,
                                "reason": f"test.py ออกด้วยรหัส {proc.returncode}",
                                "stderr": proc.stderr[-2000:]})
                print(f"  ล้มเหลว: {proc.stderr[-400:]}", flush=True)
                continue
            report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            psnr, ssim = report["mean_y_psnr"], report.get("mean_y_ssim")
            ref = JOURNAL_LOG_PSNR.get(scale, {}).get(dataset)
            pap = PAPER.get(scale, {}).get(dataset)
            row = {
                "scale": scale, "dataset": dataset, "images": report["count"],
                "psnr": round(psnr, 4), "ssim": round(ssim, 6) if ssim is not None else None,
                "journal_log_psnr": ref,
                "psnr_minus_journal_log": round(psnr - ref, 4) if ref is not None else None,
                "paper_psnr": pap[0] if pap else None,
                "paper_ssim": pap[1] if pap else None,
                "psnr_minus_paper": round(psnr - pap[0], 4) if pap else None,
                "ssim_minus_paper": round(ssim - pap[1], 6) if (pap and ssim is not None) else None,
                "peak_allocated_mib": round(report.get("peak_allocated_mib", 0), 1),
                "seconds_total": round(sum(i["inference_seconds"] for i in report["images"]), 3),
                "run_dir": str(run_dir),
            }
            rows.append(row)
            print("  PSNR %.4f (log เดิม %s · บทความ %s) · SSIM %s"
                  % (psnr, ref, pap[0] if pap else "-", f"{ssim:.4f}" if ssim else "-"), flush=True)

    summary = {"generated_at": datetime.now(TZ).isoformat(), "benchmark_root": a.benchmark,
               "device": a.device, "rows": rows, "skipped": skipped}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        md = ["# ผลวัดซ้ำ MFCC เทียบบทความ IEEE Access 2023",
              "", f"วัดเมื่อ {summary['generated_at']} · อุปกรณ์ {a.device} · ชุดภาพ `{a.benchmark}`",
              "", "`log เดิม` คือค่าที่พิมพ์ใน `2023-ResultImage/rc20r2rg1nup/log.txt` ตอนรันปี 2564",
              "`บทความ` คือค่าที่พิมพ์ในตาราง 2 ของบทความ", "",
              "| สเกล | ชุด | ภาพ | PSNR วัดใหม่ | log เดิม | ต่าง | บทความ | ต่าง | SSIM วัดใหม่ | บทความ | ต่าง |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            md.append("| ×{scale} | {dataset} | {images} | {psnr:.4f} | {jl} | {dj} | {pp} | {dp} | {ss} | {ps} | {ds} |".format(
                scale=r["scale"], dataset=r["dataset"], images=r["images"], psnr=r["psnr"],
                jl=f'{r["journal_log_psnr"]:.3f}' if r["journal_log_psnr"] is not None else "-",
                dj=f'{r["psnr_minus_journal_log"]:+.4f}' if r["psnr_minus_journal_log"] is not None else "-",
                pp=f'{r["paper_psnr"]:.2f}' if r["paper_psnr"] is not None else "-",
                dp=f'{r["psnr_minus_paper"]:+.4f}' if r["psnr_minus_paper"] is not None else "-",
                ss=f'{r["ssim"]:.4f}' if r["ssim"] is not None else "-",
                ps=f'{r["paper_ssim"]:.4f}' if r["paper_ssim"] is not None else "-",
                ds=f'{r["ssim_minus_paper"]:+.4f}' if r["ssim_minus_paper"] is not None else "-"))
        if skipped:
            md += ["", "## ที่ข้ามไป", ""]
            md += [f"- {s['dataset']} ×{s['scale']}: {s['reason']}" for s in skipped]
        (out / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nสรุป: {out/'SUMMARY.md'}")
    print(f"สำเร็จ {len(rows)} ช่อง · ข้าม {len(skipped)} ช่อง")


if __name__ == "__main__":
    main()
