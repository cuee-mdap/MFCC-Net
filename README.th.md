# MFCC Journal — PyTorch train / test (ภาษาไทย)

English: [README.md](README.md)

โค้ดแยกสำหรับ **Multi-FusNet of Cross Channel Network for Image Super-Resolution**, IEEE Access 2023, DOI [10.1109/ACCESS.2023.3282571](https://doi.org/10.1109/ACCESS.2023.3282571)

พอร์ตจาก `2021-08-29 dgx/srn20210829` โดยคงโครงสร้าง โมดูล การแชร์น้ำหนัก และรูปแบบ state dict ของโมเดล Journal ไว้ ต้นฉบับและผลทดลองเก่าไม่ได้ถูกแก้ไข โฟลเดอร์นี้ย้ายเครื่องได้โดยไม่ต้องมีต้นฉบับอยู่ข้าง ๆ

ทดสอบจริงบน Windows, Python 3.13.12, **PyTorch 2.12.1+cpu**, NumPy 2.4.3, Pillow 12.1.1 ไม่ได้อ้างว่าทดสอบ PyTorch รุ่นใหม่กว่านี้หรือ CUDA แล้ว ผลละเอียดอยู่ใน [VALIDATION.md](VALIDATION.md) ซึ่งตอนนี้วัดบน RTX 3060 แล้ว

## ผลวัดซ้ำ — ตรงกับตาราง 2 ของบทความทุกช่อง

วัดบน NVIDIA RTX 3060 12 GB (PyTorch 2.2.2+cu121) ด้วยไฟล์ LR/HR ชุดเดียวกับที่ใช้ตอนทำบทความ
ครบทั้ง 20 ช่อง (Set5, Set14, B100, Urban100, Manga109 × ×2/×3/×4/×8)

| เทียบกับ | PSNR ต่างสูงสุด | เฉลี่ย | SSIM ต่างสูงสุด |
|---|---:|---:|---:|
| ตาราง 2 ของบทความ | **0.0091 dB** | 0.0033 dB | **0.0001** |
| log ของรอบที่ทำบทความ (2564) | 0.0072 dB | 0.0015 dB | — |

ปัดทศนิยมสองตำแหน่งแล้วตรงกับที่พิมพ์ในบทความ 16/20 ช่อง อีก 4 ช่องต่างไม่ถึง 0.01 dB (×2 Set14, ×2 Urban100, ×4 Urban100, ×8 Set14) · ตารางเต็มและเงื่อนไขการวัดอยู่ใน
[VALIDATION.md](VALIDATION.md) · ค่าดิบรายช่องอยู่ใน [results/](results/)

สร้างผลซ้ำเองด้วย

```bash
python reproduce.py --benchmark /path/to/benchmark --out /path/to/output --device cuda
```

`--benchmark` ต้องเป็นโฟลเดอร์ที่มี `<ชุด>/HR` และ `<ชุด>/LR_bicubic/X<สเกล>` (รับ `x<สเกล>` ตัวเล็กด้วย)

## ไฟล์หลัก

- [train.py](train.py): ฝึกใหม่ / fine-tune / resume, gradient accumulation และ CUDA AMP แบบเลือกเปิด
- [test.py](test.py): ทดสอบ LR/HR หรือขยายภาพจาก LR อย่างเดียว
- [profile_gpu.py](profile_gpu.py): วัดเวลาและ peak VRAM ของ training step ก่อนฝึกจริง
- [verify.py](verify.py): ตรวจน้ำหนักทั้งสี่สเกล การแชร์น้ำหนัก gradient และ metric
- [mfcc/model.py](mfcc/model.py): โมเดล Journal ขนาดคงที่ 64 channels, 20 RCAB, reduction 16, RGB range 1
- [weights/manifest.json](weights/manifest.json): ที่มา SHA-256 และ config เดิมของน้ำหนักทั้งสี่สเกล

`weights/mfcc_x2.pt`, `mfcc_x3.pt`, `mfcc_x4.pt`, `mfcc_x8.pt` เป็นสำเนาน้ำหนักเดิมที่ตรวจ hash แล้ว รวมประมาณ 35 MB จึงย้ายทั้งโฟลเดอร์ไปทดสอบได้ น้ำหนักทั้งสี่ไฟล์อยู่ใน Git พร้อม manifest

## 1. เตรียม Python

ตัวอย่างต่อไปนี้ให้รันจากโฟลเดอร์นี้ สร้าง environment นอก Drive เพื่อไม่ให้ไฟล์ติดตั้งจำนวนมากถูก sync:

```powershell
$venvPath = Join-Path $env:LOCALAPPDATA 'mfcc-env'
python -m venv $venvPath
$pyMFCC = Join-Path $venvPath 'Scripts\python.exe'
& $pyMFCC -m pip install --upgrade pip
```

บนเครื่อง CPU:

```powershell
& $pyMFCC -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& $pyMFCC -m pip install -r requirements.txt
& $pyMFCC verify.py
```

บนเครื่อง RTX 3060 ให้ติดตั้ง **CUDA build** ของ torch ใน environment ของเครื่องนั้น โดยใช้คำสั่งจาก [PyTorch installation selector](https://pytorch.org/get-started/locally/) ที่ตรงกับ Windows/Linux และ driver ที่ติดตั้ง ไม่ต้องติดตั้ง torchvision หรือ torchaudio สำหรับโค้ดนี้ จากนั้น:

```powershell
& $pyMFCC -m pip install -r requirements.txt
& $pyMFCC -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
& $pyMFCC verify.py
```

ต้องได้ `True` และชื่อ RTX 3060 ก่อนใช้ `--device cuda` คำสั่งที่ขอ CUDA จะหยุดพร้อมข้อผิดพลาดหากใช้ไม่ได้ ไม่เปลี่ยนไป CPU เงียบ ๆ สำหรับ Linux ใช้ environment ภายนอกโฟลเดอร์เช่นกัน แล้วแทน `& $pyMFCC` ด้วย Python ใน environment นั้น

## 2. ทดสอบน้ำหนัก Journal บนเครื่องปัจจุบัน

ใช้ชุด benchmark แบบ RCAN/SAN (`<ชุด>/HR`, `<ชุด>/LR_bicubic/X<สเกล>`) ของ Set5, Set14, B100, Urban100, Manga109 ไม่ต้องคัดลอก dataset มาไว้ในโฟลเดอร์โค้ด

```powershell
$benchmark = 'D:/SR/benchmark'
foreach ($scale in @(2,3,4,8)) {
    & $pyMFCC test.py --scale $scale --device cpu --lr-dir "$benchmark/Set5/LR_bicubic/X$scale" --hr-dir "$benchmark/Set5/HR" --ssim --save-images
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
}
```

บนเครื่องใหม่เปลี่ยน `$benchmark` เป็นตำแหน่ง dataset จริง และเปลี่ยน `--device cpu` เป็น `--device cuda` ใช้ชื่อชุดอื่นแทน Set5 เพื่อวัดผลเพิ่มเติม

ทดสอบภาพ LR ของตนเองโดยไม่มี HR:

```powershell
& $pyMFCC test.py --scale 4 --lr-dir 'D:/SR/my-lr-images' --device cuda --save-images
```

ผล `metrics.json` และภาพจะถูกเขียนใน `./runs/` ภายใต้ชื่อรอบที่ไม่ซ้ำ ระบุตำแหน่งเองได้ด้วย `--run-dir` ซึ่งต้องเป็นโฟลเดอร์ใหม่ โปรแกรมไม่เขียนทับรอบเดิม ไม่สร้าง cache ใน dataset และไม่บันทึกภาพหากไม่ได้ใช้ `--save-images`

### วิธีวัดและแบ่งภาพ

- ค่าเริ่มต้น `--mode journal` ใช้ **chop สำหรับ ×2/×3/×4 และ direct สำหรับ ×8** ตาม config ที่บันทึกภาพผล Journal จริง ไม่ใช่เพียง config ตอนฝึก
- PSNR: clamp/round เป็นระดับภาพ 8-bit, วัด Y ด้วยสัมประสิทธิ์และขั้นตอนเดิม, ตัดขอบเท่ากับ scale, เฉลี่ย dB ของแต่ละภาพ
- `--ssim`: Y-SSIM แบบ Gaussian 11×11, sigma 1.5, population covariance, valid window และตัดขอบเท่ากับ scale มีคำอธิบายใน [mfcc/metrics.py](mfcc/metrics.py) ตัวโค้ดเก่าที่ตรวจพบคำนวณ PSNR เท่านั้น จึงไม่อ้างว่า SSIM implementation นี้เป็นไฟล์เดิม
- ภาพใหญ่บน 3060 เริ่ม `--mode chop --chop-min-size 20000` เพื่อลดขนาดชิ้นที่ส่งเข้าโมเดล ค่าเดิมคือ 160000 **การเปลี่ยนขนาด chop อาจเปลี่ยนค่า PSNR** เพราะ channel attention ใช้ค่าเฉลี่ยของพื้นที่ภาพ ไม่ควรปะปนผลกับโหมด Journal
- โหมด chop ยังคงประกอบภาพ output ทั้งภาพไว้ในหน่วยความจำ จึงไม่รับประกันว่าจะรับภาพทุกขนาดได้ ลดขนาด input หรือวัด VRAM เมื่อใช้ภาพใหญ่มาก
- รับ LR/HR ชื่อ `0001x4.png` / `0001.png` หรือชื่อเหมือนกันทั้งคู่ ตรวจขนาดภาพและคู่ที่หายก่อนใช้งาน ตัดเฉพาะเศษขอบล่าง/ขวาที่ไม่หารด้วย scale ลงตัว ไม่สร้าง LR ใหม่ด้วย resize คนละ implementation

## 3. ตรวจ RTX 3060 12GB ก่อนฝึก

```powershell
& $pyMFCC profile_gpu.py --device cuda --scale 4 --batch-size 4 --lr-patch 48 --steps 10
& $pyMFCC profile_gpu.py --device cuda --scale 8 --batch-size 4 --lr-patch 48 --steps 10
```

อ่าน `peak_allocated_mib`, `peak_reserved_mib` และเวลาต่อ step ใน `profile.json` หากหน่วยความจำไม่พอให้ลด batch เป็น 2 หรือ 1; หากเหลือมากค่อยลอง 8/16 ตัว profiler ใช้น้ำหนักสำเนาในหน่วยความจำและไม่บันทึกน้ำหนักที่อัปเดตแล้ว ทดลอง `--amp` ได้ภายหลัง FP32 baseline ตาม [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)

## 4. ฝึกใหม่ / fine-tune

ใช้ภาพ DIV2K 800 คู่ที่เตรียม bicubic LR ไว้แล้ว ตัวอย่างพาธด้านล่างต้องเปลี่ยนตามเครื่องจริง ข้อมูลฝึก DIV2K ยังไม่ได้พบ/ทดสอบในรอบนี้:

```powershell
& $pyMFCC train.py --scale 4 --device cuda --lr-dir 'D:/SR/DIV2K/DIV2K_train_LR_bicubic/X4' --hr-dir 'D:/SR/DIV2K/DIV2K_train_HR' --val-lr-dir 'D:/SR/benchmark/Set5/LR_bicubic/X4' --val-hr-dir 'D:/SR/benchmark/Set5/HR' --init weights/mfcc_x4.pt --lr 1e-5 --batch-size 4 --accumulate 4 --lr-patch 48 --epochs 100 --steps-per-epoch 1000 --workers 0
```

นี่คือตัวอย่าง **fine-tune** ด้วย optimizer ใหม่ หากฝึกจาก random initialization ให้เอา `--init ...` ออก และเริ่ม `--lr 1e-4 --epochs 600` ค่าเหล่านี้เป็นจุดเริ่มตามบทความ ไม่ใช่การรับประกันว่าจะได้ผลเดิม เพราะน้ำหนัก Journal ผ่านการฝึกต่อหลายรอบตาม config ใน manifest

- `--lr-patch 48` หมายถึง LR 48×48 ทุกสเกล; HR เป็น 96/144/192/384 สำหรับ ×2/×3/×4/×8 ตามลำดับ ต่างจาก option เก่าที่ `patch_size` หมายถึง HR
- batch 4 × accumulate 4 = effective batch 16 ลดหน่วยความจำของ activation โดยสะสม gradient ก่อน Adam update
- Adam β=(0.9,0.999), ε=1e-8, L1 loss, flip/transpose augmentation ตามโค้ดเดิม; ไม่มีการหมุน 120°
- หนึ่ง epoch มี `steps-per-epoch` optimizer updates ดังนั้นค่าเริ่มต้น 1000 × decay-every 200 = ลด LR ครึ่งหนึ่งทุก 200,000 updates
- ใช้ index ภาพวนซ้ำแล้ว shuffle ต่อ epoch เหมือนแนวทางเดิม; 800 ภาพ/effective batch 16/1000 updates ทำให้ภาพละ 20 samples ต่อ epoch หากเปลี่ยนจำนวนข้อมูล/steps จำนวน samples ต่อภาพอาจไม่เท่ากัน
- `--amp` เป็นตัวเลือก CUDA; baseline ใช้ FP32 ไม่เปลี่ยนค่าความละเอียดเงียบ ๆ ผลฝึกใหม่อาจต่างจากเดิมตาม precision/device/order of operations
- เริ่ม Windows ด้วย workers 0; ได้ทดสอบ workers 2 แล้ว ส่วน Linux ปรับเพิ่มได้ตามเครื่อง

แต่ละรอบบันทึก `last.pt` (model/optimizer/scheduler/scaler/RNG), `metrics.jsonl`, `config.json` และ `best.pt` เมื่อมี validation ที่ดีขึ้น ไม่มี validation จะไม่สร้าง `best.pt`

### ฝึกต่อหลังปิดโปรแกรม

```powershell
& $pyMFCC train.py --scale 4 --device cuda --lr-dir 'D:/SR/DIV2K/DIV2K_train_LR_bicubic/X4' --hr-dir 'D:/SR/DIV2K/DIV2K_train_HR' --val-lr-dir 'D:/SR/benchmark/Set5/LR_bicubic/X4' --val-hr-dir 'D:/SR/benchmark/Set5/HR' --resume 'D:/SR/runs/previous-run/last.pt' --lr 1e-5 --batch-size 4 --accumulate 4 --lr-patch 48 --epochs 150 --steps-per-epoch 1000 --workers 0
```

เปลี่ยนพาธ checkpoint เป็นไฟล์จริง คง scale/batch/accumulate/patch/LR schedule/seed/AMP/workers ให้ตรงรอบก่อน ระบบตรวจค่าเหล่านี้และปฏิเสธหากต่างกัน การ resume เริ่มหลัง epoch ที่บันทึกเสร็จและเขียนในโฟลเดอร์รอบใหม่ ไม่กู้ตำแหน่งกลาง epoch ใช้ `--init` เมื่อตั้งใจเปลี่ยนการทดลองหรือเริ่มจากไฟล์ model-only เก่า

## 5. License และการอ้างอิง

เก็บ MIT notices จากโค้ดต้นทางทั้งสองฉบับไว้ใน [LICENSE](LICENSE) และข้อมูลอ้างอิงใน [CITATION.cff](CITATION.cff) ชุด benchmark ไม่ได้รวมใน repo น้ำหนัก Journal ทั้งสี่สเกลอยู่ใน `weights/` (ตรวจ SHA-256 กับ `weights/manifest.json`)

## ข้อแตกต่างที่ตั้งใจแก้

แก้ positional `bias` ที่กลายเป็น boolean stride; ใช้ strict state-dict loading พร้อมตรวจ shared aliases; ไม่มี scheduler `verbose` ที่ถูกยกเลิก; ทุก entrypoint มี main guard; ใช้การบันทึกภาพแบบตรงแทน nested multiprocessing target; ไม่ใช้ binary dataset cache หรือ `rm -rf`; บันทึก checkpoint แบบ atomic และใช้ `weights_only=True` โหลดไฟล์

ทั้งสาม body ต้องอ้าง **RG และ convolution ตัวเดียวกัน** การสร้าง RG ใหม่สามตัวจะเปลี่ยนโมเดล อย่าแก้ส่วนนี้เป็น independent branches หากยังต้องการผล Journal
