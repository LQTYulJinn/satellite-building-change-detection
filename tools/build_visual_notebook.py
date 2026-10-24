"""Build the interactive Kaggle notebook from the validated base notebook."""

import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5


root = Path(__file__).resolve().parents[1]
source = root / "notebooks" / "LEVIR_CD_Kaggle.ipynb"
target = root / "notebooks" / "LEVIR_CD_Visual_Train.ipynb"
notebook = json.loads(source.read_text(encoding="utf-8"))


def lines(value):
    return value.splitlines(keepends=True)


notebook["cells"][0]["source"] = lines("""# LEVIR-CD: train và xem biểu đồ từng epoch

**Trước khi chạy:** Add Input `mdrifaturrahman33/levir-cd`; bật GPU và Internet. Trong Session options, đặt **Persistence → Files only** trước khi train. Kaggle lưu file ở `/kaggle/working` giữa các phiên tương tác theo chế độ này, nhưng vẫn nên tải ZIP hoặc Quick Save kèm output files ngay khi chạy xong.

Notebook này giữ nguyên kiến trúc, loss, split và seed của thí nghiệm trước. Biểu đồ là để quan sát, không tự làm tăng IoU/F1. Chỉ điều chỉnh cấu hình bằng validation; không chọn bằng test.
""")

notebook["cells"][5]["source"] = lines("""## Early-fusion U-Net: xem tiến độ trực tiếp

Cell sau vẽ **train loss** và **validation F1/IoU** sau mỗi epoch, đồng thời lưu `best.pt` trong `/kaggle/working/satellite-building-change-detection/runs/early/`. Muốn thử cấu hình mới, đổi các biến ở đầu cell trước khi chạy. Không chạy lại cell khi một phiên huấn luyện vẫn đang chạy.
""")

notebook["cells"][6]["source"] = lines("""import csv, time
import matplotlib.pyplot as plt
from IPython.display import clear_output, FileLink

EPOCHS = 30
BATCH_SIZE = 8
BASE_CHANNELS = 32
LEARNING_RATE = 1e-3
WORKERS = 2
SEED = 42
RUN_DIR = REPO / 'runs/early'
RUN_DIR.mkdir(parents=True, exist_ok=True)
command = [sys.executable, '-u', str(REPO / 'train.py'), '--data', str(DATA),
           '--run', str(RUN_DIR), '--architecture', 'early', '--epochs', str(EPOCHS),
           '--batch-size', str(BATCH_SIZE), '--base', str(BASE_CHANNELS),
           '--learning-rate', str(LEARNING_RATE), '--workers', str(WORKERS),
           '--seed', str(SEED)]
log_path = RUN_DIR / 'console.log'

def show_progress(rows):
    epochs = [int(row['epoch']) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [float(row['loss']) for row in rows], marker='o')
    axes[0].set(title='Train loss', xlabel='Epoch', ylabel='BCE + Dice')
    axes[1].plot(epochs, [float(row['val_f1']) for row in rows], marker='o', label='Val F1')
    axes[1].plot(epochs, [float(row['val_iou']) for row in rows], marker='o', label='Val IoU')
    axes[1].set(title='Validation (threshold 0.5)', xlabel='Epoch', ylabel='Score', ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    display(fig)
    plt.close(fig)
    best = max(rows, key=lambda row: float(row['val_f1']))
    print(f\"Epoch {epochs[-1]}/{EPOCHS} | best val F1 {float(best['val_f1']):.4f} at epoch {best['epoch']}\")
    print('Checkpoint:', RUN_DIR / 'best.pt')

print('Running:', ' '.join(command), flush=True)
training_started = time.time()
with log_path.open('w', encoding='utf-8') as log_file:
    process = subprocess.Popen(command, cwd=REPO, stdout=log_file, stderr=subprocess.STDOUT)
    last_epoch = 0
    while process.poll() is None:
        history_path = RUN_DIR / 'history.csv'
        if history_path.exists() and history_path.stat().st_mtime >= training_started:
            with history_path.open(newline='', encoding='utf-8') as history_file:
                rows = list(csv.DictReader(history_file))
            if rows and int(rows[-1]['epoch']) > last_epoch:
                last_epoch = int(rows[-1]['epoch'])
                clear_output(wait=True)
                show_progress(rows)
        time.sleep(10)
    return_code = process.wait()
if return_code:
    raise RuntimeError('Train failed; last log lines:\\n' + '\\n'.join(log_path.read_text(encoding='utf-8').splitlines()[-25:]))
with (RUN_DIR / 'history.csv').open(newline='', encoding='utf-8') as history_file:
    rows = list(csv.DictReader(history_file))
clear_output(wait=True)
show_progress(rows)
print('Training complete. Checkpoint saved:', (RUN_DIR / 'best.pt').exists())
""")

notebook["cells"].insert(7, {
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": lines("""# Chạy cell này sau khi train xong; đánh giá test đúng một lần.
run('evaluate.py', '--data', DATA, '--checkpoint', 'runs/early/best.pt', '--split', 'test')
run('summarize.py')
display(Markdown((REPO / 'runs/results.md').read_text(encoding='utf-8')))
"""),
})

# Keep the optional Siamese comparison and the figure display. Replace the export
# cell with an immediate download link and a check that the checkpoint is present.
export_cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code"
                   and "make_archive" in "".join(cell["source"]))
export_cell["source"] = lines("""import shutil
assert (REPO / 'runs/early/best.pt').is_file(), 'Thiếu best.pt; kiểm tra phiên train.'
archive = shutil.make_archive('/kaggle/working/levir_cd_results', 'zip', root_dir=REPO, base_dir='runs')
print('ZIP đã tạo:', archive)
display(FileLink(archive))  # nhấp để tải ngay; sau đó Quick Save kèm output files
""")

for index, cell in enumerate(notebook["cells"]):
    cell["id"] = str(uuid5(NAMESPACE_URL, f"visual-levir-cd-{index}"))[:8]

target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(target)
