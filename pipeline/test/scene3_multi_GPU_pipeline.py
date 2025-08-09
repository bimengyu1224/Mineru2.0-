import os
import time
import json
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

pdf_folder = "/root/mineru2.0/mineru-sglang/测试pdf"

def get_max_gpu_used():
    import subprocess, os
    pid = os.getpid()
    try:
        out = subprocess.check_output(f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv,nounits", shell=True)
        for line in out.decode().splitlines()[1:]:
            fields = [f.strip() for f in line.split(",")]
            if str(pid) == fields[0]:
                return int(fields[1])
    except Exception:
        return None
    return None

def parse_wrapper(args):
    """
    子进程负责设定当前唯一可见的GPU，并初始化模型
    """
    file_path, pdf_id, gpu_id = args

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import torch
    from local_test2 import PDFTool

    try:
        pdf_tool = PDFTool(lang="ch")
        torch.cuda.reset_peak_memory_stats()

        start = time.time()
        md_content, mid_data, pdf_image_datas, pdf_table_datas = pdf_tool.pdf2md_minerU_v2(file_path, pdf_id)
        duration = time.time() - start

        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else None
        smi_mb = get_max_gpu_used()

        return {
            "pdf_id": pdf_id,
            "file_path": file_path,
            "gpu_id": gpu_id,
            "success": md_content is not None,
            "duration": duration,
            "peak_mem_MB": round(peak_mb,2) if peak_mb else None,
            "nvidia_smi_MB": smi_mb,
            "error": "" if md_content is not None else "解析失败"
        }
    except Exception as e:
        import traceback
        return {
            "pdf_id": pdf_id,
            "file_path": file_path,
            "gpu_id": gpu_id,
            "success": False,
            "duration": 0,
            "peak_mem_MB": None,
            "nvidia_smi_MB": None,
            "error": str(e) + "\n" + traceback.format_exc()
        }

if __name__ == "__main__":
    # 1. 获取PDF
    pdf_files = glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        print(f"未找到任何PDF文件在目录：{pdf_folder}")
        exit(1)

    # 2. 指定显卡
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpu_ids = list(range(count))
    except Exception:
        print("自动获取GPU失败，默认用[0,1]")
        gpu_ids = [0, 1]
    num_gpus = len(gpu_ids)

    # 3. 按GPU数量打包任务，每个进程只处理指定卡上的任务
    tasks = []
    for i, file in enumerate(pdf_files[:num_gpus * 10]):  # 每卡最多10个任务举例
        gpu_id = gpu_ids[i % num_gpus]
        tasks.append((file, f"test_{i}", gpu_id))

    results = []
    time_start = time.time()
    with Pool(processes=num_gpus) as pool:
        for r in tqdm(pool.imap_unordered(parse_wrapper, tasks), total=len(tasks), desc="多卡并发解析PDF"):
            results.append(r)

    time_total = time.time() - time_start
    print(f"\n全部解析完成，总耗时：{time_total:.2f}s")
    print(f"平均每个PDF耗时：{time_total/len(tasks):.2f}s")
    succ = sum(1 for r in results if r["success"])
    fail = sum(1 for r in results if not r["success"])
    print(f"成功：{succ}，失败：{fail}")

    valid = [r["peak_mem_MB"] for r in results if r["peak_mem_MB"]]
    if valid:
        print(f"平均peak显存占用：{sum(valid)/len(valid):.2f} MB")
    valid_smi = [r["nvidia_smi_MB"] for r in results if r["nvidia_smi_MB"]]
    if valid_smi:
        print(f"平均nvidia-smi报告显存占用：{sum(valid_smi)/len(valid_smi):.2f} MB")

    with open("pipeline_batch_multigpu_test_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("详细统计见 pipeline_batch_multigpu_test_result.json")