import os
import time
import json
import multiprocessing as mp
from glob import glob
from tqdm import tqdm

pdf_folder = "/root/mineru2.0/mineru-sglang/测试pdf"

def get_pdf_pages(file_path):
    import fitz
    try:
        with fitz.open(file_path) as doc:
            return doc.page_count
    except Exception:
        return None

def get_max_gpu_used():
    import subprocess
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,nounits",
            shell=True
        )
        lines = out.decode().splitlines()[1:]
        for line in lines:
            fields = [f.strip() for f in line.split(",")]
            if str(pid) == fields[0]:
                return int(fields[1])
    except Exception:
        return None
    return None

def parse_wrapper(args):
    file_path, pdf_id = args
    
    import fitz
    import torch
    from local_test_pipeline import PDFTool
    num_pages = get_pdf_pages(file_path)
    try:
        torch.cuda.reset_peak_memory_stats()
        pdf_tool = PDFTool()
        start = time.time()
        md_content, mid_data, pdf_image_datas, pdf_table_datas = pdf_tool.pdf2md_minerU_v2(file_path, pdf_id)
        duration = time.time() - start
        parsed_pages = len(mid_data) if mid_data else 0

        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else None
        smi_mb = get_max_gpu_used()
        return {
            "pdf_id": pdf_id,
            "file_path": file_path,
            "total_pages": num_pages,
            "parsed_pages": parsed_pages,
            "success": md_content is not None,
            "duration": duration,
            "peak_mem_MB": round(peak_mb, 2) if peak_mb else None,
            "nvidia_smi_MB": smi_mb,
            "error": "" if md_content is not None else "解析失败",
        }
    except Exception as e:
        return {
            "pdf_id": pdf_id,
            "file_path": file_path,
            "total_pages": num_pages,
            "parsed_pages": 0,
            "success": False,
            "duration": 0,
            "peak_mem_MB": None,
            "nvidia_smi_MB": None,
            "error": str(e),
        }

if __name__ == "__main__":
    # launch前，spawn模式保护
    mp.set_start_method('spawn', force=True)

    pdf_files = glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        print(f"未找到任何PDF文件在目录：{pdf_folder}")
        exit(1)

    tasks = [(file, f"test_{i}") for i, file in enumerate(pdf_files[:20])]#要前20个，实际只有14个
    results = []
    pool_size = 14  

    time_start = time.time()
    with mp.Pool(processes=pool_size) as pool:
        for res in tqdm(pool.imap(parse_wrapper, tasks), total=len(tasks), desc="并发解析PDF"):
            results.append(res)

    time_total = time.time() - time_start
    print(f"\n全部解析完成，总耗时：{time_total:.2f}s")
    print(f"平均每个PDF耗时：{time_total/len(tasks):.2f}s")

    succ_list = [r for r in results if r["success"]]
    fail_list = [r for r in results if not r["success"]]
    succ = len(succ_list)
    fail = len(fail_list)

    if len(tasks) > 0:
        succ_rate = succ / len(tasks)
        fail_rate = fail / len(tasks)
    else:
        succ_rate = fail_rate = 0

    if succ > 0:
        avg_duration = sum(r["duration"] for r in succ_list) / succ
        min_duration = min(r["duration"] for r in succ_list)
        max_duration = max(r["duration"] for r in succ_list)
    else:
        avg_duration = min_duration = max_duration = 0

    if succ > 0:
        avg_pages = sum(r["total_pages"] for r in succ_list if r["total_pages"] is not None) / succ
        min_pages = min(r["total_pages"] for r in succ_list if r["total_pages"] is not None)
        max_pages = max(r["total_pages"] for r in succ_list if r["total_pages"] is not None)
    else:
        avg_pages = min_pages = max_pages = 0

    succ_peak_mem = [r["peak_mem_MB"] for r in succ_list if r["peak_mem_MB"]]
    succ_smi_mem = [r["nvidia_smi_MB"] for r in succ_list if r["nvidia_smi_MB"]]
    if succ_peak_mem:
        avg_peak = sum(succ_peak_mem) / len(succ_peak_mem)
        max_peak = max(succ_peak_mem)
        min_peak = min(succ_peak_mem)
    else:
        avg_peak = max_peak = min_peak = 0
    if succ_smi_mem:
        avg_smi = sum(succ_smi_mem) / len(succ_smi_mem)
        max_smi = max(succ_smi_mem)
        min_smi = min(succ_smi_mem)
    else:
        avg_smi = max_smi = min_smi = 0

    if succ > 0:
        long_pdf = max((r for r in succ_list if r["total_pages"]), key=lambda r: r["total_pages"])
        print(f"最长论文: {os.path.basename(long_pdf['file_path'])}, 页数: {long_pdf['total_pages']}, 解析耗时: {long_pdf['duration']:.2f}s")
    else:
        long_pdf = None

    from collections import Counter
    fail_reasons = [r["error"] for r in fail_list]
    fail_counter = Counter(fail_reasons)

    print("\n==== 指标统计 ====")
    print(f"总任务数: {len(tasks)}")
    print(f"解析成功数: {succ}")
    print(f"解析失败数: {fail}")
    print(f"解析成功率: {succ_rate:.2%}")
    print(f"平均解析耗时: {avg_duration:.2f}s，最长耗时: {max_duration:.2f}s，最短: {min_duration:.2f}s")
    print(f"平均页数: {avg_pages:.2f}，最大: {max_pages}，最小: {min_pages}")
    print(f"（PyTorch）平均GPU显存: {avg_peak:.2f} MB，最大: {max_peak:.2f} MB，最小: {min_peak:.2f} MB")
    print(f"（nvidia-smi）平均GPU显存: {avg_smi:.2f} MB，最大: {max_smi:.2f} MB，最小: {min_smi:.2f} MB")
    if fail > 0:
        print(f"失败/异常占比: {fail_rate:.2%}")
        print("失败原因top:")
        for k, v in fail_counter.most_common(5):
            print(f"  {k[:60]}...: {v} 次")
    print("========================\n")

    with open("scene_batch_test_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("详细统计见 scene_batch_test_result.json")
