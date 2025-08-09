import os
import time
import json
import multiprocessing as mp
from glob import glob
from tqdm import tqdm

pdf_folder = "/root/mineru2.0/mineru-sglang/测试pdf/极限测试/100篇pdf"

def get_pdf_pages(file_path):
    import fitz
    try:
        with fitz.open(file_path) as doc:
            return doc.page_count
    except Exception:
        return None

def get_process_memory_mb():
    """返回当前进程常驻内存(MB)"""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory_mb():
    """
    返回占用的GPU显存(MB)，多个卡则返回第0卡的情况
    """
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # 只统计第0号卡
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 / 1024
    except Exception:
        return None

def parse_wrapper(args):
    file_path, pdf_id = args
    
    from local_test_vlm_sglang import PDFToolSGLang
    import fitz

    mem_peak_mb = 0
    gpu_mem_used = []
    num_pages = get_pdf_pages(file_path)
    try:
        pdf_tool = PDFToolSGLang(lang="ch", backend="sglang-client", server_url="http://localhost:30000")
        start = time.time()
        # 内存监控间隔采样
        import threading

        def mem_gpu_monitor():
            nonlocal mem_peak_mb, gpu_mem_used
            while not done[0]:
                m = get_process_memory_mb()
                mem_peak_mb = max(mem_peak_mb, m)
                g = get_gpu_memory_mb()
                if g is not None:
                    gpu_mem_used.append(g)
                time.sleep(0.5)  # half second periodic

        done = [False]
        t = threading.Thread(target=mem_gpu_monitor)
        t.daemon = True
        t.start()

        md_content, mid_data, pdf_image_datas, pdf_table_datas = pdf_tool.pdf2md_sglang(file_path, pdf_id)
        duration = time.time() - start
        parsed_pages = len(mid_data) if mid_data else 0

        done[0] = True
        t.join(timeout=1)

        avg_gpu_mem = sum(gpu_mem_used)/len(gpu_mem_used) if gpu_mem_used else 0

        return {
            "pdf_id": pdf_id,
            "file_path": file_path,
            "total_pages": num_pages,
            "parsed_pages": parsed_pages,
            "success": md_content is not None,
            "duration": duration,
            "mem_peak_mb": round(mem_peak_mb,2),
            "gpu_mem_avg_mb": round(avg_gpu_mem,2),
            "error": "" if md_content is not None else "解析失败",
        }
    except Exception as e:
        done[0] = True
        if 't' in locals():
            t.join(timeout=1)
        avg_gpu_mem = sum(gpu_mem_used)/len(gpu_mem_used) if gpu_mem_used else 0
        return {
            "pdf_id": pdf_id,
            "file_path": file_path,
            "total_pages": num_pages,
            "parsed_pages": 0,
            "success": False,
            "duration": 0,
            "mem_peak_mb": round(mem_peak_mb,2),
            "gpu_mem_avg_mb": round(avg_gpu_mem,2),
            "error": str(e),
        }

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    pdf_files = glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        print(f"未找到任何PDF文件在目录：{pdf_folder}")
        exit(1)

    N = min(50,len(pdf_files))  
    tasks = [(file, f"test_{i}") for i, file in enumerate(pdf_files[:N])]
    results = []
    pool_size = 50

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
        avg_pages = sum(r["total_pages"] for r in succ_list if r["total_pages"] is not None) / succ
        min_pages = min(r["total_pages"] for r in succ_list if r["total_pages"] is not None)
        max_pages = max(r["total_pages"] for r in succ_list if r["total_pages"] is not None)
        mem_peaks = [r["mem_peak_mb"] for r in succ_list]
        gpu_avgs = [r["gpu_mem_avg_mb"] for r in succ_list]
        mem_peak_max = max(mem_peaks) if mem_peaks else 0
        mem_peak_avg = sum(mem_peaks)/len(mem_peaks) if mem_peaks else 0
        gpu_mem_peak = max(gpu_avgs) if gpu_avgs else 0
        gpu_mem_avg = sum(gpu_avgs)/len(gpu_avgs) if gpu_avgs else 0
    else:
        avg_duration = min_duration = max_duration = 0
        avg_pages = min_pages = max_pages = 0
        mem_peak_max = mem_peak_avg = gpu_mem_peak = gpu_mem_avg = 0

    if succ > 0:
        long_pdf = max((r for r in succ_list if r["total_pages"]), key=lambda r: r["total_pages"])
        print(f"最大页数论文: {os.path.basename(long_pdf['file_path'])}, 页数: {long_pdf['total_pages']}, 解析耗时: {long_pdf['duration']:.2f}s")
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
    print(f"资源消耗峰值 (内存MB): {mem_peak_max:.2f}，平均: {mem_peak_avg:.2f}")
    print(f"平均GPU显存用量 (MB): {gpu_mem_avg:.2f}，任务最大(平均): {gpu_mem_peak:.2f}")
    if fail > 0:
        print(f"失败/异常占比: {fail_rate:.2%}")
        print("失败原因top:")
        for k, v in fail_counter.most_common(5):
            print(f"  {k[:60]}...: {v} 次")
    print("========================\n")

    with open("scene_batch_test_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("详细统计见 scene_batch_test_result.json")
