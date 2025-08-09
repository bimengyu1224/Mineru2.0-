import os
import time
import json
import multiprocessing as mp
from glob import glob
from tqdm import tqdm

pdf_folder = "/root/mineru2.0/mineru-sglang/场景二_超长分页论文"

def get_pdf_pages(file_path):
    import fitz
    try:
        with fitz.open(file_path) as doc:
            return doc.page_count
    except Exception:
        return None

def get_process_memory_mb():
    # 当前进程常驻内存MB
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def get_gpu_memory_mb():
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 / 1024
    except Exception:
        return 0

def parse_wrapper(args):
    file_path, pdf_id = args
    import fitz
    from local_test_vlm_sglang import PDFToolSGLang

    num_pages = get_pdf_pages(file_path)
    mem_peak_mb = 0
    gpu_mem_samples = []

    try:
        pdf_tool = PDFToolSGLang(lang="ch", backend="sglang-client", server_url="http://localhost:30000")
        start = time.time()
        import threading

        # 实时监控线程
        stop_flag = [False]
        def monitor():
            nonlocal mem_peak_mb, gpu_mem_samples
            while not stop_flag[0]:
                m = get_process_memory_mb()
                mem_peak_mb = max(mem_peak_mb, m)
                gmem = get_gpu_memory_mb()
                gpu_mem_samples.append(gmem)
                time.sleep(0.5)  # 每0.5秒采样一次

        t = threading.Thread(target=monitor)
        t.daemon = True
        t.start()

        md_content, mid_data, pdf_image_datas, pdf_table_datas = pdf_tool.pdf2md_sglang(file_path, pdf_id)
        duration = time.time() - start
        parsed_pages = len(mid_data) if mid_data else 0

        stop_flag[0] = True
        t.join(timeout=1)
        avg_gpu_mem = sum(gpu_mem_samples) / len(gpu_mem_samples) if gpu_mem_samples else 0

        return {
            "pdf_id": pdf_id,
            "file_path": file_path,
            "total_pages": num_pages,
            "parsed_pages": parsed_pages,
            "success": md_content is not None,
            "duration": duration,
            "mem_peak_mb": round(mem_peak_mb, 2),
            "gpu_mem_avg_mb": round(avg_gpu_mem, 2),
            "error": "" if md_content is not None else "解析失败",
        }
    except Exception as e:
        stop_flag[0] = True
        if 't' in locals():
            t.join(timeout=1)
        avg_gpu_mem = sum(gpu_mem_samples) / len(gpu_mem_samples) if gpu_mem_samples else 0
        return {
            "pdf_id": pdf_id,
            "file_path": file_path,
            "total_pages": num_pages,
            "parsed_pages": 0,
            "success": False,
            "duration": 0,
            "mem_peak_mb": round(mem_peak_mb, 2),
            "gpu_mem_avg_mb": round(avg_gpu_mem, 2),
            "error": str(e),
        }

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    pdf_files = glob(os.path.join(pdf_folder, "*.pdf"))
    print(f"总找到PDF文件数：{len(pdf_files)}")

    if not pdf_files:
        print(f"在目录 {pdf_folder} 下没有PDF文件!")
        exit(1)
    else:
        print(f"共{len(pdf_files)}个PDF将全部用于并行解析。")
        all_pdfs = []
        for file in tqdm(pdf_files, desc="统计页数"):
            pages = get_pdf_pages(file)
            all_pdfs.append((file, pages))
        all_pdfs.sort(key=lambda x: -x[1] if x[1] is not None else 0)

    N = min(len(all_pdfs), 20)
    tasks = [(file_path, f"longpdf_{i}_pages{pages}") for i, (file_path, pages) in enumerate(all_pdfs[:N])]
    results = []
    pool_size = os.cpu_count() // 2 or 4

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

    # ------------------ 资源统计 ------------------
    mem_peaks = [r["mem_peak_mb"] for r in succ_list]
    gpu_avgs = [r["gpu_mem_avg_mb"] for r in succ_list]
    mem_peak_max = max(mem_peaks) if mem_peaks else 0
    mem_peak_avg = sum(mem_peaks)/len(mem_peaks) if mem_peaks else 0
    gpu_mem_peak = max(gpu_avgs) if gpu_avgs else 0
    gpu_mem_avg = sum(gpu_avgs)/len(gpu_avgs) if gpu_avgs else 0
    # -------------------------------------------------

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
    else:
        avg_duration = min_duration = max_duration = 0
        avg_pages = min_pages = max_pages = 0

    if succ > 0:
        long_pdf = max((r for r in succ_list if r["total_pages"]), key=lambda r: r["total_pages"])
        print(f"最长论文: {os.path.basename(long_pdf['file_path'])}, 页数: {long_pdf['total_pages']}, 解析耗时: {long_pdf['duration']:.2f}s")
    else:
        long_pdf = None

    from collections import Counter
    fail_reasons = [r["error"] for r in fail_list]
    fail_counter = Counter(fail_reasons)

    print("\n==== 指标统计 ====")
    print(f"批量论文总数: {len(tasks)}")
    print(f"解析成功数: {succ}")
    print(f"解析失败数: {fail}")
    print(f"解析成功率: {succ_rate:.2%}")
    print(f"平均解析耗时: {avg_duration:.2f}s，最长耗时: {max_duration:.2f}s，最短: {min_duration:.2f}s")
    print(f"平均页数: {avg_pages:.2f}，最大: {max_pages}，最小: {min_pages}")
    print(f"资源消耗峰值 (MB): {mem_peak_max:.2f}，平均: {mem_peak_avg:.2f}")
    print(f"平均GPU显存占用 (MB): {gpu_mem_avg:.2f}（所有任务周期均值），任务最大GPU均值: {gpu_mem_peak:.2f}")
    if fail > 0:
        print(f"失败/异常占比: {fail_rate:.2%}")
        print("失败原因top:")
        for k, v in fail_counter.most_common(5):
            print(f"  {k[:60]}...: {v} 次")
    print("========================\n")

    with open("scene_batch_test_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("详细统计见 scene_batch_test_result.json")