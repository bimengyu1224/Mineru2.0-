import arxiv
import os

download_dir = "/root/mineru2.0/mineru-sglang/测试pdf/极限测试/100篇pdf"
os.makedirs(download_dir, exist_ok=True)

search = arxiv.Search(
    query="all",              # or "computer vision" or "llm", etc.
    max_results=100,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

cnt = 0
for result in search.results():
    pdf_path = os.path.join(download_dir, f"{cnt}_{result.get_short_id()}.pdf")
    if not os.path.exists(pdf_path):
        result.download_pdf(dirpath=download_dir)
        cnt += 1
        print(f"Downloaded: {pdf_path}")
    if cnt >= 100:
        break