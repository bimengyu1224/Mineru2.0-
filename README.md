# MinerU2.0 本地部署与测试指南
---

## 目录
1. [环境准备](#环境准备)
2. [模型下载](#模型下载)
3. [Pipeline 本地部署说明](#pipeline-代码说明)
4. [sglang docker部署说明](#sglang-模型部署)
5. [测试代码说明](#测试代码说明)
---
### 1.环境准备
- Python 版本：推荐 Python 3.10
    ```bash  
 
    conda create -n mineru310 python=3.10 -y  
    conda activate mineru310 
### 2.模型下载
- 安装依赖包
    ```bash  
    pip install uv -i https://mirrors.aliyun.com/pypi/simple
    uv pip install -U "mineru[core]" -i https://mirrors.aliyun.com/pypi/simple #mineru[core]包含除sgalng加速外的所有核心功能
    # 也可直接选择安装mineru[all]
    #uv pip install -U "mineru[all]" -i https://mirrors.aliyun.com/pypi/simple
- 模型权重下载
    ```bash
    mineru-models-download
    #运行后会出现选择：选择modelscope下载路径，选择all
    #下载完成后，mineru.json会自动写入用户目录下
    #可以自由移动模型文件夹到其他位置，但需要在 mineru.json 中更新模型路径。
    #如果后续需要更新模型文件，可以再次运行mineru-models-download 命令
### 3.Pipeline 本地部署说明
- 下载依赖包
    ```bash
    pip install -r requirements_pdf.txt 
- 将log文件夹和constant.py复制到项目根目录下
- 运行Pipeline本地测试代码
    ```bash
    python pipeline_local_test.py
    #会输出result.md和mid_data.json两个结果文件
### 4.sglang docker部署

- 构建docker镜像
    ```bash
    wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/Dockerfile
    docker build -t mineru-sglang:latest -f Dockerfile 
    #Mineru的docker使用了lmsysorg/sglang作为基础镜像，在docker中默认集成了sglang推理加速框架和必需的依赖环境。可以直接使用sglang加速VLM模型推理。
- 启动docker容器
    ```bash
    docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 -p 7860:7860 -p 8000:8000 \
    --ipc=host \
    -it mineru-sglang:latest \
    /bin/bash
    #如果端口被占用，请端口改为其他端口。并对应修改compose.yaml文件中的端口映射
- 通过Docker compose快速启动
    ```bash
    # 下载 compose.yaml 文件
    wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/compose.yaml
- 启动sglang服务
    ```bash
    # 通过vlm-sglang-client后端连接sglang-server
    docker compose -f compose.yaml --profile sglang-server up -d
- 启动Web API服务
    ```bash
    docker compose -f compose.yaml --profile api up -d

- 如果端口有改动：
对应修改local_test_vlm_sglang.py文件中server_url映射的端口位置代码和测试程序入口位置模型调用代码
    ```bash
    class PDFToolSGLang:
    def __init__(self,lang="ch", backend="sglang-client",server_url="http://localhost:3000"):
        self.lang = lang
        self.backend = backend
        self.server_url = server_url
        log.info(f"MinerU SGLang Tool 初始化完成 backend={self.backend} server_url={self.server_url}")

    #测试程序入口位置模型调用代码
    pdf_tool = PDFToolSGLang(lang="ch",backend="sglang-client", server_url="http://localhost:30000")
- 挂起sglang后，运行代码即可
    ```bash
    python local_test_vlm_sglang.py
    #会输出resuly.md和mid_data.json两个结果文件
### 5.测试代码说明
- 测试文件准备
  * 极限测试的100个pdf文件可直接运行极限测试_pdf.py脚本自动下载
    ```bash
    python 极限测试_pdf.py
    #脚本在极限测试文件夹下
    ```
  * 其余测试场景所用pdf在对应场景名文件夹下
- 测试不同场景时在对应的场景测试py文件中修改pdf_folder即可
    ```bash
    pdf_folder = "/root/mineru2.0/mineru-sglang/测试pdf/极限测试/100篇pdf"
    ```
