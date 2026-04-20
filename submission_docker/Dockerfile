#FROM python:3.9-slim
#WORKDIR /workspace
#
#
#COPY requirements.txt .
#
#RUN pip install --no-cache-dir -r requirements.txt
#
#
#COPY lib/ lib/
#COPY model/ model/
#COPY main.py .
#COPY run.sh .
#
#RUN sed -i 's/\r$//' run.sh && chmod +x run.sh
#
#CMD ["sh", "run.sh"]


FROM python:3.9-slim

WORKDIR /workspace

# 使用国内源加速
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .

# 安装依赖（从 PyTorch 官方源获取 CPU 版本）
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY lib/ lib/
COPY model/ model/
COPY main.py .
COPY run.sh .
RUN sed -i 's/\r$//' run.sh && chmod +x run.sh

CMD ["sh", "run.sh"]