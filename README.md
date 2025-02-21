# pyRAGflow

这是一个用于与 RAGflow API 交互的 Python 客户端，支持数据集管理、文件管理、分块管理、聊天助手管理以及代理管理的完整功能。

## 安装

```bash
pip install pyRAGflow
```
或者从源码安装：
```bash
git clone https://github.com/shipinlei/pyRAGflow.git
cd ragflow-client
pip install .
```
## 使用示例
```python
from pyRAGflow import RAGflowClient

# 初始化客户端
client = RAGflowClient(base_url="http://localhost:5000", api_key="YOUR_API_KEY")

# 创建数据集
response = client.create_dataset(name="my_dataset", description="测试数据集")
dataset_id = response["data"]["id"]
print(f"数据集 ID: {dataset_id}")
```
更多用法请参考 example_usage.py。
## 依赖
Python 3.9+
requests>=2.28.0

## 开发
运行测试：
```bash
pytest tests/
```
## 许可
MIT License
## 依赖项列表 (requirements.txt)
requests>=2.28.0
pytest>=7.0.0  # 可选，用于测试

## 测试文件示例 (tests/test_api.py)
```python
import pytest
from pyRAGflow import RAGflowClient

@pytest.fixture
def client():
    return RAGflowClient("http://localhost:5000", "TEST_API_KEY")

def test_create_dataset(client):
    # 注意：此测试需要真实的 RAGflow 服务运行
    response = client.create_dataset(name="test_dataset")
    assert response["code"] == 0
    assert "data" in response

```

