import requests
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
from .exceptions import RAGflowAPIError

class RAGflowClient:
    """RAGflow API 客户端，用于与 RAGflow 服务交互。"""

    def __init__(self, base_url: str, api_key: str):
        """
        初始化 RAGflow 客户端。

        Args:
            base_url (str): RAGflow 服务的基础 URL，例如 'http://localhost:5000'
            api_key (str): API 密钥，用于身份验证
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    # 数据集管理
    def create_dataset(self, name: str, avatar: Optional[str] = None, description: Optional[str] = None,
                      language: str = "English", embedding_model: str = "BAAI/bge-zh-v1.5",
                      permission: str = "me", chunk_method: str = "naive",
                      parser_config: Optional[Dict] = None) -> Dict:
        """
        创建一个新的数据集。

        Args:
            name (str): 数据集的唯一名称
            avatar (Optional[str]): Base64 编码的头像
            description (Optional[str]): 数据集描述
            language (str): 数据集语言，默认 "English"
            embedding_model (str): 嵌入模型名称，默认 "BAAI/bge-zh-v1.5"
            permission (str): 访问权限，默认 "me"
            chunk_method (str): 分块方法，默认 "naive"
            parser_config (Optional[Dict]): 解析器配置

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets"
        data = {
            "name": name,
            "avatar": avatar,
            "description": description,
            "language": language,
            "embedding_model": embedding_model,
            "permission": permission,
            "chunk_method": chunk_method,
            "parser_config": parser_config or {}
        }
        response = requests.post(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"创建数据集失败: {response.status_code} - {response.text}")
        return response.json()

    def delete_datasets(self, ids: List[str]) -> Dict:
        """
        根据 ID 删除数据集。

        Args:
            ids (List[str]): 要删除的数据集 ID 列表

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets"
        data = {"ids": ids}
        response = requests.delete(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"删除数据集失败: {response.status_code} - {response.text}")
        return response.json()

    def update_dataset(self, dataset_id: str, name: Optional[str] = None,
                      embedding_model: Optional[str] = None, chunk_method: Optional[str] = None) -> Dict:
        """
        更新指定数据集的配置。

        Args:
            dataset_id (str): 数据集 ID
            name (Optional[str]): 更新的名称
            embedding_model (Optional[str]): 更新的嵌入模型
            chunk_method (Optional[str]): 更新的分块方法

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}"
        data = {k: v for k, v in {"name": name, "embedding_model": embedding_model, "chunk_method": chunk_method}.items() if v is not None}
        response = requests.put(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"更新数据集失败: {response.status_code} - {response.text}")
        return response.json()

    def list_datasets(self, page: int = 1, page_size: int = 30, orderby: str = "create_time",
                     desc: bool = True, name: Optional[str] = None, id: Optional[str] = None) -> Dict:
        """
        列出数据集。

        Args:
            page (int): 页码，默认 1
            page_size (int): 每页数量，默认 30
            orderby (str): 排序字段，默认 "create_time"
            desc (bool): 是否降序，默认 True
            name (Optional[str]): 数据集名称过滤
            id (Optional[str]): 数据集 ID 过滤

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets"
        params = {"page": page, "page_size": page_size, "orderby": orderby, "desc": str(desc).lower()}
        if name:
            params["name"] = name
        if id:
            params["id"] = id
        response = requests.get(url, params=params, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"列出数据集失败: {response.status_code} - {response.text}")
        return response.json()

    # 文件管理
    def upload_documents(self, dataset_id: str, file_paths: List[Union[str, Path]]) -> Dict:
        """
        上传文档到指定数据集。

        Args:
            dataset_id (str): 数据集 ID
            file_paths (List[Union[str, Path]]): 要上传的文件路径列表

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/documents"
        files = [('file', (Path(fp).name, open(fp, 'rb'))) for fp in file_paths]
        headers = {'Authorization': self.headers['Authorization']}
        response = requests.post(url, files=files, headers=headers)
        for _, f in files:
            f[1].close()
        if not response.ok:
            raise RAGflowAPIError(f"上传文档失败: {response.status_code} - {response.text}")
        return response.json()

    def update_document(self, dataset_id: str, document_id: str, name: Optional[str] = None,
                       chunk_method: Optional[str] = None, parser_config: Optional[Dict] = None) -> Dict:
        """
        更新指定文档的配置。

        Args:
            dataset_id (str): 数据集 ID
            document_id (str): 文档 ID
            name (Optional[str]): 更新后的名称
            chunk_method (Optional[str]): 更新后的分块方法
            parser_config (Optional[Dict]): 更新后的解析器配置

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/documents/{document_id}"
        data = {k: v for k, v in {"name": name, "chunk_method": chunk_method, "parser_config": parser_config}.items() if v is not None}
        response = requests.put(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"更新文档失败: {response.status_code} - {response.text}")
        return response.json()

    def download_document(self, dataset_id: str, document_id: str, output_path: Union[str, Path]) -> None:
        """
        下载指定文档。

        Args:
            dataset_id (str): 数据集 ID
            document_id (str): 文档 ID
            output_path (Union[str, Path]): 保存文件的路径

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/documents/{document_id}"
        response = requests.get(url, headers=self.headers, stream=True)
        if not response.ok:
            raise RAGflowAPIError(f"下载文档失败: {response.status_code} - {response.text}")
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def list_documents(self, dataset_id: str, page: int = 1, page_size: int = 30,
                      orderby: str = "create_time", desc: bool = True, keywords: Optional[str] = None,
                      id: Optional[str] = None, name: Optional[str] = None) -> Dict:
        """
        列出指定数据集中的文档。

        Args:
            dataset_id (str): 数据集 ID
            page (int): 页码，默认 1
            page_size (int): 每页数量，默认 30
            orderby (str): 排序字段，默认 "create_time"
            desc (bool): 是否降序，默认 True
            keywords (Optional[str]): 标题关键词过滤
            id (Optional[str]): 文档 ID 过滤
            name (Optional[str]): 文档名称过滤

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/documents"
        params = {"page": page, "page_size": page_size, "orderby": orderby, "desc": str(desc).lower()}
        if keywords:
            params["keywords"] = keywords
        if id:
            params["id"] = id
        if name:
            params["name"] = name
        response = requests.get(url, params=params, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"列出文档失败: {response.status_code} - {response.text}")
        return response.json()

    def delete_documents(self, dataset_id: str, ids: List[str]) -> Dict:
        """
        删除指定数据集中的文档。

        Args:
            dataset_id (str): 数据集 ID
            ids (List[str]): 要删除的文档 ID 列表

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/documents"
        data = {"ids": ids}
        response = requests.delete(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"删除文档失败: {response.status_code} - {response.text}")
        return response.json()

    def parse_documents(self, dataset_id: str, document_ids: List[str]) -> Dict:
        """
        解析指定数据集中的文档。

        Args:
            dataset_id (str): 数据集 ID
            document_ids (List[str]): 要解析的文档 ID 列表

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/chunks"
        data = {"document_ids": document_ids}
        response = requests.post(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"解析文档失败: {response.status_code} - {response.text}")
        return response.json()

    def stop_parsing_documents(self, dataset_id: str, document_ids: List[str]) -> Dict:
        """
        停止解析指定数据集中的文档。

        Args:
            dataset_id (str): 数据集 ID
            document_ids (List[str]): 要停止解析的文档 ID 列表

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/chunks"
        data = {"document_ids": document_ids}
        response = requests.delete(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"停止解析文档失败: {response.status_code} - {response.text}")
        return response.json()

    # 分块管理
    def add_chunk(self, dataset_id: str, document_id: str, content: str,
                 important_keywords: Optional[List[str]] = None) -> Dict:
        """
        添加分块到指定文档。

        Args:
            dataset_id (str): 数据集 ID
            document_id (str): 文档 ID
            content (str): 分块内容
            important_keywords (Optional[List[str]]): 重要关键词列表

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks"
        data = {"content": content, "important_keywords": important_keywords or []}
        response = requests.post(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"添加分块失败: {response.status_code} - {response.text}")
        return response.json()

    def list_chunks(self, dataset_id: str, document_id: str, keywords: Optional[str] = None,
                   page: int = 1, page_size: int = 1024, id: Optional[str] = None) -> Dict:
        """
        列出指定文档中的分块。

        Args:
            dataset_id (str): 数据集 ID
            document_id (str): 文档 ID
            keywords (Optional[str]): 内容关键词过滤
            page (int): 页码，默认 1
            page_size (int): 每页数量，默认 1024
            id (Optional[str]): 分块 ID 过滤

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks"
        params = {"page": page, "page_size": page_size}
        if keywords:
            params["keywords"] = keywords
        if id:
            params["id"] = id
        response = requests.get(url, params=params, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"列出分块失败: {response.status_code} - {response.text}")
        return response.json()

    def delete_chunks(self, dataset_id: str, document_id: str, chunk_ids: List[str]) -> Dict:
        """
        删除指定文档中的分块。

        Args:
            dataset_id (str): 数据集 ID
            document_id (str): 文档 ID
            chunk_ids (List[str]): 要删除的分块 ID 列表

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks"
        data = {"chunk_ids": chunk_ids}
        response = requests.delete(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"删除分块失败: {response.status_code} - {response.text}")
        return response.json()

    def update_chunk(self, dataset_id: str, document_id: str, chunk_id: str,
                    content: Optional[str] = None, important_keywords: Optional[List[str]] = None,
                    available: Optional[bool] = None) -> Dict:
        """
        更新指定分块的内容或配置。

        Args:
            dataset_id (str): 数据集 ID
            document_id (str): 文档 ID
            chunk_id (str): 分块 ID
            content (Optional[str]): 更新后的内容
            important_keywords (Optional[List[str]]): 更新后的重要关键词
            available (Optional[bool]): 更新后的可用性状态

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks/{chunk_id}"
        data = {k: v for k, v in {"content": content, "important_keywords": important_keywords, "available": available}.items() if v is not None}
        response = requests.put(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"更新分块失败: {response.status_code} - {response.text}")
        return response.json()

    def retrieve_chunks(self, question: str, dataset_ids: Optional[List[str]] = None,
                       document_ids: Optional[List[str]] = None, page: int = 1, page_size: int = 30,
                       similarity_threshold: float = 0.2, vector_similarity_weight: float = 0.3,
                       top_k: int = 1024, rerank_id: Optional[str] = None, keyword: bool = False,
                       highlight: bool = False) -> Dict:
        """
        从指定数据集中检索分块。

        Args:
            question (str): 用户查询
            dataset_ids (Optional[List[str]]): 数据集 ID 列表
            document_ids (Optional[List[str]]): 文档 ID 列表
            page (int): 页码，默认 1
            page_size (int): 每页数量，默认 30
            similarity_threshold (float): 相似度阈值，默认 0.2
            vector_similarity_weight (float): 向量相似度权重，默认 0.3
            top_k (int): 向量计算中使用的分块数，默认 1024
            rerank_id (Optional[str]): 重排模型 ID
            keyword (bool): 是否启用关键词匹配，默认 False
            highlight (bool): 是否高亮匹配项，默认 False

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/retrieval"
        data = {
            "question": question,
            "dataset_ids": dataset_ids or [],
            "document_ids": document_ids or [],
            "page": page,
            "page_size": page_size,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "top_k": top_k,
            "rerank_id": rerank_id,
            "keyword": keyword,
            "highlight": highlight
        }
        response = requests.post(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"检索分块失败: {response.status_code} - {response.text}")
        return response.json()

    # 聊天助手管理
    def create_chat(self, name: str, avatar: Optional[str] = None, dataset_ids: Optional[List[str]] = None,
                   llm: Optional[Dict] = None, prompt: Optional[Dict] = None) -> Dict:
        """
        创建一个聊天助手。

        Args:
            name (str): 聊天助手名称
            avatar (Optional[str]): Base64 编码的头像
            dataset_ids (Optional[List[str]]): 关联的数据集 ID 列表
            llm (Optional[Dict]): LLM 设置
            prompt (Optional[Dict]): 提示设置

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/chats"
        data = {
            "name": name,
            "avatar": avatar,
            "dataset_ids": dataset_ids or [],
            "llm": llm or {},
            "prompt": prompt or {}
        }
        response = requests.post(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"创建聊天助手失败: {response.status_code} - {response.text}")
        return response.json()

    def update_chat(self, chat_id: str, name: Optional[str] = None, avatar: Optional[str] = None,
                   dataset_ids: Optional[List[str]] = None, llm: Optional[Dict] = None,
                   prompt: Optional[Dict] = None) -> Dict:
        """
        更新指定聊天助手的配置。

        Args:
            chat_id (str): 聊天助手 ID
            name (Optional[str]): 更新后的名称
            avatar (Optional[str]): 更新后的头像
            dataset_ids (Optional[List[str]]): 更新后的数据集 ID 列表
            llm (Optional[Dict]): 更新后的 LLM 设置
            prompt (Optional[Dict]): 更新后的提示设置

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/chats/{chat_id}"
        data = {k: v for k, v in {"name": name, "avatar": avatar, "dataset_ids": dataset_ids, "llm": llm, "prompt": prompt}.items() if v is not None}
        response = requests.put(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"更新聊天助手失败: {response.status_code} - {response.text}")
        return response.json()

    def delete_chats(self, ids: List[str]) -> Dict:
        """
        删除指定聊天助手。

        Args:
            ids (List[str]): 要删除的聊天助手 ID 列表

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/chats"
        data = {"ids": ids}
        response = requests.delete(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"删除聊天助手失败: {response.status_code} - {response.text}")
        return response.json()

    def list_chats(self, page: int = 1, page_size: int = 30, orderby: str = "create_time",
                  desc: bool = True, name: Optional[str] = None, id: Optional[str] = None) -> Dict:
        """
        列出聊天助手。

        Args:
            page (int): 页码，默认 1
            page_size (int): 每页数量，默认 30
            orderby (str): 排序字段，默认 "create_time"
            desc (bool): 是否降序，默认 True
            name (Optional[str]): 聊天助手名称过滤
            id (Optional[str]): 聊天助手 ID 过滤

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/chats"
        params = {"page": page, "page_size": page_size, "orderby": orderby, "desc": str(desc).lower()}
        if name:
            params["name"] = name
        if id:
            params["id"] = id
        response = requests.get(url, params=params, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"列出聊天助手失败: {response.status_code} - {response.text}")
        return response.json()

    # 会话管理
    def create_session(self, chat_id: str, name: str, user_id: Optional[str] = None) -> Dict:
        """
        创建与聊天助手的会话。

        Args:
            chat_id (str): 聊天助手 ID
            name (str): 会话名称
            user_id (Optional[str]): 用户定义的 ID

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/chats/{chat_id}/sessions"
        data = {"name": name}
        if user_id:
            data["user_id"] = user_id
        response = requests.post(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"创建会话失败: {response.status_code} - {response.text}")
        return response.json()

    def update_session(self, chat_id: str, session_id: str, name: Optional[str] = None,
                      user_id: Optional[str] = None) -> Dict:
        """
        更新聊天助手的会话。

        Args:
            chat_id (str): 聊天助手 ID
            session_id (str): 会话 ID
            name (Optional[str]): 更新后的名称
            user_id (Optional[str]): 更新后的用户 ID

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/chats/{chat_id}/sessions/{session_id}"
        data = {k: v for k, v in {"name": name, "user_id": user_id}.items() if v is not None}
        response = requests.put(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"更新会话失败: {response.status_code} - {response.text}")
        return response.json()

    def list_sessions(self, chat_id: str, page: int = 1, page_size: int = 30,
                     orderby: str = "create_time", desc: bool = True, name: Optional[str] = None,
                     id: Optional[str] = None, user_id: Optional[str] = None) -> Dict:
        """
        列出聊天助手的会话。

        Args:
            chat_id (str): 聊天助手 ID
            page (int): 页码，默认 1
            page_size (int): 每页数量，默认 30
            orderby (str): 排序字段，默认 "create_time"
            desc (bool): 是否降序，默认 True
            name (Optional[str]): 会话名称过滤
            id (Optional[str]): 会话 ID 过滤
            user_id (Optional[str]): 用户 ID 过滤

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/chats/{chat_id}/sessions"
        params = {"page": page, "page_size": page_size, "orderby": orderby, "desc": str(desc).lower()}
        if name:
            params["name"] = name
        if id:
            params["id"] = id
        if user_id:
            params["user_id"] = user_id
        response = requests.get(url, params=params, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"列出会话失败: {response.status_code} - {response.text}")
        return response.json()

    def delete_sessions(self, chat_id: str, ids: List[str]) -> Dict:
        """
        删除聊天助手的会话。

        Args:
            chat_id (str): 聊天助手 ID
            ids (List[str]): 要删除的会话 ID 列表

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/chats/{chat_id}/sessions"
        data = {"ids": ids}
        response = requests.delete(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"删除会话失败: {response.status_code} - {response.text}")
        return response.json()

    def converse_with_chat(self, chat_id: str, question: str, stream: bool = True,
                         session_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict:
        """
        与聊天助手进行对话。

        Args:
            chat_id (str): 聊天助手 ID
            question (str): 问题
            stream (bool): 是否流式输出，默认 True
            session_id (Optional[str]): 会话 ID
            user_id (Optional[str]): 用户 ID

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/chats/{chat_id}/completions"
        data = {"question": question, "stream": stream}
        if session_id:
            data["session_id"] = session_id
        if user_id:
            data["user_id"] = user_id
        response = requests.post(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"与聊天助手对话失败: {response.status_code} - {response.text}")
        return response.json()

    # Agent 管理
    def create_agent_session(self, agent_id: str, params: Optional[Dict] = None, user_id: Optional[str] = None) -> Dict:
        """
        创建与代理的会话。

        Args:
            agent_id (str): 代理 ID
            params (Optional[Dict]): 开始组件的参数
            user_id (Optional[str]): 用户定义的 ID

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/agents/{agent_id}/sessions"
        data = params or {}
        if user_id:
            data["user_id"] = user_id
        response = requests.post(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"创建代理会话失败: {response.status_code} - {response.text}")
        return response.json()

    def converse_with_agent(self, agent_id: str, question: str, stream: bool = True,
                          session_id: Optional[str] = None, user_id: Optional[str] = None,
                          extra_params: Optional[Dict] = None) -> Dict:
        """
        与代理进行对话。

        Args:
            agent_id (str): 代理 ID
            question (str): 问题
            stream (bool): 是否流式输出，默认 True
            session_id (Optional[str]): 会话 ID
            user_id (Optional[str]): 用户 ID
            extra_params (Optional[Dict]): 额外的参数

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/agents/{agent_id}/completions"
        data = {"question": question, "stream": stream}
        if session_id:
            data["session_id"] = session_id
        if user_id:
            data["user_id"] = user_id
        if extra_params:
            data.update(extra_params)
        response = requests.post(url, json=data, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"与代理对话失败: {response.status_code} - {response.text}")
        return response.json()

    def list_agent_sessions(self, agent_id: str, page: int = 1, page_size: int = 30,
                           orderby: str = "create_time", desc: bool = True,
                           id: Optional[str] = None, user_id: Optional[str] = None) -> Dict:
        """
        列出代理的会话。

        Args:
            agent_id (str): 代理 ID
            page (int): 页码，默认 1
            page_size (int): 每页数量，默认 30
            orderby (str): 排序字段，默认 "create_time"
            desc (bool): 是否降序，默认 True
            id (Optional[str]): 会话 ID 过滤
            user_id (Optional[str]): 用户 ID 过滤

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/agents/{agent_id}/sessions"
        params = {"page": page, "page_size": page_size, "orderby": orderby, "desc": str(desc).lower()}
        if id:
            params["id"] = id
        if user_id:
            params["user_id"] = user_id
        response = requests.get(url, params=params, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"列出代理会话失败: {response.status_code} - {response.text}")
        return response.json()

    def list_agents(self, page: int = 1, page_size: int = 30, orderby: str = "create_time",
                   desc: bool = True, name: Optional[str] = None, id: Optional[str] = None) -> Dict:
        """
        列出代理。

        Args:
            page (int): 页码，默认 1
            page_size (int): 每页数量，默认 30
            orderby (str): 排序字段，默认 "create_time"
            desc (bool): 是否降序，默认 True
            name (Optional[str]): 代理名称过滤
            id (Optional[str]): 代理 ID 过滤

        Returns:
            Dict: API 响应数据

        Raises:
            RAGflowAPIError: 如果 API 请求失败
        """
        url = f"{self.base_url}/api/v1/agents"
        params = {"page": page, "page_size": page_size, "orderby": orderby, "desc": str(desc).lower()}
        if name:
            params["name"] = name
        if id:
            params["id"] = id
        response = requests.get(url, params=params, headers=self.headers)
        if not response.ok:
            raise RAGflowAPIError(f"列出代理失败: {response.status_code} - {response.text}")
        return response.json()