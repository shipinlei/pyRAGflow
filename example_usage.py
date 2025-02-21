from ragflow_client import RAGflowClient
from pathlib import Path

def main():
    # 初始化客户端
    base_url = "http://localhost:5000"  # 请替换为实际的 RAGflow 服务地址
    api_key = "YOUR_API_KEY"  # 请替换为实际的 API 密钥
    client = RAGflowClient(base_url, api_key)

    try:
        # 创建数据集
        dataset_name = "example_dataset"
        print("创建数据集...")
        dataset_response = client.create_dataset(name=dataset_name, description="示例数据集")
        dataset_id = dataset_response["data"]["id"]
        print(f"数据集创建成功，ID: {dataset_id}")

        # 上传文档
        test_files_dir = Path("test_files")
        file_paths = [test_files_dir / "test1.txt", test_files_dir / "test2.pdf"]
        print("上传文档...")
        upload_response = client.upload_documents(dataset_id, file_paths)
        document_ids = [doc["id"] for doc in upload_response["data"]]
        print(f"文档上传成功，IDs: {document_ids}")

        # 解析文档
        print("解析文档...")
        parse_response = client.parse_documents(dataset_id, document_ids)
        print("文档解析成功")

        # 添加分块
        print("添加分块...")
        chunk_response = client.add_chunk(dataset_id, document_ids[0], "这是一个测试分块", ["test", "example"])
        chunk_id = chunk_response["data"]["chunk"]["id"]
        print(f"分块添加成功，ID: {chunk_id}")

        # 列出分块
        print("列出分块...")
        chunks_response = client.list_chunks(dataset_id, document_ids[0])
        print(f"分块列表: {chunks_response['data']['chunks']}")

        # 创建聊天助手
        chat_name = "example_chat"
        print("创建聊天助手...")
        chat_response = client.create_chat(name=chat_name, dataset_ids=[dataset_id])
        chat_id = chat_response["data"]["id"]
        print(f"聊天助手创建成功，ID: {chat_id}")

        # 创建会话并对话
        print("创建会话并对话...")
        session_response = client.create_session(chat_id, "example_session")
        session_id = session_response["data"]["id"]
        converse_response = client.converse_with_chat(chat_id, "RAGflow 有什么优势？", session_id=session_id)
        print(f"对话响应: {converse_response['data']['answer']}")

        # 清理资源
        print("清理资源...")
        client.delete_sessions(chat_id, [session_id])
        client.delete_chats([chat_id])
        client.delete_documents(dataset_id, document_ids)
        client.delete_datasets([dataset_id])
        print("资源清理完成")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()