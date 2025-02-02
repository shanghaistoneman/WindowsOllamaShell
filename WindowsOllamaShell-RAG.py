import tkinter as tk
from tkinter import ttk

from ollama import generate
from ollama import list
from ollama import ListResponse

import ollama

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb

from langchain_community.document_loaders import TextLoader, CSVLoader, UnstructuredFileLoader, DirectoryLoader, UnstructuredHTMLLoader, JSONLoader, PyPDFLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader, UnstructuredImageLoader

from tkinter import filedialog, messagebox

if not os.path.exists(r".\rag_docs"):
    os.makedirs(r".\rag_docs")

rag_filename = ''

def load_file():
    # 打开文件选择对话框，仅显示文本文件（可根据需要调整）
    global rag_filename
    file_path = filedialog.askopenfilename(
        title="选择要加载的文件",
        initialdir=r".\rag_docs",
        filetypes=(("文本文件", "*.txt"), ("所有文件", "*.*"))
    )
    rag_filename = os.path.basename(file_path)

    if rag_filename:
        # 在文本框中显示选中的文件路径
        text_box.delete(1.0, tk.END)  # 清空文本框
        text_box.insert(tk.END, rag_filename)
    else:
        # 用户取消选择
        text_box.delete(1.0, tk.END)  # 清空文本框
        messagebox.showinfo("提示", "未选择任何文件。")

def creat_rag():
    global rag_filename
    loader = TextLoader(".\\rag_docs\\" + rag_filename, encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    rag_path = os.path.join(os.getcwd(),"rag_docs", "_" + rag_filename + "_","rag")
    if not os.path.exists(rag_path):
        os.makedirs(rag_path)
    DBclient = chromadb.PersistentClient(path = rag_path)
    DBcollection = DBclient.get_or_create_collection(name = rag_filename)

    if not os.path.exists(os.path.join(rag_path,"rag_ok.txt")):
        # 基于ollama运行嵌入模型 granite-embedding:278m （支持包括中文简体在内的多语言）
        for i, d in enumerate(splits):
            response = ollama.embeddings(model="granite-embedding:278m", prompt=d.page_content)
            embedding = response["embedding"]
            DBcollection.add(ids=[str(i)], embeddings=[embedding], documents=[d.page_content])
        with open(os.path.join(rag_path,"rag_ok.txt"), 'w') as file:
            file.write("文件RAG向量化已经完成。")
    return DBcollection

def fetch_models():
    response: ListResponse = ollama.list()
    model_names = [model['model'] for model in response.models]
    return model_names

def on_send():
    model_name = model_var.get()
    question = question_text.get("1.0", tk.END).strip()

    if not model_name or not question:
        return

    global rag_filename
    if rag_filename:
        DBcollection = creat_rag()
        # 从向量库中查询与问题相关的文本块
        response = ollama.embeddings(prompt=question, model="granite-embedding:278m")
        results = DBcollection.query(query_embeddings=[response["embedding"]], n_results=3)
        data = ("参考资料" + results['ids'][0][0] + results['documents'][0][0] + "\n" + 
               "参考资料" + results['ids'][0][1] + results['documents'][0][1] + "\n" + 
               "参考资料" + results['ids'][0][2] + results['documents'][0][2] + "\n")
     
        #将提问的问题与查询到的文本块进行拼接
        question = question + "\n" + data
    response = generate(model_name, question)
    answer =response['response']
    answer_text.delete("1.0", tk.END)
    answer_text.insert("1.0", answer)

# 创建主窗口
root = tk.Tk()
root.title("Windows Ollama Shell")

# 模型下拉选择框
model_label = tk.Label(root, text="选择大模型:")
model_label.pack()
model_var = tk.StringVar(root)
model_combo = ttk.Combobox(root, textvariable=model_var)
model_combo['values'] = fetch_models()
model_combo.pack()

# 创建一个框架来组织“加载知识库”按钮和备注标签
frame = ttk.Frame(root, padding="10 10 10 10")
frame.pack(expand=True)
# 创建“加载”按钮
load_button = ttk.Button(frame, text="加载知识库", command=load_file,)
load_button.pack(side=tk.LEFT, padx=(0, 10))  # 左边放置按钮，右边留空隙
# 创建备注文本标签
tip_label = ttk.Label(frame, text=r"所加载的文件必须在.\rag_docs文件夹内，而且文件名必须为英文", foreground="gray")
tip_label.pack(side=tk.LEFT) 


# 创建并放置文本框用于显示加载的知识库文件
text_box = tk.Text(root, height=1, width=50)
text_box.pack()

# 输入文本框
question_label = tk.Label(root, text="输入你的问题:")
question_label.pack()
question_text = tk.Text(root, height=20, width=200)
question_text.pack()

# 发送按钮
send_button = tk.Button(root, text="发送问题", command=on_send)
send_button.pack()

# 显示答案的文本框
answer_label = tk.Label(root, text="大模型返回结果:")
answer_label.pack()
answer_text = tk.Text(root, height=20, width=200)
answer_text.pack()

# 运行主循环
root.mainloop()
