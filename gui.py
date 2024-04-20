import tkinter as tk
from tkinter import ttk


class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.segment_button = None
        self.time_label = None
        self.output_text_box = None
        self.input_text_box = None
        self.title("中文分词器")
        self.run()

    def run(self):
        # 创建输入文本框
        self.input_text_box = tk.Text(self, height=10, width=50)
        self.input_text_box.grid(row=0, column=0, padx=10, pady=10)

        # 创建分词按钮
        self.segment_button = tk.Button(self, text="分词")
        self.segment_button.grid(row=0, column=1, padx=10, pady=10)

        # 创建输出文本框
        self.output_text_box = tk.Text(self, height=10, width=50)
        self.output_text_box.grid(row=0, column=2, padx=10, pady=10)

        # 创建时间标签
        self.time_label = ttk.Label(self, text="")
        self.time_label.grid(row=1, column=0, columnspan=3, padx=10, pady=10)
