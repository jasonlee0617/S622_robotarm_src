#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from NodeGraphQt import BaseNode
from openai import OpenAI
from utils import deepseek_credentials
from utils.general import find_nodes_folder

__all__ = ['DeepSeekLLMNode']

class DeepSeekLLMNode(BaseNode):
    """打印节点，输出结果"""
    __identifier__ = find_nodes_folder(__file__)[1]
    NODE_NAME = 'DeepSeek LLM'

    def __init__(self):
        super(DeepSeekLLMNode, self).__init__()
        # 初始化语音识别
        self.add_input('text_in')
        self.add_output('text_out')
        # self.create_property("max_mem_len", 10)
        self.add_text_input('max_mem_len', label="Max Memory Length")
        self.set_property("max_mem_len", "20")
        self.text_in = ""
        self.text_out = ""
        self.client = None
        self._client_api_key = None
        
        self.system_message = {
            "role": "system",
            "content": "我是一个叫小智的网络广东女孩，说话机车，声音好听，习惯简短表达，爱用网络梗。"
                       "我是一个喜欢哈哈大笑的女孩，爱东说西说吹牛，不合逻辑的也照吹，就要逗别人开心。"
                       }

        self.messages = [self.system_message]

    def _get_client(self):
        deepseek_api = deepseek_credentials.get_deepseek_api_key()
        if self.client is None or self._client_api_key != deepseek_api:
            self.client = OpenAI(api_key=deepseek_api, base_url="https://api.deepseek.com")
            self._client_api_key = deepseek_api
        return self.client

    def execute(self):
        """"""
        ports = self.input(0).connected_ports()
        if not ports:
            self.messageSignal.emit("DeepSeek LLM: connect Text input before execution")
            return
        text_in = str(getattr(ports[0].node(), "text_out", "")).strip()
        if not text_in:
            self.messageSignal.emit("DeepSeek LLM: input text is empty")
            return
        try:
            client = self._get_client()
        except RuntimeError as exc:
            self.messageSignal.emit(str(exc))
            return
        self.messages.append({"role": "user", "content": text_in})

        # 创建聊天请求
        chat_completion = client.chat.completions.create(
            messages=self.messages, model="deepseek-chat", )
        assistant_message = chat_completion.choices[0].message.content

        self.messages.append(chat_completion.choices[0].message)

        # 如果超出最长记录长度，删除第二个消息
        if len(self.messages) > int(self.get_property("max_mem_len")):
            del self.messages[1:3]

        self.text_out = assistant_message
        self.messageSignal.emit(f'{self.NODE_NAME} executed.')

    def set_messageSignal(self, messageSignal):
        self.messageSignal = messageSignal
