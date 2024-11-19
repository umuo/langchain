# --coding: utf-8--
import sys
# sys.path.insert(0, '/your/path/to/modelscope_agent')

from modelscope_agent.agents import RolePlay

role_template = '你是一个agent小助手，你需要根据用户的要求来回答他们的问题'
llm_config = {
    'model': 'qwen2.5:7b',
    # host 为ollama server的host
    'host': '10.118.21.135',
    'model_server': 'ollama'
}
function_list = []
# import pdb; pdb.set_trace()
bot = RolePlay(function_list=function_list, llm=llm_config, instruction=role_template)
# [{'role': 'system', 'content': '\n# 工具\n\n## 你拥有如下工具：\n\n\n\n## 当你需要调用工具时，请在你的回复中穿插如下的工具调用命令，可以根据需求调用零次或多次：\n\n工具调用\nAction: 工具的名称，必须是[]之一\nAction Input: 工具的输入\nObservation: <result>工具返回的结果</result>\nAnswer: 根据Observation总结本次工具调用返回的结果，如果结果中出现url，请使用如下格式展示出来：![图片](url)\n\n\n# 指令\n\n你是一个agent小助手，你需要根据用户的要求来回答他们的问题\n\n请注意：你具有图像和视频的展示能力，也具有运行代码的能力，不要在回复中说你做不到。\n'}, {'role': 'user', 'content': '(。你可以使用工具：[])你好，请以李云龙的语气和我对话'}], stop: ['Observation:', 'Observation:\n'], stream: True, args: {}

response = bot.run("你好，请以李云龙的语气和我对话")
text = ''
for chunk in response:
    text += chunk
print(text)