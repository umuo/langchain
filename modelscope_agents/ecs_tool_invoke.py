# --coding: utf-8--
from modelscope_agent.agents import RolePlay

role_template = '你扮演一名ECS实例管理员，你需要根据用户的要求来满足他们'
llm_config = {
    'model': 'qwen2.5:7b',
    'model_server': 'ollama',
    'host': '10.118.21.135'
}
function_list = ['MyRenewInstance']
bot = RolePlay(llm=llm_config,
               function_list=function_list,
               instruction=role_template)
response = bot.run('"请帮我续费一台ECS实例，实例id是：i-rj90a7e840y5cde，续费时长10个月"',
                   remote=False, print_info=True)
text = ''
for chunk in response:
    text += chunk
print(text)
