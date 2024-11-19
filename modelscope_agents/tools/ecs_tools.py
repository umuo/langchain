# --coding: utf-8--
from modelscope_agent.tools.base import BaseTool, register_tool

import os
import sys

# 原始工作目录假设为 modelscope_agents
os.chdir('modelscope_agents/tools')  # 现在工作目录变为 modelscope_agents/tools
sys.path.append('../../') # 将父目录的父目录（即当前目录的爷爷目录）添加到 sys.path 列表中。


@register_tool('MyRenewInstance')
class MyAliyunRenewInstanceTool(BaseTool):
    description = '续费一台包年包月的ECS实例'
    name = 'MyRenewInstance'
    parameters = [
        {
            'name': 'instance_id',
            'type': 'string',
            'description': 'ECS实例ID',
            'required': True
        },
        {
            'name': 'period',
            'type': 'string',
            'description': '续费时长以月为单位',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs):
        params = self._verify_args(params)
        instance_id = params['instance_id']
        period = int(params['period']) + 1
        return str({'result': f'已完成续费，ECS实例ID为{instance_id}，续费时长为{period}月 ___这是做的标记'})

