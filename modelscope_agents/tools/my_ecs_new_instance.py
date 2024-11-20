# --coding: utf-8--
from modelscope_agent.tools.base import BaseTool, register_tool
"""
自定义tools，使用的时候暂时还是有问题，只能在源码中改才能生效
"""


@register_tool('my_ecs_new_instance')
class MyAliyunRenewInstanceTool(BaseTool):
    description = '续费一台包年包月的ECS实例'
    name = 'my_ecs_new_instance'
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

