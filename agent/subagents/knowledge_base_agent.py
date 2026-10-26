"""知识库子智能体：通过 RAGFlow 检索内部 SOP、数据字典等文档。"""

from agent.prompts import sub_agents_content
from tools.ragflow_tools import get_assistant_list , create_ask_delete

knowledge_base_agent = {
    "name":sub_agents_content['ragflow']['name'],
    "description":sub_agents_content['ragflow']['description'],
    "system_prompt":sub_agents_content['ragflow']['system_prompt'],
    "tools":[get_assistant_list,create_ask_delete]
}