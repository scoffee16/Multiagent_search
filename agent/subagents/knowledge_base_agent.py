"""知识库子智能体：通过 RAGFlow 检索内部 SOP、数据字典等文档。"""

from agent.prompts import sub_agents_content
from agent.tool_observer import instrument_tool
from tools.ragflow_tools import create_ask_delete, get_assistant_list

knowledge_base_agent = {
    "name": sub_agents_content["ragflow"]["name"],
    "description": sub_agents_content["ragflow"]["description"],
    "system_prompt": sub_agents_content["ragflow"]["system_prompt"],
    "tools": [instrument_tool(get_assistant_list), instrument_tool(create_ask_delete)],
}
