"""网络搜索子智能体：通过 Tavily 检索行业动态、竞品动作与消费者趋势。"""

from agent.prompts import sub_agents_content
from tools.tavily_tool import internet_search


network_search_agent = {
    "name":sub_agents_content['tavily']['name'],
    "description":sub_agents_content['tavily']['description'],
    "system_prompt":sub_agents_content['tavily']['system_prompt'],
    "tools":[internet_search]
}