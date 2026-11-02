"""数据库查询子智能体：自然语言转 SQL，查询 MySQL 中的订单与经营数据。"""

from agent.prompts import sub_agents_content
from agent.tool_observer import instrument_tool
from tools.db_tools import execute_sql_query, get_table_data, list_sql_tables

database_query_agent = {
    "name": sub_agents_content["db"]["name"],
    "description": sub_agents_content["db"]["description"],
    "system_prompt": sub_agents_content["db"]["system_prompt"],
    "tools": [
        instrument_tool(list_sql_tables),
        instrument_tool(get_table_data),
        instrument_tool(execute_sql_query),
    ],
}
