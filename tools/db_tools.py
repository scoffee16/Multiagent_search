# --- 允许直接运行本文件（python tools/db_tools.py）：把项目根目录加入 sys.path ---
# 直接运行子目录脚本时 sys.path[0] 是脚本所在目录，找不到 api/tools/utils 包；
# 被当作模块导入时（例如 api/server.py 导入）__package__ 不为空，这段自动跳过。
if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import re

from dotenv import load_dotenv
from api.monitor import monitor
from mysql.connector import connect, Error
from langchain_core.tools import tool

load_dotenv()


# 加载配置文件方便后续使用
def get_db_config():
    """Get database configuration from environment variables."""
    config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4"),
        "collation": os.getenv("MYSQL_COLLATION", "utf8mb4_unicode_ci"),
        "autocommit": True,
        "connection_timeout": int(os.getenv("MYSQL_TIMEOUT", "10")),
        "sql_mode": os.getenv("MYSQL_SQL_MODE", "TRADITIONAL"),
    }
    # 移除 None 值（核心必要操作）
    config = {k: v for k, v in config.items() if v is not None}

    # 补充：校验核心配置是否存在（可选但推荐）
    required_keys = ["user", "password", "database"]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"缺失数据库核心配置：{', '.join(missing_keys)}")

    return config


# ======================== SQL 只读守卫 ========================
# SQL 由模型自己拼，prompt 里的“只许查询”约束靠不住，这里做一层硬拦截：
#   1) 只放行 SELECT / WITH / SHOW / DESC(RIBE) / EXPLAIN
#   2) 禁止多语句（分号拼接）
#   3) 剥离字符串字面量与注释后再查写操作关键字，避免 'update' 出现在数据里被误杀
_ALLOWED_SQL_PREFIXES = ("select", "with", "show", "desc", "describe", "explain")
_FORBIDDEN_SQL_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "truncate", "create",
    "replace", "rename", "grant", "revoke", "call", "lock", "unlock",
    "load", "outfile", "dumpfile",
)


def _strip_sql_literals(sql: str) -> str:
    """把字符串字面量、反引号标识符、注释替换成空格，返回仅含代码骨架的 SQL。"""
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in ("'", '"', "`"):
            quote = ch
            i += 1
            while i < n:
                if sql[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                if sql[i] == quote:
                    if i + 1 < n and sql[i + 1] == quote:  # '' / "" 转义
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append(" ")
        elif sql.startswith("--", i) or ch == "#":
            nl = sql.find("\n", i)
            i = n if nl == -1 else nl
            out.append(" ")
        elif sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            out.append(" ")
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def assert_read_only_sql(query: str) -> str:
    """校验 SQL 为单条只读语句；不合法抛 ValueError，合法返回去掉尾部空语句的 SQL。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("SQL 不能为空")

    sql = query.strip()
    while sql.endswith(";"):
        sql = sql[:-1].strip()
    if not sql:
        raise ValueError("SQL 不能为空")

    bare = _strip_sql_literals(sql)
    if ";" in bare:
        raise ValueError("只允许执行单条 SQL（检测到多条语句）")

    head = bare.strip().lstrip("(").strip().lower()
    if not head.startswith(_ALLOWED_SQL_PREFIXES):
        raise ValueError(
            "只允许只读查询（SELECT/WITH/SHOW/DESC/EXPLAIN），当前语句："
            f"{sql[:40]}"
        )

    lowered = bare.lower()
    for keyword in _FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            raise ValueError(f"只允许只读查询，语句里出现了写操作关键字：{keyword.upper()}")

    return sql

@tool
def list_sql_tables()->str:
    """
    查询当前库中所有可用的表！
    作用：为了模型识别有哪些可用的表！方便进行后续的自定义sql查询
    :return: 有表： 可用的表有：表1,表2,表3....  没有表: 没有可用的表   出现异常：查询出现异常：异常信息
    """

    # 埋点,调用工具了告诉前端哪个工具被调用了！！
    monitor.report_tool(tool_name="数据库表名查询工具：list_sql_tables", args={})
    # 加载数据库信息配置
    config = get_db_config()

    # 1. 创建一个链接
    # 2. 创建cursor
    # 3. cursor执行sql语句
    # 4. cursor获取返回结果
    # 5. 释放连接和cursor资源
    # 确保要捕捉异常信息，返回异常提示，避免直接报错！
    try:
        # 确保资源使用完毕一定释放 with
        with connect(**config) as  conn:
            with conn.cursor() as cursor:
                sql = "show tables"
                cursor.execute(sql)
                # 捕捉执行结果 要所有的表名称
                # [(表1),(表2),(表3)]
                tables = cursor.fetchall()
                if not tables:
                    return "没有可用的表"
                # 可用的表有：表1,表2,表3....
                # [表1,表2,表3]
                table_names = [table[0] for table in tables]
                return f"可用的表有：{', '.join(table_names)}"
    except Error as e:
        return f"查询出现异常：{str(e)}"


@tool
def get_table_data(table_name)->str:
    """
    查询指定表名的数据！当前工具调用之前，必须先调用list_sql_tables完成表名的校验！
    此工具的作用：1.可以完成单表数据的查询 2. 可以为多表查询提供表结果信息（列名&数据格式）
    :param table_name: 表名
    :return: csv格式的数据（模拟表格数据格式）
             1.第一行是列信息，列之间使用,（英文的逗号）分割
             2.第二行开始是表数据，值之间也使用,(英文的逗号)分割
             3.行和行之间使用\n分割
             4.至多表数据查询100条
             例如：
                id,name,age\n -> 列头
                1,张三,18\n
                1,张三,18\n    -> 至多查询100条
                1,张三,18\n
                1,张三,18\n
    """
    # 埋点,调用工具了告诉前端哪个工具被调用了！！
    monitor.report_tool(tool_name="数据库表数据查询工具：get_table_data", args={"table_name":table_name})

    # 获取数据库参数
    config = get_db_config()
    # 1. 创建一个链接
    # 2. 创建cursor
    # 3. cursor执行sql语句
    # 4. cursor获取返回结果
    # 5. 释放连接和cursor资源
    # 确保要捕捉异常信息，返回异常提示，避免直接报错！
    try:
        # 1. 创建一个链接
        with connect(**config) as  conn:
            # 2. 创建cursor
            with conn.cursor() as cursor:
                # 3. cursor执行sql语句
                sql = f"select * from {table_name} limit 100"
                cursor.execute(sql)
                # 4. cursor获取返回结果
                # 4.1 获取列的信息
                # 返回的查询结果的列的信息
                # description => [(id,列长度...),(),()]
                # 如果查询没有结果 -》 description 也是None
                description = cursor.description
                if not description:
                    return f"数据表：{table_name}为空没有数据！"
                # 4.2 获取查询结果
                # description =>  [(id,列长度...),(date,....),()] => 元组 index = 0 列名
                # [列1,列2,列3...]
                columns = [ desc[0] for desc in description ] # [1,2,3,4]
                # 表数据
                # [(1,张三),(2,李四),(3,二狗子)]
                rows = cursor.fetchall()
                # (1,张三) -> ('1','张三') -> '1,张三'
                # ['1,张三','1,张三','1,张三','1,张三','1,张三']
                results = [ ",".join(map(str,row)) for row in rows]

                # columns -> csv -> header
                # id,name,age
                header_str = ",".join(columns)
                # '1,张三'\n
                data_str = "\n".join(results)
                return f"{header_str}\n{data_str}"
    except Error as e:
        return f"查询出现异常：{str(e)}"


@tool
def execute_sql_query(query)->str:
    """
    执行自定义查询sql语句！切记：执行之前，需要通过执行 list_sql_tables明确表名！执行get_table_data
    明确表结构和数据格式！
    :param query: 要执行的自定义sql语句
    :return: csv格式的数据（模拟表格数据格式）
             1.第一行是列信息，列之间使用,（英文的逗号）分割
             2.第二行开始是表数据，值之间也使用,(英文的逗号)分割
             3.行和行之间使用\n分割
             4.至多表数据查询100条
             例如：
                id,name,age\n -> 列头
                1,张三,18\n
                1,张三,18\n    -> 至多查询100条
                1,张三,18\n
                1,张三,18\n
    """
    # 埋点,调用工具了告诉前端哪个工具被调用了！！
    monitor.report_tool(tool_name="数据库表数据查询工具：execute_sql_query", args={"query":query})

    # 1. 创建一个链接
    # 2. 创建cursor
    # 3. cursor执行sql语句
    # 4. cursor获取返回结果
    # 5. 释放连接和cursor资源
    # 确保要捕捉异常信息，返回异常提示，避免直接报错！
    try:
        # 只读守卫：必须在连库之前，拒掉写操作/多语句（ValueError 由下面统一兜住返回给模型）
        query = assert_read_only_sql(query)
        # 获取数据库参数
        config = get_db_config()
        # 1. 创建一个链接
        with connect(**config) as  conn:
            # 2. 创建cursor
            with conn.cursor() as cursor:
                # 3. cursor执行sql语句
                cursor.execute(query)
                # 4. cursor获取返回结果
                # 4.1 获取列的信息
                # 返回的查询结果的列的信息
                # description => [(id,列长度...),(),()]
                # 如果查询没有结果 -》 description 也是None
                description = cursor.description
                if not description:
                    return f"执行自定义SQL语句查询没有结果，sql为：{query}！"
                # 4.2 获取查询结果
                # description =>  [(id,列长度...),(date,....),()] => 元组 index = 0 列名
                # [列1,列2,列3...]
                columns = [ desc[0] for desc in description ] # [1,2,3,4]
                # 表数据
                # [(1,张三),(2,李四),(3,二狗子)]
                rows = cursor.fetchall()
                # (1,张三) -> ('1','张三') -> '1,张三'
                # ['1,张三','1,张三','1,张三','1,张三','1,张三']
                results = [ ",".join(map(str,row)) for row in rows]

                # columns -> csv -> header
                # id,name,age
                header_str = ",".join(columns)
                # '1,张三'\n
                data_str = "\n".join(results)
                return f"{header_str}\n{data_str}"
    except (Error, ValueError) as e:
        return f"查询出现异常：{str(e)}"



if __name__ == "__main__":
    # execute_sql_query 被 @tool 装饰成了 StructuredTool，不能像普通函数那样直接调用，
    # 必须走 .invoke(...)（直接调用会抛 TypeError: 'StructuredTool' object is not callable）
    print(execute_sql_query.invoke("SELECT * FROM orders WHERE category = '数码电子'"))






