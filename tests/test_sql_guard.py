"""SQL 只读守卫的单测。

背景：SQL 由模型自己拼，prompt 里的约束靠不住。assert_read_only_sql 在连库之前
把非只读语句拦掉，execute_sql_query 负责把 ValueError 变成模型能看懂的提示。
"""

import pytest

from tools.db_tools import assert_read_only_sql, execute_sql_query

ALLOWED = [
    "SELECT * FROM orders",
    "select count(*) from orders where category = '数码电子'",
    "  SELECT 1  ",
    "SELECT 1;",  # 尾部空语句，去掉后仍然合法
    "WITH t AS (SELECT 1 AS a) SELECT * FROM t",
    "SHOW TABLES",
    "DESC orders",
    "describe orders",
    "EXPLAIN SELECT * FROM orders",
    "(SELECT 1) UNION (SELECT 2)",
    # 关键字出现在字符串字面量里不算违规
    "SELECT * FROM orders WHERE note = 'please update this row'",
    # 注释同理
    "SELECT * FROM orders -- update later\n",
]

REJECTED = [
    "DELETE FROM orders",
    "delete from orders where id = 1",
    "UPDATE orders SET amount = 0",
    "DROP TABLE orders",
    "TRUNCATE orders",
    "INSERT INTO orders VALUES (1)",
    "CREATE TABLE t (id INT)",
    "ALTER TABLE orders ADD COLUMN x INT",
    "REPLACE INTO orders VALUES (1)",
    "SELECT 1; DROP TABLE orders",  # 多语句
    "SELECT * FROM orders FOR UPDATE",  # 写锁
    "SHOW TABLES; DELETE FROM orders",
    "SELECT * INTO OUTFILE '/tmp/x' FROM orders",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_read_only_sql_allowed(sql):
    assert assert_read_only_sql(sql).strip()


@pytest.mark.parametrize("sql", REJECTED)
def test_write_sql_rejected(sql):
    with pytest.raises(ValueError):
        assert_read_only_sql(sql)


@pytest.mark.parametrize("sql", ["", "   ", "\n\t", None, 123])
def test_empty_or_non_string_rejected(sql):
    with pytest.raises(ValueError):
        assert_read_only_sql(sql)


def test_trailing_semicolons_stripped():
    assert assert_read_only_sql("SELECT 1;;;  ") == "SELECT 1"


def test_empty_literal_does_not_confuse_parser():
    """'' 是合法空串转义，不能被当成没闭合的引号把后半段吞掉。"""
    assert assert_read_only_sql("SELECT '' AS a, 'delete' AS b")


def test_tool_blocks_write_before_touching_db(monkeypatch):
    """写操作必须在连库之前拦下：这里把 get_db_config 换成炸弹，证明压根没走到。"""

    def _boom():
        raise AssertionError("不应该走到连库这一步")

    monkeypatch.setattr("tools.db_tools.get_db_config", _boom)

    out = execute_sql_query.invoke("DROP TABLE orders")

    assert "查询出现异常" in out
    assert "只读" in out
