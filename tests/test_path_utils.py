"""utils.path_utils.resolve_path 的路径隔离单测。

这几个用例就是文档里那张表的可执行版本：虚拟前缀清洗、会话内相对路径、
session 嵌套防护、updated/ 重定向，以及新增的 `..` 越界拦截。
"""

from pathlib import Path

import pytest

from utils.path_utils import resolve_path


@pytest.fixture()
def session(tmp_path) -> Path:
    s = tmp_path / "output" / "session_abc"
    s.mkdir(parents=True)
    return s.resolve()


def test_strip_virtual_prefix(session):
    assert Path(resolve_path("/workspace/report.md", str(session))) == session / "report.md"


def test_mnt_data_prefix(session):
    assert Path(resolve_path("/mnt/data/notes.md", str(session))) == session / "notes.md"


def test_unix_style_absolute_without_drive(session):
    assert Path(resolve_path("/sub/test.md", str(session))) == session / "sub" / "test.md"


def test_relative_subdir(session):
    assert Path(resolve_path("sub1/sub2/a.md", str(session))) == session / "sub1" / "sub2" / "a.md"


def test_output_prefix_is_stripped(session):
    assert Path(resolve_path("output/report.md", str(session))) == session / "report.md"


def test_session_name_duplication_guarded(session):
    """session_abc/report.md 不能再拼一层 session_abc。"""
    assert Path(resolve_path("session_abc/report.md", str(session))) == session / "report.md"


def test_nested_absolute_path_collapsed(session):
    nested = str(session / "session_abc" / "report.md")
    assert Path(resolve_path(nested, str(session))) == session / "report.md"


def test_updated_redirects_to_cwd(session):
    got = Path(resolve_path("abc/updated/upload/f.pdf", str(session)))
    assert got == (Path.cwd() / "updated" / "upload" / "f.pdf").resolve()


def test_safe_dotdot_inside_session(session):
    """sub/../ok.md 还在会话内，允许，并且要把 .. 归一化掉。"""
    assert Path(resolve_path("sub/../ok.md", str(session))) == session / "ok.md"


@pytest.mark.parametrize("evil", ["../../etc/passwd", "..\\..\\Windows\\win.ini", "a/../../b.md"])
def test_dotdot_escape_rejected(session, evil):
    with pytest.raises(ValueError, match="越出会话目录"):
        resolve_path(evil, str(session))


def test_dotdot_without_session_is_plain_resolve():
    """没有会话上下文时保持老行为：就当普通相对路径解析。"""
    assert Path(resolve_path("../x.md")) == Path("../x.md").resolve()
