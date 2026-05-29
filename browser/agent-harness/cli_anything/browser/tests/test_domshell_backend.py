"""Wire-format tests for cli_anything.browser.utils.domshell_backend.

These tests patch the async ``_call_execute`` helper and assert the exact
command string sent to the DOMShell ``domshell_execute`` tool, so wire-format
regressions (quoting, command names, multi-line layout, restore ordering)
fail loudly.
"""

import pytest
from unittest.mock import AsyncMock, call, patch

from cli_anything.browser.utils import domshell_backend as backend


# ── grep: command string and call sequencing ──────────────────────────


@patch.object(backend, "_call_execute", new_callable=AsyncMock)
def test_grep_unrooted_produces_single_grep_call(mock_call):
    """Unrooted grep dispatches one ``grep <pattern>`` call."""
    mock_call.return_value = {}

    backend.grep("Login")

    assert mock_call.call_args_list == [call("grep Login", False)]


@patch.object(backend, "_call_execute", new_callable=AsyncMock)
def test_grep_rooted_emits_single_multiline_call(mock_call):
    """Rooted grep is ONE multi-line ``cd / grep / cd back`` execute call.

    Each ``_call_execute`` in non-daemon mode opens a fresh MCP session
    that lands in its own DOMShell 2.x lane, so splitting cd / grep /
    restore across separate calls would lose the cwd between them. The
    three lines must travel in one ``domshell_execute`` call to share
    a lane.
    """
    mock_call.return_value = {}

    backend.grep("Login", path="/main", prev="/")

    assert mock_call.call_args_list == [
        call("cd /main\ngrep Login\ncd /", False),
    ]


@patch.object(backend, "_call_execute", new_callable=AsyncMock)
def test_grep_rooted_uses_single_call_for_lane_isolation(mock_call):
    """Documents the lane-isolation contract: ONE call, not three.

    The trailing ``cd prev`` is delivered as the final line of the same
    multi-line command and runs even if ``grep`` errors, per DOMShell's
    documented continue-on-error semantics
    (`apireno/DOMShell#46 <https://github.com/apireno/DOMShell/issues/46>`_).
    """
    mock_call.return_value = {}

    backend.grep("Login", path="/main", prev="/")

    assert mock_call.call_count == 1


@patch.object(backend, "_call_execute", new_callable=AsyncMock)
def test_grep_rooted_quotes_path_with_spaces(mock_call):
    """Paths with whitespace are shell-quoted inside the multi-line command."""
    mock_call.return_value = {}

    backend.grep("Login", path="/path with spaces", prev="/")

    cmd = mock_call.call_args.args[0]
    # shlex.quote single-quotes anything containing whitespace; the rest
    # of the multi-line layout (grep + cd back) stays intact.
    assert cmd.startswith("cd '/path with spaces'\n")
    assert "\ngrep Login\n" in cmd
    assert cmd.endswith("\ncd /")


@patch.object(backend, "_call_execute", new_callable=AsyncMock)
def test_grep_pattern_with_shell_metacharacters_quoted(mock_call):
    """Patterns with shell metacharacters get quoted (no injection via grep)."""
    mock_call.return_value = {}

    backend.grep("$(rm -rf /)")

    grep_cmd = mock_call.call_args_list[0].args[0]
    # shlex.quote will single-quote the dangerous payload.
    assert grep_cmd == "grep '$(rm -rf /)'"


def test_grep_rejects_positional_path():
    """grep(pattern, path) — positional path raises TypeError.

    Pre-migration callers writing ``grep("Login", True)`` to mean
    ``use_daemon=True`` must not silently get ``path=True``.
    """
    with pytest.raises(TypeError):
        backend.grep("Login", True)  # type: ignore[misc]


def test_grep_rejects_positional_use_daemon():
    """Even the third positional slot is blocked."""
    with pytest.raises(TypeError):
        backend.grep("Login", "/main", "/", True)  # type: ignore[misc]


def test_grep_keyword_use_daemon_still_works():
    """Keyword call against the new signature still type-checks at call time."""
    with patch.object(backend, "_call_execute", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {}
        backend.grep("Login", use_daemon=True)
        assert mock_call.call_args_list == [call("grep Login", True)]


# ── type_text: focus+type pairing and newline injection guard ─────────


@patch.object(backend, "_call_execute", new_callable=AsyncMock)
def test_type_text_emits_focus_then_type_in_one_call(mock_call):
    """type_text builds a single multi-line ``focus … \\ntype …`` execute call."""
    mock_call.return_value = {}

    backend.type_text("search_input", "machine learning")

    assert mock_call.call_args_list == [
        call("focus search_input\ntype 'machine learning'", False),
    ]


def test_type_text_rejects_newline_in_text():
    """``\\n`` in text would inject a new DOMShell command — must raise."""
    with pytest.raises(ValueError, match="newline"):
        backend.type_text("search_input", "line1\nline2")


def test_type_text_rejects_carriage_return_in_text():
    """``\\r`` is just as dangerous as ``\\n`` for DOMShell's line splitter."""
    with pytest.raises(ValueError, match="newline"):
        backend.type_text("search_input", "line1\rline2")


def test_type_text_rejects_newline_in_path():
    """A newline in the path argument also injects — guard both fields."""
    with pytest.raises(ValueError, match="newline"):
        backend.type_text("input\nclick /admin", "anything")


# ── grep: newline guard on rooted multi-step path ─────────────────────


def test_grep_rejects_newline_in_path():
    """Rooted grep interpolates path into a multi-line cd/grep/cd — reject newlines."""
    with pytest.raises(ValueError, match="newline"):
        backend.grep("Login", path="/main\nclick /admin", prev="/")


def test_grep_rejects_newline_in_pattern():
    with pytest.raises(ValueError, match="newline"):
        backend.grep("Login\nclick /admin", path="/main", prev="/")


def test_grep_rejects_newline_in_prev():
    with pytest.raises(ValueError, match="newline"):
        backend.grep("Login", path="/main", prev="/\nclick /admin")


# ── Centralized newline guard in _q ──────────────────────────────────
#
# The per-wrapper _assert_single_line calls above cover type_text and
# rooted grep with field-named error messages. The newline check inside
# _q itself catches the same class of injection for every OTHER wrapper
# that flows user input through the quoting layer (open_url, click, cd,
# cat, unrooted grep, etc.) — without needing a per-call guard at each
# site.


def test_q_rejects_line_feeds():
    with pytest.raises(ValueError, match="[Nn]ewline"):
        backend._q("foo\nbar")


def test_q_rejects_carriage_returns():
    with pytest.raises(ValueError, match="[Nn]ewline"):
        backend._q("foo\rbar")


def test_q_accepts_normal_strings():
    """Plain strings pass through to shlex.quote unchanged."""
    assert backend._q("simple") == "simple"
    assert backend._q("hello world") == "'hello world'"


@patch.object(backend, "_call_execute", new_callable=AsyncMock)
def test_unrooted_grep_pattern_rejects_newlines(mock_call):
    """The unrooted grep path was not field-guarded — _q catches it."""
    mock_call.return_value = {}
    with pytest.raises(ValueError, match="[Nn]ewline"):
        backend.grep("evil\nclick /admin")


@patch.object(backend, "_call_execute", new_callable=AsyncMock)
def test_open_url_rejects_newlines(mock_call):
    """open_url has no per-call _assert_single_line — _q must catch it."""
    mock_call.return_value = {}
    with pytest.raises(ValueError, match="[Nn]ewline"):
        backend.open_url("https://example.com\nclick /admin")


@patch.object(backend, "_call_execute", new_callable=AsyncMock)
def test_click_rejects_newlines(mock_call):
    """click is covered structurally by _q without a per-call guard."""
    mock_call.return_value = {}
    with pytest.raises(ValueError, match="[Nn]ewline"):
        backend.click("/main/button[0]\nclick /admin")


@patch.object(backend, "_call_execute", new_callable=AsyncMock)
def test_cd_rejects_newlines(mock_call):
    mock_call.return_value = {}
    with pytest.raises(ValueError, match="[Nn]ewline"):
        backend.cd("/main\nclick /admin")
