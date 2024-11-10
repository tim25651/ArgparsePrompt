"""Test the prompter with validators and retries.

lib/test/test_argparse.py must work with the new class.
"""

# ruff: noqa: A002
from __future__ import annotations

import argparse
import unittest.mock
from typing import TYPE_CHECKING

import pytest

from argparse_prompt import UNSET, UNSET_T, Prompt, PromptParser, Validator

if TYPE_CHECKING:
    import re
    from collections.abc import Callable, Iterable, Sequence


def get_error_regex(name: str) -> str:
    return rf"^argument --{name}: invalid <Prompt required \(failed: Missing required argument\)> value: ''$"  # noqa: E501


@pytest.fixture
def parser(request: pytest.FixtureRequest) -> PromptParser:
    """Fixture to create a parser with optional retries."""
    retries = getattr(request, "param", 1)
    return PromptParser(retries=retries, exit_on_error=False)


@pytest.mark.parametrize(("interactive"), [(True), (False)])
def test_parser_interactive(interactive: bool) -> None:
    """Test the prompter."""
    parser = PromptParser(interactive=interactive)
    assert parser.parse_args().non_interactive != interactive


@pytest.mark.parametrize(("retries"), [(-1), (0), (1), (2)])
def test_parser_retries(retries: int) -> None:
    """Test the prompter."""
    parser = PromptParser(retries=retries)
    if retries < 1:
        assert parser.retries == 1
    else:
        assert isinstance(PromptParser(retries=retries), PromptParser)


@pytest.fixture
def caperr(
    capsys: pytest.CaptureFixture[str], ansi_escape: re.Pattern[str]
) -> Callable[[], str]:
    """Capture stderr."""

    def _caperr() -> str:
        return ansi_escape.sub("", capsys.readouterr().err)

    return _caperr


def _validate_allow(val: str | None, namespace: argparse.Namespace) -> str | UNSET_T:
    """Fails on empty or "fail" else returns val."""
    del namespace  # unused
    if not val:
        return UNSET
    return val


def _validate(val: str | None, namespace: argparse.Namespace) -> str:
    """Fails on empty or "fail" else returns val."""
    del namespace  # unused
    if not val:
        raise ValueError("must not be empty")
    if val == "fail":
        raise ValueError("must not be fail")
    return val


def _validate_default(val: str | None, namespace: argparse.Namespace) -> str:
    """Returns "default" when val is empty else val."""
    del namespace  # unused
    if not val:
        return "default"
    return val


def _validate_from_other(val: str | None, namespace: argparse.Namespace) -> str:
    """Returns value from .other when val is empty else val."""
    if not val:
        return namespace.other  # type: ignore[no-any-return]
    return val  # pragma: no cover


# INITIALIZATION TESTS
def test_initialization_fail_name() -> None:
    """Test the prompter."""
    prompt = Prompt(argparse.Namespace(), type=str)

    with pytest.raises(ValueError, match="^Name not set$"):
        prompt.name  # noqa: B018


@pytest.mark.parametrize(("prompt"), [(True), (False)])
def test_initialization_fail_default(prompt: bool, parser: PromptParser) -> None:
    """Test the prompter."""
    parser.add_argument("--optional", prompt=prompt, default="default")

    with pytest.raises(
        ValueError, match="^Cannot have a required argument with a default value$"
    ):
        parser.add_argument(
            "--required", prompt=prompt, required=True, default="default"
        )


def assert_output(  # noqa: C901
    args: argparse.Namespace,
    expected: Sequence[tuple[str, str | None]] | dict[str, str | None],
    err: str,
) -> None:
    """Assert the output and stderr."""
    if isinstance(expected, dict):
        expected = list(expected.items())
    errlines = err.splitlines()
    rev_expected = [key for key, _ in reversed(expected)]
    for ix, (errline, (key, value)) in enumerate(zip(errlines, expected)):
        last_appearance = len(expected) - rev_expected.index(key) - 1 == ix
        if value == "unset":
            if last_appearance:
                assert not hasattr(args, key)
            assert errline == f"{key} UNSET"
        elif value == "missing":
            if last_appearance:
                assert not hasattr(args, key)
            assert errline == f"Invalid value () for {key}: Missing required argument"
        elif value and value.startswith("invalid_choice:"):
            if last_appearance:
                assert not hasattr(args, key)
            actual_val = value.split(":")[1]
            assert errline == f"Invalid value ({actual_val}) for {key}: Invalid choice"
        else:
            if last_appearance:
                if value is None:
                    assert getattr(args, key) is None
                else:
                    assert getattr(args, key) == value
            assert errline == f"{key} set to {value} ({type(value).__name__})"


@pytest.mark.parametrize(("required"), [(True), (False)])
def test_required(
    required: bool,
    parser: PromptParser,
    input_patcher: Callable[[Iterable[str]], None],
    caperr: Callable[[], str],
) -> None:
    """Test the prompter."""
    parser.add_argument("--required", type=str, required=required)
    input_patcher([""])
    if required:
        with pytest.raises(argparse.ArgumentError, match=get_error_regex("required")):
            parser.parse_args()
        assert_output(argparse.Namespace(), {"required": "missing"}, caperr())
    else:
        assert_output(parser.parse_args(), {"required": "unset"}, caperr())


@pytest.mark.parametrize(("parser"), [(2)], indirect=True)
@pytest.mark.parametrize(("required"), [(True), (False)])
def test_retry_required_success(
    required: bool,
    parser: PromptParser,
    input_patcher: Callable[[Iterable[str]], None],
    caperr: Callable[[], str],
) -> None:
    parser.add_argument("--required", type=str, required=required)
    if required:
        input_patcher(["", "input"])
        assert_output(
            parser.parse_args(),
            [("required", "missing"), ("required", "input")],
            caperr(),
        )
    else:
        input_patcher([""])
        assert_output(parser.parse_args(), {"required": "unset"}, caperr())


@pytest.mark.parametrize(("parser"), [(2)], indirect=True)
def test_retry_required_fail(
    parser: PromptParser,
    input_patcher: Callable[[Iterable[str]], None],
    caperr: Callable[[], str],
) -> None:
    parser.add_argument("--required", type=str, required=True)
    input_patcher(["", ""])
    with pytest.raises(argparse.ArgumentError, match=get_error_regex("required")):
        parser.parse_args()
    assert_output(
        argparse.Namespace(),
        [("required", "missing"), ("required", "missing")],
        caperr(),
    )


@pytest.mark.parametrize(("input", "expected"), [("", None), ("input", "input")])
def test_optional(
    input: str,
    expected: str | None,
    parser: PromptParser,
    input_patcher: Callable[[Iterable[str]], None],
    caperr: Callable[[], str],
) -> None:
    """Test the prompter."""
    parser.add_argument("--optional", type=str, default=None)
    input_patcher([input])
    assert_output(parser.parse_args(), {"optional": expected}, caperr())


@pytest.mark.parametrize(
    ("func", "success"), [(_validate_allow, None), (_validate, False)]
)
@pytest.mark.parametrize(("required"), [(True), (False)])
def test_validate_empty(
    func: Validator,
    success: bool | None,
    required: bool,
    parser: PromptParser,
    input_patcher: Callable[[Iterable[str]], None],
    caperr: Callable[[], str],
) -> None:
    """Test the prompter."""
    success = success if success is not None else not required
    parser.add_argument("--required", type=func, required=required)

    input_patcher([""])

    if not success:
        with pytest.raises(argparse.ArgumentError, match=get_error_regex("required")):
            parser.parse_args()
        assert_output(argparse.Namespace(), {"required": "missing"}, caperr())
    else:
        assert_output(parser.parse_args(), {"required": "unset"}, caperr())


def test_validate_default(
    parser: PromptParser,
    input_patcher: Callable[[Iterable[str]], None],
    caperr: Callable[[], str],
) -> None:
    parser.add_argument("--required", type=_validate, default=None)
    input_patcher([""])
    assert parser.parse_args().required is None
    assert caperr() == "required set to None (NoneType)\n"


@pytest.mark.parametrize(("input"), [("fail"), ("input")])
@pytest.mark.parametrize(("default"), [(UNSET), ("default")])
def test_validate(
    input: str,
    default: str | UNSET_T,
    parser: PromptParser,
    input_patcher: Callable[[Iterable[str]], None],
    caperr: Callable[[], str],
) -> None:
    """Test the prompter."""
    parser.add_argument("--required", type=_validate, default=default)
    input_patcher([input])

    fail_regex = r"^argument --required: invalid <Prompt required \(failed: must not be fail\)> value: ''$"  # noqa: E501

    if default is UNSET:
        if input == "input":
            assert_output(parser.parse_args(), {"required": "input"}, caperr())
        elif input == "fail":
            with pytest.raises(argparse.ArgumentError, match=fail_regex):
                parser.parse_args()
        else:
            assert_output(parser.parse_args(), {"required": "unset"}, caperr())

    elif input == "fail":
        with pytest.raises(argparse.ArgumentError, match=fail_regex):
            parser.parse_args()
    else:
        assert_output(parser.parse_args(), {"required": input or default}, caperr())


@pytest.mark.parametrize(("input", "expected"), [("", "default"), ("input", "input")])
def test_validate_default_func(
    input: str,
    expected: str,
    parser: PromptParser,
    input_patcher: Callable[[Iterable[str]], None],
    caperr: Callable[[], str],
) -> None:
    """Test the prompter."""
    parser.add_argument("--required", type=_validate_default)
    input_patcher([input])
    assert_output(parser.parse_args(), {"required": expected}, caperr())


@pytest.mark.parametrize(
    ("default", "expected"), [(UNSET, "from_other"), ("default", "default")]
)
def test_validate_from_other_arg(
    default: str | UNSET_T,
    expected: str,
    input_patcher: Callable[[Iterable[str]], None],
    caperr: Callable[[], str],
) -> None:
    """Test the prompter."""
    parser = PromptParser()
    parser.add_argument("--other", default="from_other")
    parser.add_argument("--required", type=_validate_from_other, default=default)
    input_patcher(["", ""])
    assert_output(
        parser.parse_args(), {"other": "from_other", "required": expected}, caperr()
    )


def test_non_prompt_not_required(
    parser: PromptParser, caperr: Callable[[], str]
) -> None:
    """Test the prompter."""
    parser.add_argument("--optional", type=str, prompt=False)
    assert not hasattr(parser.parse_args(), "optional")
    assert caperr() == ""


@pytest.mark.parametrize(
    ("input", "errmsg"),
    [("", "the following arguments are required: --required"), ("input", "")],
)
def test_non_prompt_required(
    input: str,
    errmsg: str,
    parser: PromptParser,
    patch_argv: Callable[[Iterable[str]], None],
    caperr: Callable[[], str],
) -> None:
    """Test the prompter."""
    parser.add_argument("--required", type=str, prompt=False, required=True)

    if input:
        patch_argv(["test.py", "--required", input])
        assert parser.parse_args().required == input
    else:
        patch_argv(["test.py"])
        with pytest.raises((SystemExit, argparse.ArgumentError)) as e:
            parser.parse_args()
        if e.type is argparse.ArgumentError:
            assert errmsg in str(e.value)
            errmsg = ""

    err = caperr()
    if errmsg:
        assert errmsg in err
    else:
        assert err == ""


@pytest.mark.parametrize(("input"), [(""), ("input")])
def test_non_prompt_default(
    input: str,
    parser: PromptParser,
    patch_argv: Callable[[Iterable[str]], None],
    caperr: Callable[[], str],
) -> None:
    if input:
        patch_argv(["test.py", "--optional", input])
        expected = input
    else:
        patch_argv(["test.py"])
        expected = "default"

    parser.add_argument("--optional", prompt=False, default="default")
    assert parser.parse_args().optional == expected
    assert caperr() == ""


@pytest.mark.parametrize(("special_action"), [("help"), ("version")])
def test_special_action_global(
    special_action: str,
    patch_argv: Callable[[Iterable[str]], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_argv(["test.py", f"--{special_action}"])

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version", action="version", version="%(prog)s 1.0")
    with pytest.raises(SystemExit, match="^0$"):
        argparser.parse_args()

    out, err = capsys.readouterr()

    parser = PromptParser()
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")
    with pytest.raises(SystemExit, match="^0$"):
        parser.parse_args()

    out2, err2 = capsys.readouterr()

    if special_action == "help":
        opt_str = "[--non-interactive]"
        opt_help = "--non-interactive  Disable interactive mode.\n"
        assert opt_str in out2
        assert opt_help in out2
        # don't care about the indentation
        out = out.replace(" ", "")
        out2 = out2.replace(opt_str, "").replace(opt_help, "").replace(" ", "")
    assert out == out2
    assert err == err2


@pytest.mark.parametrize(("prompt"), [(True), (False)])
@pytest.mark.parametrize(("type", "expected"), [(str, "input"), (int, 1)])
def test_type(
    prompt: bool,
    type: type,
    expected: str | int,
    parser: PromptParser,
    patch_argv: Callable[[Iterable[str]], None],
) -> None:
    """Test the prompter."""
    parser.add_argument("--optional", type=type, prompt=prompt)
    patch_argv(["test.py", "--optional", str(expected)])
    assert parser.parse_args().optional == expected


@pytest.mark.parametrize(("prompt"), [(True), (False)])
def test_type_fail(
    prompt: bool, parser: PromptParser, patch_argv: Callable[[Iterable[str]], None]
) -> None:
    """Test the prompter."""
    parser.add_argument("--optional", type=int, prompt=prompt)
    patch_argv(["test.py", "--optional", "fail"])
    if prompt:
        errmatch = r"failed: invalid literal for int\(\) with base 10: 'fail'"
    else:
        errmatch = r"invalid int value: 'fail'"
    with pytest.raises(argparse.ArgumentError, match=errmatch):
        parser.parse_args()


@pytest.mark.parametrize(
    ("type", "expected"), [(str, "input"), (int, 1), (None, "input")]
)
@pytest.mark.parametrize(("default"), [(UNSET), (12)])
def test_type_prompt(
    type: type,
    expected: str | int,
    default: str | UNSET_T,
    parser: PromptParser,
    input_patcher: Callable[[Iterable[str]], None],
) -> None:
    parser.add_argument("--optional", type=type, default=default)
    if default is UNSET:
        input_patcher([str(expected)])
        assert parser.parse_args().optional == expected
    else:
        input_patcher([""])
        assert parser.parse_args().optional == default


@unittest.mock.patch("getpass.getpass")
def test_secure(input_mock: unittest.mock.Mock, parser: PromptParser) -> None:
    parser.add_argument("--optional", type=str, secure=True)
    parser.parse_args()
    assert parser.parse_args().optional == str(input_mock.return_value)


@pytest.mark.parametrize(("input", "success"), [("a", True), ("c", False)])
def test_choices(
    input: str,
    success: bool,
    parser: PromptParser,
    input_patcher: Callable[[Iterable[str]], None],
    caperr: Callable[[], str],
) -> None:
    parser.add_argument("--optional", type=str, choices=["a", "b"])
    input_patcher([input])
    fail_regex = r"^argument --optional: invalid <Prompt optional \(failed: Invalid choice\)> value: ''$"  # noqa: E501
    if success:
        assert_output(parser.parse_args(), {"optional": input}, caperr())
    else:
        with pytest.raises(argparse.ArgumentError, match=fail_regex):
            parser.parse_args()
        assert_output(argparse.Namespace(), {"optional": "invalid_choice:c"}, caperr())


@pytest.mark.parametrize(
    ("choices"),
    [
        (["a", "b"]),  # not convertable
        (1),  # not iterable
    ],
)
def test_choices_fail(
    choices: Iterable, parser: PromptParser, caperr: Callable[[], str]
) -> None:
    with pytest.raises(ValueError, match="^Invalid choices:"):
        parser.add_argument("--optional", type=int, choices=choices)
    assert caperr() == ""


def test_choices_validator(parser: PromptParser, caperr: Callable[[], str]) -> None:
    with pytest.raises(ValueError, match="^Cannot have choices with a validator$"):
        parser.add_argument("--optional", type=_validate, choices=["a", "b"])
    assert caperr() == ""


@pytest.mark.parametrize(("interactive"), [(True), (False)])
def test_success_msg(
    interactive: bool,
    patch_argv: Callable[[Iterable[str]], None],
    caperr: Callable[[], str],
) -> None:
    parser = PromptParser(interactive=interactive)
    patch_argv(["test.py", "--optional", "input"])
    parser.add_argument("--optional")
    if interactive:
        assert_output(parser.parse_args(), {"optional": "input"}, caperr())
    else:
        parser.parse_args()
        assert caperr() == ""
