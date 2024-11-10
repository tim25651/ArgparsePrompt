"""A module for prompting the user for missing command line arguments."""

# forked from https://github.com/multimeric/ArgparsePrompt (v0.0.5)
# ruff: noqa: T201, E501
from __future__ import annotations

import argparse
import getpass
import inspect
import os
import sys
from contextlib import redirect_stdout
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeAlias, TypeGuard

import termcolor
from typing_extensions import TypeVar, override

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

Colors: TypeAlias = Literal["blue", "red", "green", "yellow"]


def _colored(text: str, color: Colors | None = None) -> str:
    """Wrapper for termcolor.colored with force_color=True."""
    return termcolor.colored(text, color, force_color=True)


def cerrprint(text: str, color: Colors | None = None) -> None:
    """Print text to stderr with the given color."""
    print(_colored(text, color), file=sys.stderr)


class Validator(Protocol, Generic[T_co]):
    """A protocol for validating input. Argument names must match.

    Args:
        val (str | None): The value to validate.
        namespace (argparse.Namespace): The namespace to validate against.

    Returns:
        T_co | None: The validated value or None.

    Raises:
        ValueError: If the value is invalid or None is provided for a required argument.
    """

    def __call__(
        self, val: str | None, namespace: argparse.Namespace
    ) -> T_co | UNSET_T:  # pragma: no cover
        """Validate the input with the given namespace."""
        ...


class _UnsetT(Enum):
    """Container enum for the UNSET Literal."""

    UNSET = 0

    @override
    def __str__(self) -> str:
        return self.name


class _RequiredT(Enum):
    """Container enum for the REQUIRED Literal."""

    REQUIRED = 0

    @override
    def __str__(self) -> str:
        return self.name


UNSET = _UnsetT.UNSET
UNSET_T: TypeAlias = Literal[_UnsetT.UNSET]
REQUIRED = _RequiredT.REQUIRED
REQUIRED_T: TypeAlias = Literal[_RequiredT.REQUIRED]


class PromptParser(argparse.ArgumentParser):
    """Extends ArgumentParser to allow any unspecified arguments to be input dynamically on the command line."""

    def __init__(
        self, retries: int = 1, interactive: bool = True, *args: Any, **kwargs: Any
    ) -> None:
        """Creates a new prompt parser.

        Args:
            retries: The number of times to retry prompting the user for a value.
                if the user provides an invalid value or no value (for a required argument).
            interactive: If true, the parser will prompt the user for any missing arguments
                unless --non-interactive is provided. If false, the parser will not prompt the user
                unless --interactive is provided.
            args: Positional arguments to pass to the `argparse.ArgumentParser` constructor.
            kwargs: Keyword arguments to pass to the `argparse.ArgumentParser` constructor.
        """
        self.retries = max(retries, 1)

        self.namespace = argparse.Namespace()
        super().__init__(*args, **kwargs)
        if interactive:
            self.add_argument(
                "--non-interactive",
                action="store_true",
                help="Disable interactive mode.",
                default=False,
            )
        else:
            self.add_argument(
                "--interactive",
                action="store_false",
                dest="non_interactive",
                help="Enable interactive mode.",
                default=True,
            )

    @override
    def parse_args(  # type: ignore[override]
        self, args: Sequence[str] | None = None, namespace: None = None
    ) -> argparse.Namespace:
        """Parses the arguments, prompting the user for any unspecified arguments.

        Args:
            args: The arguments to parse. If None, defaults to `sys.argv`.
            namespace: Unused. This is here to satisfy the superclass method signature.
                `self.namespace` is used instead.
        """
        super().parse_args(args, namespace=self.namespace)
        for key, value in list(self.namespace.__dict__.items()):
            if value is UNSET:
                delattr(self.namespace, key)
        return self.namespace

    @override
    def add_argument(  # type: ignore[override]
        self,
        *args: Any,
        prompt: bool = True,
        # until default Generic is supported
        type: Validator[T] | Validator[T | None] | Callable[[str], T] | None = None,
        default: T | None | UNSET_T = UNSET,
        choices: Iterable[T] | None = None,
        required: bool = False,
        secure: bool = False,
        **kwargs: Any,
    ) -> None:
        """For all unlisted arguments, refer to the parent class.

        Args:
            *args: Positional arguments to pass to the `argparse.ArgumentParser.add_argument` method.
            prompt: If False, the prompting is disabled independent of the `ARGPARSE_PROMPT_AUTO` or
                `--non-interactive` flags.
            type: The type of the argument. If a Validator is provided, it will be called with the value
                and the namespace to validate the input.
            default: The default value of the argument. If not provided, default is either `self.validator(None)`
                or `UNSET` if `self.validator` is not set.
            choices: A list of choices which must match the type of the argument.
            required: True if the argument is required.
            secure: True if the input should not be shown on the command line while it's input.
            **kwargs: Keyword arguments to pass to the `argparse.ArgumentParser.add_argument` method.

        Raises:
            ValueError: If `default` is provided for a required argument.
        """
        if required and default is not UNSET:
            raise ValueError("Cannot have a required argument with a default value")

        special_actions = {
            "help",
            "store_true",
            "store_false",
            "version",
            "parsers",
            "count",
            "extend",
            "append",
            "append_const",
        }
        if prompt and kwargs.get("action") not in special_actions:
            # Wrap the Prompt type around the type the user wants
            help = kwargs.get("help")  # noqa: A001
            prompt_type = Prompt(
                self.namespace,
                help,
                type,
                default,
                choices,
                required,
                secure,
                self.retries,
            )

            # Delegate to the parent class. Default must be '' in order to get the type function to be called
            action = super().add_argument(*args, type=prompt_type, default="", **kwargs)

            # Set the argument name, now that the parser has parsed it
            prompt_type.name = action.dest
            return

        if not required or default is not UNSET:
            kwargs["default"] = default

        for key, value in (("type", type), ("choices", choices)):
            if value is not None:
                kwargs[key] = value  # noqa: PERF403,RUF100

        if kwargs.get("action") in {"help", "version"}:
            super().add_argument(*args, **kwargs)
        else:
            super().add_argument(*args, required=required, **kwargs)


def _is_validator(
    type: Validator[T] | Validator[T | None] | Callable[[str], T],  # noqa: A002
) -> TypeGuard[Validator[T] | Validator[T | None]]:
    """Check if the type is a Validator (takes a `val` and `namespace` argument).

    Args:
        type: The type or validator to check.

    Returns:
        True if the type is a Validator, False otherwise.
    """
    try:
        signature = inspect.signature(type)
        return tuple(signature.parameters) == ("val", "namespace")
    except ValueError:
        return False


class Prompt(Generic[T]):
    """A class the pretends to be a function so that it can be used as the 'type' argument for the ArgumentParser."""

    def __init__(
        self,
        namespace: argparse.Namespace,
        help: str | None = None,  # noqa: A002
        type: Validator[T] | Validator[T | None] | Callable[[str], T] | None = None,  # noqa: A002
        default: T | None | UNSET_T = UNSET,
        choices: Iterable[T] | None = None,
        required: bool = False,
        secure: bool = False,
        retries: int = 1,
    ) -> None:
        """Creates a new prompt validator.

        Args:
            namespace: The namespace to validate against.
            help: The help string to give the user when prompting. Defaults to None.
            type: The validation function to use on the prompted data or the type of the argument.
                If None, the type will be str.
            default: The default value of the argument. Defaults to UNSET.
            choices: A list of choices which must match the type of the argument. Defaults to None.
            required: True if the argument is required. Defaults to False.
            secure: True if the input should not be shown on the command line while it's input.
                Defaults to False.
            retries: The number of times to retry prompting the user for a value. Defaults to 1.
        """
        self.namespace = namespace
        self.help = help

        self.required = required
        self.secure = secure
        self.retries = retries

        self._name: str | None = None
        self._global_non_interactive = bool(os.getenv("ARGPARSE_PROMPT_AUTO"))

        self._set_type(type, default, choices)

        self.err: str | None = None

    def _set_choices(self, choices: Iterable[Any] | None) -> list[T] | None:
        """Check if the choices match the type of the argument."""
        if choices is None:
            return None
        if self.type is None:
            raise RuntimeError("Type is not set")  # pragma: no cover
        try:
            return [self.type(choice) for choice in choices]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid choices: {e}") from e

    def _set_type(
        self,
        type: Validator[T] | Validator[T | None] | Callable[[str], T] | None,  # noqa: A002
        default: T | None | UNSET_T,
        choices: Iterable[Any] | None,
    ) -> None:
        """Set the type, validator, and choices for the prompt.

        If validator is provided, type and choices have to be None.
        If type is provided, validator has to be None.
        """
        self.type: Callable[[str], T] | None
        self.validator: (
            Callable[[str | None], T] | Callable[[str | None], T | UNSET_T] | None
        )
        if type is None:

            def _identity(x: str) -> str:
                return x

            self.type = _identity  # type: ignore[assignment]
            self.validator = None
            self.default = default
            self.choices = self._set_choices(choices)

        elif _is_validator(type):
            if choices is not None:
                raise ValueError("Cannot have choices with a validator")
            self.choices = None

            def _validator(val: str | None) -> T | UNSET_T:
                return type(val, self.namespace)  # type: ignore[return-value]

            self.type = None
            self.validator = _validator
            if default is not UNSET:
                self.default = default
            # else default will be set in __call__
        else:
            self.type = type  # type: ignore[assignment]
            self.validator = None
            self.default = default
            self.choices = self._set_choices(choices)

    @override
    def __repr__(self) -> str:
        name = self._name or "Undefined"
        err = f" (failed: {self.err})" if self.err else ""
        return f"<Prompt {name}{err}>"

    @property
    def name(self) -> str:
        """The name of the argument."""
        if self._name is None:
            raise ValueError("Name not set")
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def _get_default(self) -> T | UNSET_T | REQUIRED_T | None:
        """Get the default value for the argument.

        Tries to use the validator if no default is set.
        If this fails, the argument is required.
        """
        if not hasattr(self, "default"):
            if self.validator is None:  # pragma: no cover
                raise RuntimeError("Validator is not set")
            try:
                return self.validator(None)
            except ValueError:
                self.required = True
                return REQUIRED
        if self.required:
            return REQUIRED
        return self.default

    def _get_prompt(self, default: T | UNSET_T | REQUIRED_T | None) -> str:
        """Build the prompt string for the user.

        ```
        name [choice, choice, ...]: help
        > (default) ...
        ```
        """
        yellow_name = _colored(self.name, "yellow")
        help_str = f": {self.help}" if self.help else ""
        default_str = f"({default}) " if default else ""
        choices_str = (
            f" [{', '.join(str(x) for x in self.choices)}]" if self.choices else ""
        )
        yellow_choices = _colored(choices_str, "yellow") if self.choices else ""
        return f"{yellow_name}{yellow_choices}{help_str}\n> {default_str}"

    def _is_interactive(self) -> bool:
        """Check if either the environment or the --non-interactive flag is set."""
        _local_non_interactive = getattr(self.namespace, "non_interactive", False)
        return not self._global_non_interactive and not _local_non_interactive

    def _call(self, val: str) -> T | UNSET_T | str | None:
        """Single try to prompt the user for a value."""
        # find out if we are in interactive mode
        interactive = self._is_interactive()
        default = self._get_default()

        newval: T | str | None = None
        try:
            # If the user provided no value for this argument, prompt them for it
            if val == "" and interactive:
                prompt = self._get_prompt(default)

                if self.secure:
                    unparsed = getpass.getpass(prompt=prompt, stream=sys.stderr)
                else:
                    with redirect_stdout(sys.stderr):
                        unparsed = input(prompt)

                # If they just hit enter, they want the default value
                newval = unparsed

                # According to the argparse docs, if the default is a string we should convert it, but otherwise
                # we return it verbatim: https://docs.python.org/3/library/argparse.html#default
                # finalval = self.type(newval) if isinstance(newval, str) else
                # newval
            else:
                newval = val

            if not newval:
                if self.required or default is REQUIRED:
                    raise ValueError("Missing required argument")  # noqa: TRY301
                return default  # T | None

            if self.validator is not None:
                return self.validator(newval)

            if self.type is None:
                raise RuntimeError("Type is not set")  # pragma: no cover

            finalval = self.type(newval)

            if self.choices is not None and finalval not in self.choices:
                raise ValueError("Invalid choice")  # noqa: TRY301

            return finalval

        except ValueError as e:
            cerrprint(f"Invalid value ({newval}) for {self.name}: {e}", "red")
            raise

    def __call__(self, val: str) -> T | UNSET_T | str | None:
        """Prompts the user for a value if one is not provided.

        If it fails, it will retry up to `self.retries` times.
        If non-interactive, it will only try once.
        """
        interactive = self._is_interactive()

        # do not retry if non-interactive
        retries = self.retries if interactive else 1

        for retry in range(retries):
            try:
                retval = self._call(val)
                if interactive:
                    self._success(retval)
                return retval
            except ValueError as e:  # noqa: PERF203
                if retry < retries - 1:
                    continue
                self.err = str(e)
                raise ValueError(self.err) from e
        raise RuntimeError("Unreachable code")  # pragma: no cover

    def _success(self, finalval: T | UNSET_T | str | None) -> None:
        """Print a success message.

        `name set to value (type)` or `name UNSET` if no value is set.
        """
        unmangled_name = self.name.replace("_", "-")
        yellow_name = _colored(unmangled_name, "yellow")
        green_val = _colored(str(finalval), "green")
        if finalval is UNSET:
            cerrprint(f"{yellow_name} {green_val}")
        else:
            type_name = type(finalval).__name__
            cerrprint(f"{yellow_name} set to {green_val} ({type_name})")


__all__ = ["PromptParser", "Prompt", "Validator", "UNSET", "UNSET_T"]
