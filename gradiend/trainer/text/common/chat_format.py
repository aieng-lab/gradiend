"""Chat formats of instruction-tuned decoder-only models.

An instruction-tuned ("Instruct") model is trained and evaluated on its own family's chat format: a
system prompt, one user turn that carries the sentence prefix, and the generation prompt that opens the
assistant turn. The model's answer (the word GRADIEND supervises) is the first thing the assistant says.
For reasoning-style templates (DeepSeek-R1, gpt-oss) the prompt ends in a thinking or channel opener, so this is
only an approximation there.

This module is the one place that knows what such a prompt looks like.

:class:`ChatFormat` renders ``(system prompt, user text)`` into a prompt that ends with the generation prompt.
:class:`TemplateChatFormat` renders the tokenizer's own ``chat_template`` (Llama, Qwen, Mistral, Gemma, ...) and is
chosen by :func:`chat_format_for_tokenizer`. A family without a usable template plugs in with
:func:`register_chat_format`, without editing the tokenizer wrapper.

:func:`instruction_name_marker` is the one rule that decides from a model name whether it is an Instruct model
(:func:`legacy_instruction_name_rule` is its predecessor, kept only to read the record of models saved before the
decision was stored with them).
:func:`uses_chat_format` is the one predicate "this tokenizer chat-formats its input". Every data path that must
read plain text (answer words, sentence tokens) goes through :func:`raw_tokenizer`. :func:`chat_prompt_ids` is the
one function that turns a sentence prefix into prompt token ids.

The module has no dependency on the model or dataset code, so those can import it freely.
"""

from __future__ import annotations

import contextlib
import datetime
import re
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from gradiend.trainer.core.arguments import parse_chat_template_date
from gradiend.util.logging import get_logger, suppress_tokenizer_length_warning

logger = get_logger(__name__)

# Stands in for the user text while a template is rendered once to find the text around it. Private-use
# characters, so no template trims, escapes or otherwise rewrites it.
_USER_TEXT_SENTINEL = "gradiend-user-text"

INSTRUCTION_NAME_WORDS = ("instruct", "chat", "it")
"""Whole words of a model's base name that mark an instruction-tuned model (see :func:`instruction_name_marker`)."""


def instruction_name_marker(name: Any) -> Optional[str]:
    """The word that marks ``name`` as an instruction-tuned model, or None for a base model.

    The base name is the last component of the name or path (for a hub id, the part after the organisation).
    It is split at every character that is not a letter or digit, and the model counts as an Instruct model if
    one of the words is ``instruct``, ``chat`` or ``it`` (case-insensitive). A word inside a longer word
    (``fitness``) and a directory above the base name do not count.

    Args:
        name: Model name, hub id or local path.

    Returns:
        The matching word (``"instruct"``, ``"chat"`` or ``"it"``), or None.
    """
    base_name = re.split(r"[\\/]+", str(name).strip().rstrip("\\/"))[-1]
    words = {word for word in re.split(r"[^0-9a-z]+", base_name.lower()) if word}
    for marker in INSTRUCTION_NAME_WORDS:
        if marker in words:
            return marker
    return None


def legacy_instruction_name_rule(name: Any) -> bool:
    """The rule that decided "Instruct model" before :func:`instruction_name_marker`: ``"instruct"`` anywhere in the name.

    This is a compatibility rule. It is used ONLY to read the record of a model that was saved before the
    ``instruction_model`` decision was stored with trained models: such a model was trained under this rule, so it is
    loaded the way it was trained (an old plain-trained ``gemma-2-2b-it`` stays plain; an old Instruct model is
    chat-formatted again, and is then rejected because its label rule is not supported). It never decides for a
    new model: new models use :func:`instruction_name_marker` and store the decision.

    Args:
        name: Model name, hub id or local path, as it was given to the tokenizer loader.

    Returns:
        True if ``"instruct"`` occurs in ``name`` (any case, also in the organisation or a parent directory).
    """
    return "instruct" in str(name).lower()


DEFAULT_CHAT_TEMPLATE_DATE = "2024-07-26"
"""The date a chat template that writes the current date sees when no ``chat_template_date`` is given.

It is the date the Llama 3.1 and 3.2 templates themselves fall back to. This is the one place that knows it; a trained
model stores the date it was trained with, so changing this constant never changes an existing model.
"""


_CLOCK_LOCK = threading.Lock()


class ChatFormat(ABC):
    """How one model family wraps a user text into a chat prompt.

    A format is described by its :meth:`frame`: the text before and after the user text. The prompt is
    ``prefix + user_text + suffix``, so the user text enters the prompt verbatim and the prompt always
    ends with the generation prompt of the assistant turn.
    """

    @abstractmethod
    def frame(self, system_prompt: Optional[str]) -> Tuple[str, str]:
        """Return ``(prefix, suffix)``: the prompt text before and after the user text.

        Args:
            system_prompt: System prompt text; an empty value renders no system turn.
        """

    def render(self, system_prompt: Optional[str], user_text: str) -> str:
        """Return the chat prompt for ``user_text``, ending with the generation prompt.

        Args:
            system_prompt: System prompt text; an empty value renders no system turn.
            user_text: Text of the user turn, inserted verbatim.
        """
        prefix, suffix = self.frame(system_prompt)
        return f"{prefix}{user_text}{suffix}"

    def configure(self, chat_template_date: Optional[str] = None, *, recorded: bool = False) -> None:
        """Apply the run-level settings of the format. The default format has none and ignores them.

        Args:
            chat_template_date: The fixed date (ISO ``YYYY-MM-DD``) a format that writes the date into its prompt uses;
                None selects :data:`DEFAULT_CHAT_TEMPLATE_DATE`.
            recorded: True if the date was read from the record of a trained model (only used to word log lines).
        """


class TemplateChatFormat(ChatFormat):
    """The chat format of a tokenizer's own ``chat_template`` (Llama, Qwen, Mistral, Gemma, ...).

    The template is rendered once per system prompt, with a placeholder for the user text, to find the
    text around that placeholder. The system prompt is whitespace-normalised to one line. A template that
    has no system role gets the system prompt at the start of the user turn instead. The user text is
    inserted verbatim, which differs from calling ``apply_chat_template`` only for templates that strip or
    rewrite message content.

    Some templates write the current date into the prompt by calling ``strftime_now`` (Llama 3.1 and later,
    gpt-oss). Training rows must not change with the day they are built, so the format hands the template its
    own ``strftime_now``, which returns a fixed date instead of the system clock: the one given to
    :meth:`configure`, else :data:`DEFAULT_CHAT_TEMPLATE_DATE`. The first time a template asks for the date, one INFO
    line says which date it got. A template that never calls ``strftime_now`` is not affected.

    Attributes:
        inserts_date: True once a rendering of the template has asked for the current time.
    """

    def __init__(self, tokenizer: Any, chat_template_date: Optional[str] = None) -> None:
        self._tokenizer = tokenizer
        self._frames: Dict[Tuple[str, datetime.date], Tuple[str, str]] = {}
        self._date: Optional[datetime.date] = None
        self._date_recorded = False
        self._logged_date: Optional[datetime.date] = None
        self.inserts_date = False
        if chat_template_date is not None:
            self.configure(chat_template_date)

    @classmethod
    def supports(cls, tokenizer: Any) -> bool:
        """Whether ``tokenizer`` ships a chat template."""
        return bool(getattr(tokenizer, "chat_template", None))

    def _effective_date(self) -> datetime.date:
        """The date a template that asks for the current date sees: the configured one, else the default."""
        return self._date if self._date is not None else parse_chat_template_date(DEFAULT_CHAT_TEMPLATE_DATE)

    @property
    def chat_template_date(self) -> str:
        """The fixed date (ISO ``YYYY-MM-DD``) a template sees as the current date: the configured one, else the default."""
        return self._effective_date().isoformat()

    def configure(self, chat_template_date: Optional[str] = None, *, recorded: bool = False) -> None:
        """Set the fixed date a template that asks for the current time sees.

        Args:
            chat_template_date: ISO date ``YYYY-MM-DD``, or None for :data:`DEFAULT_CHAT_TEMPLATE_DATE`.
            recorded: True if the date was read from the record of a trained model; the log line then says so.

        Raises:
            ValueError: If ``chat_template_date`` is not a valid ISO date.
        """
        self._date = None if chat_template_date is None else parse_chat_template_date(chat_template_date)
        self._date_recorded = recorded and self._date is not None

    def frame(self, system_prompt: Optional[str]) -> Tuple[str, str]:
        system = " ".join(str(system_prompt).split()) if system_prompt else ""
        key = (system, self._effective_date())
        if key not in self._frames:
            self._frames[key] = self._render_frame(system)
        return self._frames[key]

    def _clock_read(self) -> datetime.datetime:
        """The time a template sees when it asks for the current time: midnight of the fixed date."""
        self.inserts_date = True
        date = self._effective_date()
        if self._logged_date != date:
            self._logged_date = date
            hint = " Set TrainingArguments(chat_template_date='YYYY-MM-DD') to use another date."
            if self._date is None:
                source = "the default date"
            elif self._date_recorded:
                source, hint = "recorded with the model", ""
            else:
                source = "the configured date"
            logger.info(
                "The chat template of %r inserts the current date into the prompt; using the fixed date %s (%s) so the "
                "rows are reproducible.%s",
                _tokenizer_name(self._tokenizer), date.isoformat(), source, hint,
            )
        return datetime.datetime(date.year, date.month, date.day)

    @contextlib.contextmanager
    def _fixed_clock(self) -> Iterator[None]:
        """Make the ``strftime_now`` that transformers gives every template read the fixed date.

        transformers defines ``strftime_now`` itself (``datetime.now().strftime(...)`` in its chat-template module).
        Passing our own function as a keyword argument works on transformers 4 but not on 5, which turns template
        keyword arguments into strings. So the clock that function reads is swapped for the duration of the
        rendering instead; this behaves the same on every transformers version that has ``strftime_now``.
        """
        from transformers.utils import chat_template_utils

        owner = self

        class _FixedDatetime(datetime.datetime):
            @classmethod
            def now(cls, tz=None):  # noqa: D102 - the clock the template reads
                return owner._clock_read()

        if not hasattr(chat_template_utils, "datetime"):
            raise RuntimeError(
                "This transformers version does not expose the clock of its chat-template module, so the date a "
                "chat template inserts cannot be fixed. Pin the date in the tokenizer's template or use another "
                "transformers version."
            )
        with _CLOCK_LOCK:  # the swap is process-wide, so two renderings must not overlap
            original = chat_template_utils.datetime
            chat_template_utils.datetime = _FixedDatetime
            try:
                yield
            finally:
                chat_template_utils.datetime = original

    def _render_frame(self, system: str) -> Tuple[str, str]:
        sentinel = _USER_TEXT_SENTINEL
        if system:
            attempts = [
                [{"role": "system", "content": system}, {"role": "user", "content": sentinel}],
                [{"role": "user", "content": f"{system}\n\n{sentinel}"}],
            ]
        else:
            attempts = [[{"role": "user", "content": sentinel}]]
        problems: List[str] = []
        for messages in attempts:
            try:
                with self._fixed_clock():
                    rendered = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as exc:  # noqa: BLE001 - e.g. a template without a system role
                problems.append(f"{[m['role'] for m in messages]}: {type(exc).__name__}: {exc}")
                continue
            if not isinstance(rendered, str) or rendered.count(sentinel) != 1:
                problems.append(f"{[m['role'] for m in messages]}: the template does not render the user turn verbatim")
                continue
            if system and system not in rendered:
                problems.append(f"{[m['role'] for m in messages]}: the template drops the system prompt")
                continue
            prefix, _, suffix = rendered.partition(sentinel)
            return prefix, suffix
        raise ValueError(
            f"The chat template of {_tokenizer_name(self._tokenizer)!r} cannot frame a system prompt and a user turn "
            f"({'; '.join(problems)})."
        )


ChatFormatFactory = Callable[[Any], Optional[ChatFormat]]
"""``factory(tokenizer)`` returns a format, or None for a tokenizer it does not handle."""

_CHAT_FORMAT_FACTORIES: List[ChatFormatFactory] = []


def register_chat_format(factory: ChatFormatFactory, *, first: bool = True) -> ChatFormatFactory:
    """Register a chat-format factory for a model family, so :func:`chat_format_for_tokenizer` can pick it.

    Use it for a family whose tokenizer has no usable ``chat_template``, or to replace the template rendering.

    Args:
        factory: ``factory(tokenizer)`` returning a :class:`ChatFormat`, or ``None`` for a tokenizer it does not handle.
        first: Ask this factory before the ones registered earlier (default), so it can override a built-in.

    Returns:
        ``factory``, so this can be used as a decorator.
    """
    if first:
        _CHAT_FORMAT_FACTORIES.insert(0, factory)
    else:
        _CHAT_FORMAT_FACTORIES.append(factory)
    return factory


def find_chat_format(tokenizer: Any) -> Optional[ChatFormat]:
    """The chat format of ``tokenizer``'s own model family, or None if it has none.

    Registered factories are asked first. Otherwise a tokenizer with a chat template gets
    :class:`TemplateChatFormat`. This is the one place that decides whether a chat format exists;
    :func:`chat_format_for_tokenizer` raises where this returns None.

    Args:
        tokenizer: The raw (unwrapped) tokenizer.
    """
    for factory in _CHAT_FORMAT_FACTORIES:
        chat_format = factory(tokenizer)
        if chat_format is not None:
            return chat_format
    if TemplateChatFormat.supports(tokenizer):
        return TemplateChatFormat(tokenizer)
    return None


def chat_format_for_tokenizer(tokenizer: Any) -> ChatFormat:
    """Choose the chat format of ``tokenizer``'s own model family (see :func:`find_chat_format`).

    Args:
        tokenizer: The raw (unwrapped) tokenizer.

    Raises:
        ValueError: If no format applies. A foreign family's chat tokens are never encoded as plain text.
    """
    chat_format = find_chat_format(tokenizer)
    if chat_format is not None:
        return chat_format
    raise ValueError(
        f"No chat format for {_tokenizer_name(tokenizer)!r}: its tokenizer has no chat_template, so an Instruct "
        "model of this family cannot be chat-formatted. Give the tokenizer a chat_template, register a format with "
        "gradiend.trainer.text.common.chat_format.register_chat_format, or set "
        "TrainingArguments(instruction_model=False) to read plain text."
    )


def uses_chat_format(tokenizer: Any) -> bool:
    """Whether ``tokenizer`` chat-formats its input (an Instruct tokenizer wrapper).

    This is the single predicate for "rows of this tokenizer are chat prompts"; code that unwraps a tokenizer
    to read plain text must ask it first.
    """
    return isinstance(getattr(tokenizer, "chat_format", None), ChatFormat)


def raw_tokenizer(tokenizer: Any) -> Any:
    """The tokenizer without chat formatting: ``tokenizer.tokenizer`` for a chat tokenizer, else ``tokenizer``."""
    return tokenizer.tokenizer if uses_chat_format(tokenizer) else tokenizer


def configure_chat_tokenizer(tokenizer: Any, chat_template_date: Optional[str] = None) -> Optional[str]:
    """Apply the run-level settings to a chat tokenizer and render its frame once, so a problem shows up now.

    A plain tokenizer is left alone. For a chat tokenizer the date (if given) is set on its chat format, and the
    frame of the system prompt is rendered eagerly: a template that cannot frame a system prompt and a user turn
    fails here, before anything is trained, not at the first row; and the format learns whether the template asks
    for the current date (it then logs which fixed date it uses).

    Args:
        tokenizer: Any tokenizer; only a chat tokenizer (see :func:`uses_chat_format`) is configured.
        chat_template_date: Fixed date (ISO ``YYYY-MM-DD``) for templates that write the date, or None to keep the
            one the format already has.

    Returns:
        The ISO date the template inserts into its prompt (the configured date, else
        :data:`DEFAULT_CHAT_TEMPLATE_DATE`), or None if ``tokenizer`` is plain or its template never asks for the date.

    Raises:
        ValueError: If ``chat_template_date`` is invalid, or the template cannot frame a system prompt and a user turn.
    """
    if not uses_chat_format(tokenizer):
        return None
    chat_format = tokenizer.chat_format
    if chat_template_date is not None:
        chat_format.configure(chat_template_date)
    chat_format.frame(tokenizer.system_prompt)
    if getattr(chat_format, "inserts_date", False):
        return getattr(chat_format, "chat_template_date", None)
    return None


def chat_prompt(tokenizer: Any, user_text: str) -> str:
    """The chat prompt text for ``user_text``, ending with the generation prompt.

    Args:
        tokenizer: A chat tokenizer (see :func:`uses_chat_format`).
        user_text: Text of the user turn, inserted verbatim.
    """
    return tokenizer.chat_format.render(tokenizer.system_prompt, str(user_text))


def chat_prompt_ids(tokenizer: Any, user_text: str, *, max_length: Optional[int] = None) -> List[int]:
    """Token ids of the chat prompt for ``user_text``, ending with the generation prompt.

    The prompt is tokenized as one text without extra special tokens (the format carries the beginning-of-text
    token of its family). If ``max_length`` is given and the prompt is longer, words are dropped from the LEFT of
    ``user_text`` until it fits, so the system prompt and the generation prompt always stay.

    Args:
        tokenizer: A chat tokenizer (see :func:`uses_chat_format`).
        user_text: Text of the user turn, inserted verbatim.
        max_length: Maximum number of prompt tokens, or None for no limit.
    """
    raw = raw_tokenizer(tokenizer)

    def encode(text: str) -> List[int]:
        with suppress_tokenizer_length_warning():
            return list(raw(chat_prompt(tokenizer, text), add_special_tokens=False)["input_ids"])

    ids = encode(user_text)
    if max_length is None or len(ids) <= max_length:
        return ids
    words = str(user_text).split(" ")
    lowest, highest = 0, len(words)  # number of words dropped from the left
    while lowest < highest:
        middle = (lowest + highest) // 2
        if len(encode(" ".join(words[middle:]))) <= max_length:
            highest = middle
        else:
            lowest = middle + 1
    return encode(" ".join(words[lowest:]))


def chat_user_text_start(tokenizer: Any) -> int:
    """Number of prompt tokens before the user text: the system prompt and the user turn's header.

    Trailing whitespace of the frame (the newlines after the header) is not counted, so the user text region
    starts at the first token after the role header.

    Args:
        tokenizer: A chat tokenizer (see :func:`uses_chat_format`).
    """
    prefix, _ = tokenizer.chat_format.frame(tokenizer.system_prompt)
    with suppress_tokenizer_length_warning():
        return len(raw_tokenizer(tokenizer)(prefix.rstrip(), add_special_tokens=False)["input_ids"])


def rebase_offsets_to_user_text(offsets: Sequence[Sequence[int]], prefix_length: int, user_text_length: int) -> List[Tuple[int, int]]:
    """Map character offsets in a chat prompt to offsets in the user text.

    A token of the chat frame (system prompt, headers, generation prompt) gets an empty span at the border
    of the user text, so consumers that skip empty spans see only the user text's own tokens.

    Args:
        offsets: ``(start, end)`` character offsets in the rendered prompt, one pair per token.
        prefix_length: Length of the prompt text before the user text.
        user_text_length: Length of the user text.
    """
    rebased: List[Tuple[int, int]] = []
    for start, end in offsets:
        new_start = min(max(int(start) - prefix_length, 0), user_text_length)
        new_end = min(max(int(end) - prefix_length, 0), user_text_length)
        rebased.append((new_start, new_end))
    return rebased


def _tokenizer_name(tokenizer: Any) -> str:
    return str(getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__)


def require_first_differing_token_for_chat(tokenizer: Any, label_token_protocol: str) -> None:
    """Fail early if a chat tokenizer is combined with a label-token rule other than ``first_differing_token``.

    This is the one check where a tokenizer and ``TrainingArguments.label_token_protocol`` meet: dataset
    construction, the trainer's model setup (fresh and loaded models), ``create_inputs`` and row-wise decoder
    evaluation all call it. A plain tokenizer is never rejected.

    Args:
        tokenizer: Any tokenizer; only a chat tokenizer (see :func:`uses_chat_format`) is checked.
        label_token_protocol: The rule in effect.

    Raises:
        ValueError: If ``tokenizer`` chat-formats its input and the rule is not ``"first_differing_token"``.
    """
    if uses_chat_format(tokenizer) and label_token_protocol != "first_differing_token":
        raise ValueError(
            f"Instruct models (chat tokenizer of {_tokenizer_name(raw_tokenizer(tokenizer))!r}) need "
            f"label_token_protocol='first_differing_token', got {label_token_protocol!r}: the answer opens the "
            "assistant turn after the generation prompt and only that rule supervises it (the old Instruct rule "
            "labelled the last prompt token and is not supported; retrain a model trained with it)."
        )


__all__ = [
    "DEFAULT_CHAT_TEMPLATE_DATE",
    "INSTRUCTION_NAME_WORDS",
    "ChatFormat",
    "ChatFormatFactory",
    "TemplateChatFormat",
    "chat_format_for_tokenizer",
    "chat_prompt",
    "chat_prompt_ids",
    "chat_user_text_start",
    "find_chat_format",
    "configure_chat_tokenizer",
    "instruction_name_marker",
    "legacy_instruction_name_rule",
    "raw_tokenizer",
    "rebase_offsets_to_user_text",
    "register_chat_format",
    "require_first_differing_token_for_chat",
    "uses_chat_format",
]
