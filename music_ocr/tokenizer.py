from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import pydantic


@runtime_checkable
class Tokenizer(Protocol):
    vocab_size: int
    pad_token_id: int | None
    eos_token_id: int | None
    bos_token_id: int | None

    def get_token_id(self, token: str) -> int: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> Self: ...

    def encode(self, text: str) -> list[int]: ...
    def encode_batch(self, texts: list[str], pad: bool) -> list[list[int]]: ...
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True, stop_on_eos: bool = True) -> str: ...
    def decode_batch(
        self, token_ids: list[list[int]], skip_special_tokens: bool = True, stop_on_eos: bool = True
    ) -> list[str]: ...


class SpaceTokenizerConfig(pydantic.BaseModel, extra="forbid"):
    special_tokens: list[str]
    bos_token: str | None
    pad_token: str
    eos_token: str


class SpaceTokenizerState(pydantic.BaseModel, extra="forbid"):
    config: SpaceTokenizerConfig
    id_to_token: dict[int, str]
    token_to_id: dict[str, int]
    special_token_ids: set[int]
    bos_token_id: int | None
    pad_token_id: int
    eos_token_id: int


class SpaceTokenizer:
    __slots__ = ("state",)

    def __init__(self, vocab: list[str], config: SpaceTokenizerConfig) -> None:
        tokens = list(config.special_tokens) + list(vocab)

        if len(tokens) != len(set(tokens)):
            raise ValueError("Tokenizer tokens must be unique")
        for key in config.special_tokens:
            if key not in tokens:
                raise ValueError(f"Special token '{key}' must be present in the tokenizer state")
        if config.bos_token is not None and config.bos_token not in config.special_tokens:
            raise ValueError(f"Bos token '{config.bos_token}' must be a special token")
        if config.pad_token is not None and config.pad_token not in config.special_tokens:
            raise ValueError(f"Pad token '{config.pad_token}' must be a special token")
        if config.eos_token is not None and config.eos_token not in config.special_tokens:
            raise ValueError(f"Eos token '{config.eos_token}' must be a special token")

        id_to_token = {idx: token for idx, token in enumerate(tokens)}
        token_to_id = {token: idx for idx, token in enumerate(tokens)}

        self.state = SpaceTokenizerState(
            config=config,
            id_to_token=id_to_token,
            token_to_id=token_to_id,
            special_token_ids={token_to_id[token] for token in config.special_tokens},
            bos_token_id=token_to_id[config.bos_token] if config.bos_token is not None else None,
            pad_token_id=token_to_id[config.pad_token],
            eos_token_id=token_to_id[config.eos_token],
        )

    @property
    def vocab_size(self) -> int:
        return len(self.state.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.state.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.state.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.state.bos_token_id

    def get_token_id(self, token: str) -> int:
        try:
            return self.state.token_to_id[token]
        except KeyError:
            raise KeyError(f"Token {token!r} not found in tokenizer vocabulary")

    @classmethod
    def from_state(cls, state: SpaceTokenizerState) -> Self:
        tokenizer = cls.__new__(cls)
        tokenizer.state = state
        return tokenizer

    def save(self, path: str | Path) -> None:
        state = self.state.model_dump_json(indent=2)
        Path(path).write_text(state, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        content = Path(path).read_text(encoding="utf-8")
        state = SpaceTokenizerState.model_validate_json(content)
        return cls.from_state(state)

    def encode(self, text: str) -> list[int]:
        tokens = text.split(" ")
        token_ids = []
        for token in tokens:
            token_id = self.state.token_to_id[token]
            if token_id in self.state.special_token_ids:
                raise ValueError(f"Token {token} cannot be a special token")
            token_ids.append(token_id)
        return token_ids

    def encode_batch(self, texts: list[str], pad: bool) -> list[list[int]]:
        token_ids_batch = [self.encode(text) for text in texts]
        if pad:
            max_len = max(len(token_ids) for token_ids in token_ids_batch)
            if self.state.pad_token_id is None:
                raise ValueError("pad_token_id must be defined for padding")
            for i in range(len(token_ids_batch)):
                token_ids_batch[i] += [self.state.pad_token_id] * (max_len - len(token_ids_batch[i]))
        return token_ids_batch

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True, stop_on_eos: bool = True) -> str:
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in self.state.special_token_ids:
                continue
            token = self.state.id_to_token[token_id]
            tokens.append(token)
            if stop_on_eos and token_id == self.state.eos_token_id:
                break
        return " ".join(tokens)

    def decode_batch(
        self,
        token_ids: list[list[int]],
        skip_special_tokens: bool = True,
        stop_on_eos: bool = True,
    ) -> list[str]:
        return [self.decode(x, skip_special_tokens, stop_on_eos) for x in token_ids]
