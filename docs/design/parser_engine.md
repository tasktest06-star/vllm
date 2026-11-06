# Parser Engine

Models announce reasoning and tool calls with markers in their output —
`<tool_call>`, `</think>`, `<｜DSML｜ calls>`. The parser engine turns that text
back into structured fields (`reasoning_content`, `tool_calls`) for the serving
layer, in both the non-streaming and the streaming path.

A parser is **declarative**: a model family contributes a `ParserEngineConfig`
listing its markers and what each one means, not hand-written string handling.
This page explains the concepts that config is built from. For the practical
guide to picking and enabling a parser, see
[Tool Calling](../features/tool_calling.md).

Code lives in `vllm/parser/engine/`.

## The pipeline

One delta of model output (`delta_text` plus its `delta_token_ids`) flows through
four stages:

```text
delta_text + delta_token_ids
        │
        ▼
┌───────────────────┐   special-token ids  ->  PreLexedTerminal
│  TokenIDScanner   │   everything else    ->  TextChunk
└───────────────────┘   token_id_scanner.py
        │
        ▼
┌───────────────────┐   text  ->  terminal tokens + content runs
│ IncrementalLexer  │
└───────────────────┘   incremental_lexer.py
        │
        ▼
┌───────────────────┐   (state, terminal) -> Transition
│   state machine   │   state             -> content event
└───────────────────┘   streaming_parser_engine.py
        │
        ▼
   SemanticEvent stream  (TEXT_CHUNK, REASONING_*, TOOL_CALL_*, ARG_VALUE_CHUNK …)
```

### Terminals

A **terminal** is a named marker. `ParserEngineConfig.terminals` maps a terminal
name to one spelling, or to several when a family writes the same marker more
than one way:

```python
terminals = {
    "TOOL_START": ("<tool_call>", "<tool▁call>"),
    "TOOL_END": "</tool_call>",
}
```

Everything that is not a terminal is **content**: the lexer emits it under the
reserved name `__CONTENT__`.

### Terminals are literal strings, not patterns

The lexer matches terminals by string comparison only. It never applies a
regular expression, so a terminal has to be a literal spelling you can write
out. `LexerShape` rejects a non-literal `TerminalDef` at construction rather
than letting it be silently dropped, because a terminal that never matches
parses as if it had never been declared — the marker would be streamed to the
user as ordinary content.

If a marker is genuinely variable (a counter, an id), model it as a fixed
literal that opens a state and let the state's own handling consume the variable
part; that is what `TOOL_NAME` and `TOOL_ARGS` are for.

### Special tokens arrive as ids, not text

Many markers are single tokens in the model's vocabulary. The detokenizer may
hold their text back, split it across deltas, or strip it, so matching them as
text is unreliable. `token_id_terminals` maps a terminal name to the token
*text* whose id the `TokenIDScanner` watches for; a hit becomes a
`PreLexedTerminal` and bypasses the lexer entirely.

This is the single most common cause of "my parser works non-streaming but not
streaming": a text terminal that happens to be one special token will not match
in the streaming path unless it is also listed in `token_id_terminals`.

When the scanner sees a terminal id whose text has not arrived in `delta_text`
yet, it **defers** that terminal rather than transitioning early, and resolves
it against a later delta. `flush_pending()` releases anything still deferred at
the end of the stream.

### Prefix-match buffering

The lexer's job is harder than one pass over a string, because a marker can
straddle a delta boundary. Given terminal `<tool_call>`, a delta ending in
`"...<tool_"` must not be emitted as content — the next delta may complete the
marker.

So the lexer **buffers** whenever the buffer is a proper prefix of some
terminal, and releases it when the next delta either completes the terminal or
proves it cannot. A buffer that is both a complete terminal *and* the prefix of
a longer one also waits, so `<invoke name=` is not emitted before the next chunk
decides whether it is really `<invoke name="`. Longest match wins.

`flush()` forces the buffer out as content; it is what `finish()` calls when the
stream ends mid-prefix.

The practical consequence: **chunking must not change the result**. Feeding a
string one character at a time must produce the same terminal sequence as
feeding it whole (content runs may be split differently — they are concatenated
downstream). Any new lexer behaviour should be tested that way.

### States, transitions and content events

The engine is a small state machine over `ParserState`:

| State | Meaning |
| --- | --- |
| `CONTENT` | ordinary assistant text |
| `REASONING` | inside a reasoning/thinking block |
| `MESSAGE_HEADER` | a header the model emits before its message body |
| `TOOL_PREAMBLE` | inside a tool-call block, before the name |
| `TOOL_NAME` | reading the function name |
| `TOOL_ARGS` | reading the arguments |
| `TOOL_BETWEEN` | between calls in a multi-call block |

`transitions` maps `(state, terminal)` to a `Transition`: the next state, plus
the events to emit on the way. `content_events` maps a state to the event that
text seen *in* that state produces — `CONTENT` to `TEXT_CHUNK`, `REASONING` to
`REASONING_CHUNK`, and so on.

Text emitted in a state with **no** `content_events` entry is discarded. That is
deliberate — it is how a block's syntactic scaffolding is dropped — so when you
add a state, decide explicitly whether its text should surface.

## Writing or changing a parser

- A new model family subclasses `ParserEngine` with its own config, or
  subclasses an existing config when the format is a variant of one
  (`seed_oss.py`, `ling3.py` do this).
- Keep a model-specific fix in that model's `vllm/parser/<model>.py`. Change
  `engine/` only when the behaviour is wrong for every parser, and say so in the
  PR.
- `DelegatingParser` exists for backward compatibility with the older separate
  `ReasoningParser` / `ToolParser` pair. Do not use it for new formats.

See `vllm/parser/AGENTS.md` for the full layout, and `tests/parser/engine/` for
the replay harness used to test a config against recorded model output.
