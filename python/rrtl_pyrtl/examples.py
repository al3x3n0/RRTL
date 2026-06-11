"""Small PyRTL designs used by the bridge smoke corpus."""

from __future__ import annotations

import pyrtl


def counter() -> pyrtl.Block:
    en = pyrtl.Input(1, "en")
    out = pyrtl.Output(4, "out")
    count = pyrtl.Register(4, "count", reset_value=1)

    out <<= count
    count.next <<= pyrtl.mux(en, count + 1, count)
    return pyrtl.working_block()


def alu() -> pyrtl.Block:
    a = pyrtl.Input(4, "a")
    b = pyrtl.Input(4, "b")
    sel = pyrtl.Input(1, "sel")
    out = pyrtl.Output(5, "out")

    out <<= pyrtl.mux(sel, a + b, a.zero_extended(5) ^ b.zero_extended(5))
    return pyrtl.working_block()
