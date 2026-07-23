"""Arithmetic verification for calculation questions.

The model may extract facts and propose derived values, but it is never
trusted to do the arithmetic: the ledger it returns is re-computed locally
with ``Decimal`` and any derived value whose reported result disagrees is
discarded rather than repeated downstream as authoritative evidence.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Sequence


_CALCULATION_MARKER = re.compile(
    r"计算|排序|高到低|低到高|升序|降序|合计|总计|平均|金额|现金价值|退保所得|"
    r"每股|每10股"
)


def _verify_calculation_draft(raw: str) -> str:
    """Validate a Qwen calculation ledger with local Decimal arithmetic.

    The model may extract facts and propose derived values, but it is not
    trusted to perform the arithmetic.  Invalid or incomplete ledgers are
    discarded instead of being repeated as authoritative evidence.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""

    def parse_decimal(value):
        if isinstance(value, bool) or value is None:
            raise ValueError("invalid numeric value")
        text = str(value).strip().replace(",", "")
        if text.endswith("%"):
            return Decimal(text[:-1]) / Decimal("100")
        return Decimal(text)

    values = {}
    rendered = []
    for item in payload.get("facts", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        value = item.get("value")
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            parsed = parse_decimal(value)
        except (InvalidOperation, ValueError, TypeError):
            continue
        if not parsed.is_finite():
            continue
        key = name.strip()
        values[key] = parsed
        unit = item.get("unit") or ""
        rendered.append(f"- fact {key} = {parsed}{unit}")

    operations = {"add", "subtract", "multiply", "divide"}
    pending = [item for item in payload.get("derived", []) if isinstance(item, dict)]
    while pending:
        progressed = False
        remaining = []
        for item in pending:
            name = item.get("name")
            operation = item.get("operation")
            operands = item.get("operands")
            reported = item.get("value")
            if (
                not isinstance(name, str)
                or not name.strip()
                or operation not in operations
                or not isinstance(operands, list)
                or not operands
            ):
                continue
            if operation in {"subtract", "divide"} and len(operands) != 2:
                continue
            try:
                operand_values = [
                    values[ref] if isinstance(ref, str) and ref in values else parse_decimal(ref)
                    for ref in operands
                ]
                if operation == "add":
                    computed = sum(operand_values, Decimal("0"))
                elif operation == "subtract":
                    computed = operand_values[0] - operand_values[1]
                elif operation == "multiply":
                    computed = Decimal("1")
                    for value in operand_values:
                        computed *= value
                else:
                    if operand_values[1] == 0:
                        continue
                    computed = operand_values[0] / operand_values[1]
                reported_decimal = parse_decimal(reported)
            except (InvalidOperation, ValueError, TypeError, ZeroDivisionError):
                remaining.append(item)
                continue
            if not reported_decimal.is_finite() or reported_decimal != computed:
                continue
            key = name.strip()
            values[key] = computed
            unit = item.get("unit") or ""
            rendered.append(
                f"- derived {key} = {computed}{unit} "
                f"({operation}: {', '.join(str(ref) for ref in operands)})"
            )
            progressed = True
        if not progressed:
            break
        pending = remaining

    return "\n".join(rendered)


def _needs_calculation_judgement(question: str, options: Sequence[str]) -> bool:
    """Use a second Qwen pass only when the question explicitly requires math."""
    # Do not trigger merely because a distractor contains a percentage, amount,
    # or per-share figure.  The stem must itself ask for a computation/order.
    return bool(_CALCULATION_MARKER.search(str(question)))
