"""구간별 소요 시간 계측.

동작을 바꾸지 않고 '어디서 시간이 드는가'만 로그로 남긴다. 로그를
사후 파싱해 시간 예산을 분석하기 위한 것이므로, 출력 형식을 기계가
읽기 쉽게 고정한다:

    [TIMING] <scope> <name>=<seconds>

의존성은 표준 라이브러리뿐이라 Isaac 안팎 어디서든 쓸 수 있다.
"""

from __future__ import annotations

import time
from collections import OrderedDict


class StageTimer:
    """순차 구간의 경과 시간을 이름과 함께 기록한다.

    mark()를 부를 때마다 '직전 mark 이후 걸린 시간'이 그 이름으로
    쌓인다. 같은 이름을 여러 번 부르면 합산된다.
    """

    def __init__(self, scope, log_prefix="[TIMING]", enabled=True):
        self.scope = scope
        self.log_prefix = log_prefix
        self.enabled = bool(enabled)
        self.spans = OrderedDict()
        self._started = time.perf_counter()
        self._last = self._started

    def mark(self, name):
        """직전 지점부터 지금까지를 name 구간으로 기록한다."""
        if not self.enabled:
            return 0.0

        now = time.perf_counter()
        elapsed = now - self._last
        self._last = now
        self.spans[name] = self.spans.get(name, 0.0) + elapsed

        return elapsed

    def skip(self):
        """구간으로 세지 않고 기준점만 현재로 옮긴다."""
        self._last = time.perf_counter()

    @property
    def total(self):
        return time.perf_counter() - self._started

    def summary(self, extra=None):
        """한 줄 요약을 출력하고 dict로 돌려준다."""
        result = dict(self.spans)
        result["total"] = self.total

        if not self.enabled:
            return result

        parts = [
            f"{name}={value:.3f}"
            for name, value in self.spans.items()
            if value >= 0.0005
        ]

        if extra:
            parts.extend(
                f"{key}={value}" for key, value in extra.items()
            )

        print(
            f"{self.log_prefix} {self.scope} "
            f"total={result['total']:.3f} "
            + " ".join(parts)
        )

        return result


def parse_timing_lines(log_path, log_prefix="[TIMING]"):
    """실행 로그에서 [TIMING] 줄을 파싱한다.

    [{scope, spans:{name: seconds}}, ...]를 돌려준다.
    """
    import re

    pattern = re.compile(
        re.escape(log_prefix) + r"\s+(\S+)\s+(.*)$"
    )
    records = []

    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)

            if not match:
                continue

            spans = {}

            for token in match.group(2).split():
                if "=" not in token:
                    continue

                key, _, value = token.partition("=")

                try:
                    spans[key] = float(value)
                except ValueError:
                    spans[key] = value

            records.append(
                {"scope": match.group(1), "spans": spans}
            )

    return records
