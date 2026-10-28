# HMMCond: 정확한 조건부 샘플링 워터마크. 구현은 우리 repo의 `hmm_watermark` 패키지(src/hmm_watermark/hmm_cond.py)에 있고,
# 여기서는 MarkLLM이 AutoWatermark.load("HMMCond", ...)로 찾을 수 있게 import해서 내보내기만 한다.
# 사용하려면 우리 repo에서 `pip install -e . --no-deps`로 설치되어 있어야 한다.
from hmm_watermark.hmm_cond import HMMCond, HMMCondConfig  # noqa: F401
