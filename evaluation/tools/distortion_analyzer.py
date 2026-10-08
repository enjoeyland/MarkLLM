"""왜곡 측정기(토큰당 KL, base LM perplexity). 구현은 우리 repo의 `watermark_eval` 패키지(src/watermark_eval/)에 있고, 여기서는 MarkLLM의
평가 도구와 같은 자리에서 import할 수 있게 내보내기만 한다. 사용하려면 우리 repo에서 `pip install -e . --no-deps`로 설치되어 있어야 한다.

    from evaluation.tools.distortion_analyzer import DistortionAnalyzer, detect_scores
"""
from watermark_eval.distortion import DistortionAnalyzer, detect_scores  # noqa: F401
