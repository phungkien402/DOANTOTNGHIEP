"""
Intent guard:
1. LLM classifier: is query EHC-related? (YES/NO, max_tokens=5)
2. If NO → LLM chat fallback (short, polite, scoped to avoid going off-rail)

Uses the same synchronous OpenAI client pattern as the rest of the codebase.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI, APIConnectionError

from config import VLLM_BASE_URL, VLLM_MODEL


def _load_terminology() -> str:
    """Load terminology from data/terminology.json and format for prompt injection.

    Returns empty string if file not found — classify prompt still works without it.
    """
    term_path = Path(__file__).parent.parent / "data" / "terminology.json"
    if not term_path.exists():
        return ""
    try:
        with open(term_path, encoding="utf-8") as f:
            data = json.load(f)
        lines = []
        for section in data.values():
            label = section["label"]
            terms = ", ".join(section["terms"])
            lines.append(f"- {label}: {terms}")
        return "\n".join(lines)
    except Exception:
        return ""


_TERMINOLOGY = _load_terminology()

# Module-level client — same pattern as generator.py / query_rewriter.py
_client = OpenAI(base_url=f"{VLLM_BASE_URL}/v1", api_key="not-needed")

CLASSIFY_PROMPT = """Bạn là bộ lọc câu hỏi cho hệ thống hỗ trợ nghiệp vụ nội bộ bệnh viện.
Người dùng là nhân viên bệnh viện (bác sĩ, điều dưỡng, dược sĩ, nhân viên đón tiếp, thu ngân).
Họ đang chat với bot hỗ trợ phần mềm EHC — đây là ngữ cảnh mặc định của MỌI câu hỏi.

Vì vậy, người dùng thường KHÔNG đề cập tên phần mềm hay module cụ thể.
Họ hỏi ngắn gọn, dùng jargon nội bộ, ví dụ:
  "sai giá", "không in được", "bị âm kho", "không tìm thấy bệnh nhân"

Thuật ngữ nghiệp vụ thường gặp:
{terminology}

Trả lời YES nếu câu hỏi có thể là nghiệp vụ nội bộ bệnh viện, bao gồm:
- Thao tác phần mềm, lỗi hệ thống, hướng dẫn quy trình
- Câu hỏi về bệnh nhân, thuốc, viện phí, xét nghiệm, BHYT, in phiếu
- Câu ngắn, mơ hồ nhưng nghe có vẻ liên quan đến vận hành bệnh viện
- Khi KHÔNG CHẮC → YES (ưu tiên recall)

Trả lời NO chỉ khi câu hỏi RÕ RÀNG không liên quan:
- Chào hỏi xã giao thuần túy (hello, cảm ơn, tạm biệt)
- Chủ đề hoàn toàn ngoài y tế / phần mềm (thời tiết, giải trí, tin tức)
- Lệnh phá hoại hạ tầng (xoá database, format disk, drop table)

Trả lời CHỈ bằng một từ: YES hoặc NO.
Câu hỏi: "{query}"
"""

CHAT_SYSTEM_PROMPT = """Bạn là trợ lý phần mềm EHC. Luôn trả lời bằng tiếng Việt.
Câu hỏi này nằm ngoài phạm vi hỗ trợ. Trả lời đúng 1 câu ngắn, lịch sự, từ chối và nhắc bạn chỉ hỗ trợ phần mềm EHC.
KHÔNG được trả lời nội dung câu hỏi. PHẢI trả lời bằng tiếng Việt."""

_FALLBACK_RESPONSE = "Xin chào! Mình là trợ lý hỗ trợ phần mềm EHC. Bạn có câu hỏi gì về phần mềm không?"


def classify(query: str) -> bool:
    """Return True if query is off-topic (not EHC-related)."""
    try:
        prompt = CLASSIFY_PROMPT.format(terminology=_TERMINOLOGY, query=query.strip())
        response = _client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip().upper()
        print(f"[INTENT_GUARD] Classify: \"{query}\" → {answer}")
        return answer.startswith("NO")

    except APIConnectionError:
        # Retry once after 1s
        print("[INTENT_GUARD] Classifier connection error, retrying in 1s...")
        time.sleep(1)
        try:
            prompt = CLASSIFY_PROMPT.format(terminology=_TERMINOLOGY, query=query.strip())
            response = _client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip().upper()
            print(f"[INTENT_GUARD] Classify (retry): \"{query}\" → {answer}")
            return answer.startswith("NO")
        except Exception as e:
            print(f"[INTENT_GUARD] Classifier retry failed: {e} — allowing query through")
            return False  # fail open

    except Exception as e:
        print(f"[INTENT_GUARD] Classifier failed: {e} — allowing query through")
        return False  # fail open


def chat_fallback(query: str) -> str:
    """Generate a short, scoped chat response for off-topic queries."""
    try:
        response = _client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=60,
            temperature=0.3,
        )
        result = response.choices[0].message.content.strip()
        print(f"[INTENT_GUARD] Chat fallback: \"{result}\"")
        return result

    except APIConnectionError:
        # Retry once after 1s
        print("[INTENT_GUARD] Chat fallback connection error, retrying in 1s...")
        time.sleep(1)
        try:
            response = _client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                max_tokens=60,
                temperature=0.7,
            )
            result = response.choices[0].message.content.strip()
            print(f"[INTENT_GUARD] Chat fallback (retry): \"{result}\"")
            return result
        except Exception as e:
            print(f"[INTENT_GUARD] Chat fallback retry failed: {e}")
            return _FALLBACK_RESPONSE

    except Exception as e:
        print(f"[INTENT_GUARD] Chat fallback failed: {e}")
        return _FALLBACK_RESPONSE
