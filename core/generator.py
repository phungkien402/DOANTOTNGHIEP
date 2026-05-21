"""
Generator — calls vLLM via its OpenAI-compatible API to produce a grounded
answer from the rewritten question and top reranked chunks.

The system prompt strictly instructs the LLM to answer ONLY from the provided
context — no hallucination allowed.

Run standalone: python -m core.generator
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI, APIConnectionError

from config import VLLM_BASE_URL, VLLM_MODEL
from core.models import RetrievedChunk

# Module-level client — created once when module is first imported
_client = OpenAI(base_url=f"{VLLM_BASE_URL}/v1", api_key="not-needed")

SYSTEM_PROMPT = (
    "Bạn là nhân viên hỗ trợ kỹ thuật phần mềm bệnh án điện tử EHC. "
    "Bạn trả lời câu hỏi của bác sĩ và nhân viên y tế dựa HOÀN TOÀN vào tài liệu "
    "trong phần CONTEXT bên dưới.\n\n"
    "Quy tắc:\n"
    "1. Chỉ dùng thông tin có trong CONTEXT. Không thêm gì từ bên ngoài.\n"
    "2. CONTEXT có thể chứa đường dẫn ngắn hoặc hướng dẫn tóm tắt — hãy diễn giải "
    "thành lời hướng dẫn tự nhiên, dễ hiểu. "
    "Chỉ nói \"Mình chưa tìm thấy hướng dẫn cho vấn đề này.\" nếu CONTEXT hoàn toàn "
    "không liên quan đến câu hỏi.\n"
    "3. Nếu hướng dẫn có nhiều bước (3+), dùng danh sách đánh số. "
    "Nếu chỉ 1-2 bước, viết thành câu tự nhiên, không cần đánh số.\n"
    "4. Trả lời bằng tiếng Việt, xưng \"mình\" hoặc không xưng, gọi người hỏi là \"bạn\".\n"
    "5. Giọng văn thân thiện, như đồng nghiệp hỗ trợ nhau — không quá trang trọng, "
    "không dùng \"người dùng\", không mở đầu bằng \"Để... bạn hãy thực hiện theo các bước sau:\".\n"
    "6. Mở đầu tự nhiên, đa dạng — KHÔNG dùng template cố định. Có thể bắt đầu thẳng "
    "vào hướng dẫn, hoặc 1 câu ngắn thừa nhận vấn đề nhưng KHÔNG được đoán nguyên nhân "
    "nếu CONTEXT không đề cập.\n"
    "7. TUYỆT ĐỐI không đoán hoặc suy diễn nguyên nhân nếu không có trong CONTEXT. "
    "Nếu CONTEXT không giải thích nguyên nhân, bỏ qua phần giải thích và đi thẳng vào hướng dẫn.\n"
    "8. Kết thúc bằng: \"Nếu vẫn gặp khó khăn, bạn có thể liên hệ thêm nhé!\"\n"
    "9. Không hỏi lại trừ khi câu hỏi thực sự mơ hồ."
)


def _build_user_prompt(query: str, chunks: list[RetrievedChunk], history: list[dict] = None, user_intent: str = None, knowledge_context: str = None) -> str:
    """Build the user prompt with context chunks, knowledge content, conversation history, intent, and question."""
    parts = []

    # Inject user intent at the top if available
    if user_intent:
        parts.append(f"[USER INTENT] {user_intent}\n")

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        label = "[PRIMARY REFERENCE]" if i == 1 else f"[SUPPLEMENTARY {i}]"
        context_parts.append(f"{label}\n{chunk.text}")

    context = "\n\n---\n\n".join(context_parts)
    parts.append(f"CONTEXT:\n{context}")

    # Inject knowledge content as supplementary operational guidance
    if knowledge_context:
        parts.append(f"\n---\n\n[OPERATIONAL GUIDANCE]\n{knowledge_context}")

    # Add last 2 turns of history if available
    if history:
        recent = history[-4:]  # last 2 turns = 4 entries (user+bot x2)
        history_lines = []
        for turn in recent:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {turn['text']}")
        parts.append("\nCONVERSATION HISTORY (for context only):\n" + "\n".join(history_lines))

    parts.append(f"\n---\n\nQUESTION: {query}\n\nNote: Answer based primarily on the [PRIMARY REFERENCE] above.")

    return "\n".join(parts)


class GeneratorError(Exception):
    """Raised when the generator cannot produce an answer (e.g. vLLM down)."""
    pass


class LLMUnavailableError(GeneratorError):
    """Raised specifically when vLLM is unreachable after retry (APIConnectionError)."""
    pass


def generate(query: str, chunks: list[RetrievedChunk], history: list[dict] = None, user_intent: str = None, knowledge_context: str = None) -> str:
    """
    Generate an answer grounded in the provided chunks.
    Optionally includes knowledge_context from data/knowledge/ files.
    Returns the answer text string.
    Raises GeneratorError if vLLM is unavailable.
    """
    print(f"[GENERATOR] Context chunks: {len(chunks)}")
    if user_intent:
        print(f"[GENERATOR] User intent: \"{user_intent}\"")
    if knowledge_context:
        print(f"[GENERATOR] Knowledge context: {len(knowledge_context)} chars")

    user_prompt = _build_user_prompt(query, chunks, history, user_intent=user_intent, knowledge_context=knowledge_context)
    print(f"[GENERATOR] Prompt length: ~{len(user_prompt)} chars")

    try:
        response = _client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=250 + 50,
            temperature=0.1,
        )

        answer = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if response.usage else "N/A"
        print(f"[GENERATOR] Response: \"{answer[:100]}...\"")
        print(f"[GENERATOR] Tokens used: {tokens_used}")
        return answer

    except APIConnectionError as e:
        # Retry once after 2s — vLLM may be busy with another request
        print(f"[GENERATOR] Connection error, retrying in 2s...")
        time.sleep(2)
        try:
            response = _client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=250 + 50,
                temperature=0.1,
            )
            answer = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else "N/A"
            print(f"[GENERATOR] Response (retry): \"{answer[:100]}...\"")
            print(f"[GENERATOR] Tokens used: {tokens_used}")
            return answer
        except Exception as retry_e:
            print(f"[GENERATOR] Retry failed ({type(retry_e).__name__}: {retry_e})")
            raise LLMUnavailableError(str(retry_e)) from retry_e

    except Exception as e:
        error_msg = f"[GENERATOR] vLLM unavailable ({type(e).__name__}: {e})"
        print(error_msg)
        raise GeneratorError(str(e)) from e


if __name__ == "__main__":
    dummy_chunks = [
        RetrievedChunk(
            text="Câu hỏi: Cách gộp hồ sơ bệnh nhân trùng\nHướng dẫn: Module Hành chính → Quản lý bệnh nhân → Chọn 2 hồ sơ → Bấm Gộp",
            score=0.94,
            metadata={"subject": "Cách gộp hồ sơ bệnh nhân trùng"}
        ),
    ]
    answer = generate("Làm sao để gộp hồ sơ bệnh nhân trùng trong EHC?", dummy_chunks)
    print(f"\n[GENERATOR] Final answer: {answer}")
