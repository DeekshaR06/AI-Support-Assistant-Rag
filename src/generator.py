"""
Answer generation layer.

Two backends, chosen automatically:
  1. LLM mode  - if an OPENAI_API_KEY environment variable is set (and the `openai`
     package is installed), the retrieved FAQ snippets are stuffed into a prompt and
     a real LLM writes the final answer. This is "real" RAG generation.
  2. Template mode (default, no API key needed) - composes a clean answer directly
     from the top retrieved FAQ, adapts its tone to the detected sentiment, and is
     fully offline. This is what runs out-of-the-box so the whole project is
     demoable with zero setup and zero cost.

This split is the point worth explaining in an interview: you designed the system so
the *retrieval and business logic* (intent, sentiment, knowledge lookup) are
completely decoupled from *which* text-generation backend produces the final
sentence — swapping GPT for Claude, or for a local model, only touches this file.
"""
import os


SYSTEM_PROMPT = (
    "You are a helpful, concise customer support assistant. Answer the customer's "
    "question using ONLY the provided knowledge base snippets. If the snippets don't "
    "say you cannot answer and suggest the customer contact the support team instead of guessing"
    "Keep the answer under 80 words and match a tone appropriate to the customer's "
    "detected sentiment (extra empathetic if they sound frustrated)."
)


def _llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _generate_with_llm(query: str, retrieved: list, sentiment_label: str) -> str:
    from openai import OpenAI  # imported lazily so the package is optional

    client = OpenAI()
    context = "\n\n".join(f"- {r['question']}: {r['answer']}" for r in retrieved)
    user_prompt = (
        f"Customer sentiment: {sentiment_label}\n\n"
        f"Knowledge base snippets:\n{context}\n\n"
        f"Customer question: {query}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def _generate_with_template(query: str, retrieved: list, sentiment_label: str) -> str:
    if not retrieved:
        return ("I couldn't find anything specific about that in our knowledge base. "
                "Please contact our support team, who can help further.")

    top = retrieved[0]
    prefix = ""
    if sentiment_label == "negative":
        prefix = "I'm really sorry for the trouble this has caused. "
    elif sentiment_label == "positive":
        prefix = "Glad to hear it! "

    answer = f"{prefix}{top['answer']}"
    if len(retrieved) > 1 and retrieved[1]["similarity"] > 0.5:
        answer += f" You may also find this useful: {retrieved[1]['answer']}"
    return answer


def generate_answer(query: str, retrieved: list, sentiment_label: str = "neutral") -> dict:
    if _llm_available():
        try:
            text = _generate_with_llm(query, retrieved, sentiment_label)
            return {"answer": text, "backend": "openai:gpt-4o-mini"}
        except Exception as exc:  # falls back gracefully instead of crashing a demo
            fallback = _generate_with_template(query, retrieved, sentiment_label)
            return {"answer": fallback, "backend": f"template (LLM call failed: {exc})"}

    text = _generate_with_template(query, retrieved, sentiment_label)
    return {"answer": text, "backend": "template"}


if __name__ == "__main__":
    fake_retrieved = [
        {"question": "How do I get a refund for a duplicate charge?",
         "answer": "Contact billing support with your order ID; duplicate charges are refunded within 3-5 business days.",
         "similarity": 0.97}
    ]
    print(generate_answer("I was charged twice, help!", fake_retrieved, "negative"))
