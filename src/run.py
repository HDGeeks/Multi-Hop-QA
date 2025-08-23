from models.openai_client import query_openai

result = query_openai(
    prompt="Who was the first president of the United States?",
    run_id=1,
    qid="test1",
    domain="history",
    setting="gold"
)

print(result)