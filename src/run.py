from models.gpt import query_openai
from dotenv import load_dotenv
import pprint as pp
result = query_openai(
    prompt="Who was the first president of the United States?",
    run_id=1,
    qid="test1",
    domain="history",
    setting="gold"
)

pp.pprint(result)