import requests
import json

# Test cases

with open('data/rundown-test/rundown_1.txt', 'r', encoding='utf-8') as file:
    text_w_format = file.read()


test_cases = [
    {
        "query": "Amazon's New AI Model",
        "text": text_w_format
    },
    # {
    #     "query": "climate change",
    #     "text": text_w_format
    # },
    # {
    #     "query": "programming",
    #     "text": text_w_format
    # }
]

# Test each case
for test in test_cases:
    print(f"\nTesting relevancy for:")
    print(f"Query: {test['query']}")
    print(f"Text:  {test['text']}")
    
    response = requests.post(
        "http://localhost:8000/relevancy",
        headers={"Content-Type": "application/json"},
        json=test
    )
    print(f"Response: {response}")
    
    # result = response.json()
    # print(f"Result: {json.dumps(result, indent=2)}")

