from openai import OpenAI
import os

# Initialize OpenAI client
# client = OpenAI()

"""
Format Rules:
- Single sentence only
- Max 15 words
- Preserve technical terms exactly
- Focus on the primary concept only
- Remove subjective language
"""

def filter_query(query: str) -> str:
    """Filter and focus the query using GPT-3.5-turbo."""
    
    messages = [
        {"role": "system", "content": """Extract the main topic from the query and rephrase it as a clear search intent.

Format Rules:
- Single sentence only
- Max 15 words
- Preserve technical terms exactly
- Focus on the primary concept only
- Remove subjective language

Example:
Input: "I want to learn about how neural networks process data and also understand backpropagation in deep learning"
Output: neural networks data processing and backpropagation mechanisms

Input: "Could you help me find information about implementing OAuth2 authentication in Python web applications"
Output: OAuth2 authentication implementation in Python web applications"""},
        {"role": "user", "content": query}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in query filtering: {e}")
        return query  # Return original query if there's an error

def process_user_query(raw_query: str) -> str:
    """Main function to process and filter a user query."""
    return filter_query(raw_query)
