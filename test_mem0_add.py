import os
from mem0 import Memory

# Mocking environment for a quick check
os.environ["MEM0_DIR"] = "/tmp/.mem0_test"

config = {
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o-mini", "api_key": "fake"}
    }
}
# We don't actually need to init it fully to check the signature if we trust inspect
# but let's see if we can at least call it (it will fail on LLM call but we check args first)

try:
    m = Memory()
    print("Memory initialized")
    # messages can be a string or list
    m.add("test", user_id="123", agent_id="bot", prompt="custom prompt")
except TypeError as e:
    print(f"TypeError: {e}")
except Exception as e:
    print(f"Other Exception (expected if no API key): {type(e).__name__}")
