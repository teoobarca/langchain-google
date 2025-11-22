"""Test usage metadata handling with None values"""
from google.genai import types as genai_types
from langchain_core.messages.ai import UsageMetadata

# Simulate a response with None usage values (as might come from SDK)
class MockUsage:
    def __init__(self):
        self.prompt_token_count = None  # Can be None!
        self.candidates_token_count = 10
        self.total_token_count = None  # Can be None!
        self.cached_content_token_count = None  # Can be None!

usage = MockUsage()

# Old buggy code would do:
# input_tokens = usage.prompt_token_count  # None!
# total_tokens = usage.total_token_count  # None!
# cache_read = getattr(usage, "cached_content_token_count", 0)  # Still None if attribute exists!

# Fixed code:
input_tokens = usage.prompt_token_count or 0
output_tokens = usage.candidates_token_count or 0
total_tokens = usage.total_token_count or 0
cache_read_tokens = getattr(usage, "cached_content_token_count", None) or 0

print("✅ Fixed usage values:")
print(f"  input_tokens: {input_tokens} (type: {type(input_tokens)})")
print(f"  output_tokens: {output_tokens} (type: {type(output_tokens)})")
print(f"  total_tokens: {total_tokens} (type: {type(total_tokens)})")
print(f"  cache_read: {cache_read_tokens} (type: {type(cache_read_tokens)})")

# Create UsageMetadata - should not have None values
try:
    cumulative_usage = UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_token_details={"cache_read": cache_read_tokens},
    )
    print(f"\n✅ UsageMetadata created successfully!")
    print(f"   {cumulative_usage}")

    # Check for None values
    if any(v is None for v in cumulative_usage.values() if not isinstance(v, dict)):
        print("\n❌ ERROR: UsageMetadata contains None values!")
    else:
        print("\n✅ No None values in top-level fields")

    # Check nested dict
    if any(v is None for v in cumulative_usage.get("input_token_details", {}).values()):
        print("❌ ERROR: input_token_details contains None values!")
    else:
        print("✅ No None values in input_token_details")

except Exception as e:
    print(f"\n❌ Failed to create UsageMetadata: {e}")
    import traceback
    traceback.print_exc()
