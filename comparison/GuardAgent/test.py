"""
Simple API connectivity test for OpenAI/Azure OpenAI
"""

import openai
from openai import OpenAI
import json


def test_api_connection(config_file="config.py"):
    """
    Test API connectivity using configuration from config.py

    Args:
        config_file: Path to the config file (default: "config.py")

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        # Import configuration
        from config import model_config

        # Test with different model names
        test_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]

        for model_name in test_models:
            print(f"\n{'='*60}")
            print(f"Testing with model: {model_name}")
            print("=" * 60)

            try:
                config = model_config(model_name)
                print(f"Configuration loaded:")
                print(f"  - Model: {config.get('model', 'N/A')}")
                print(
                    f"  - API Key: {'*' * 10}{config.get('api_key', '')[-4:] if config.get('api_key') else 'N/A'}"
                )
                print(f"  - Base URL: {config.get('base_url', 'Default OpenAI')}")

                # Create client
                client_kwargs = {"api_key": config["api_key"]}
                if "base_url" in config:
                    client_kwargs["base_url"] = config["base_url"]

                client = OpenAI(**client_kwargs)

                # Simple test message
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "Say 'API connection successful!' if you can read this.",
                    },
                ]

                print(f"\nSending test request...")
                response = client.chat.completions.create(
                    model=config["model"],
                    messages=messages,
                    temperature=0,
                    max_tokens=50,
                )

                result = response.choices[0].message.content.strip()
                print(f"✓ Response received: {result}")
                print(f"✓ API connection successful for {model_name}!")
                return True

            except Exception as e:
                print(f"✗ Failed with {model_name}: {str(e)}")
                continue

        print(f"\n{'='*60}")
        print("All models failed. Please check your configuration.")
        print("=" * 60)
        return False

    except ImportError as e:
        print(f"Error importing config: {e}")
        print("Make sure config.py exists and has model_config function.")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_simple_completion(prompt="What is 2+2?", model="gpt-4"):
    """
    Test a simple completion with custom prompt

    Args:
        prompt: Custom prompt to test
        model: Model name to use

    Returns:
        str: Response or error message
    """
    try:
        from config import model_config

        config = model_config(model)
        client_kwargs = {"api_key": config["api_key"]}
        if "base_url" in config:
            client_kwargs["base_url"] = config["base_url"]

        client = OpenAI(**client_kwargs)

        messages = [{"role": "user", "content": prompt}]

        print(f"\nTesting with prompt: {prompt}")
        response = client.chat.completions.create(
            model=config["model"], messages=messages, temperature=0, max_tokens=100
        )

        result = response.choices[0].message.content.strip()
        print(f"Response: {result}")
        return result

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return error_msg


if __name__ == "__main__":
    print("=" * 60)
    print("API Connection Test")
    print("=" * 60)

    # Test API connection
    success = test_api_connection()

    if success:
        print("\n" + "=" * 60)
        print("Running additional test...")
        print("=" * 60)
        test_simple_completion()
    else:
        print("\nPlease check your API configuration in config.py")
        print("Make sure:")
        print("  1. API key is correct")
        print("  2. Base URL is correct (if using Azure)")
        print("  3. Model name is correct")
