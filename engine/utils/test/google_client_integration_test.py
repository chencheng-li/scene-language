import unittest
import os
import json
import threading

# Add PYTHONPATH environment variable to include the root directory of the project
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from engine.utils.google_client import GoogleClient

# Import googleapiclient module to ensure it is available
import googleapiclient.discovery

class TestGoogleClient(unittest.TestCase):
    current_script = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_script)
    _test_cache_path = os.path.join(parent_dir, 'test_google_cache.json')
    print(f"Test cache path: {_test_cache_path}")

    @classmethod
    def setUpClass(cls):
        """Set up a GoogleClient with a temporary cache file."""
        cls.generator = GoogleClient(cache=cls._test_cache_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up by deleting the cache file after all tests."""
        if os.path.exists(cls._test_cache_path):
            os.remove(cls._test_cache_path)

    def test_generate_basic_query(self):
        """Test a basic generation query to the API."""
        cache_key, response = self.generator.generate("What is the capital of France?", "You are a helpful geography expert.")
        self.assertIsNotNone(response)
        self.assertTrue("Paris" in response[0])

        self.assertTrue(os.path.exists(self.generator.cache_file))

        # Confirm that it is in the cache
        with open(self.generator.cache_file, "r") as f:
            cache = json.load(f)
            self.assertTrue(cache_key in cache)
            self.assertTrue("Paris" in cache[cache_key][0])

    def test_concurrent_cache_updates(self):
        """Test concurrent updates to ensure .tmp and .lock files are managed correctly."""
        def update_cache():
            self.generator.update_cache("test_key", {"data": "value"})

        # Start multiple threads to update the cache simultaneously
        threads = [threading.Thread(target=update_cache) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Check for presence of .lock or .tmp files indicating incomplete or overlapping operations
        self.assertFalse(os.path.exists(self.generator.cache_file + ".lock"), ".lock file should not exist after operations complete.")
        self.assertFalse(os.path.exists(self.generator.cache_file + ".tmp"), ".tmp file should not exist after operations complete.")

    def test_multiple_completions(self):
        """Test generating multiple completions."""
        cache_key, responses = self.generator.generate(
            "List three colors.",
            "You are a helpful assistant.",
            num_completions=3,
            temperature=0.8  
        )
        self.assertEqual(len(responses), 3, "Expected 3 responses")
        for response in responses:
            self.assertTrue(any(color in ' '.join(response).lower() for color in ['red', 'blue', 'green', 'yellow', 'purple', 'orange']),
                            "Expected color names in responses")

    def test_cache_retrieval(self):
        """Test retrieving from cache and generating new responses."""
        prompt = "What's a famous landmark in Paris?"
        system = "You are a travel guide."
        
        # First call should generate new response
        cache_key1, response1 = self.generator.generate(prompt, system, num_completions=1)
        
        # Second call should retrieve from cache
        cache_key2, response2 = self.generator.generate(prompt, system, num_completions=1)
        
        self.assertEqual(cache_key1, cache_key2, "Cache keys should be identical for the same query")
        self.assertEqual(response1, response2, "Responses should be identical when retrieved from cache")
        
        # Third call with skip_cache_completions should generate new response
        _, response3 = self.generator.generate(prompt, system, num_completions=1, skip_cache_completions=1)
        
        self.assertNotEqual(response1, response3, "New response should be generated when skipping cache")

if __name__ == '__main__':
    unittest.main()

# How to use google_client_integration_test.py
# 1. Ensure you have set up the Google API key in `engine/key.py`.
# 2. Run the tests using a test runner like `unittest` or `pytest`.
# 3. The tests will check basic Google API interactions and concurrent cache updates.
