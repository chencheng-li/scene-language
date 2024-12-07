import os
import json
import time
from googleapiclient.discovery import build
from google.oauth2 import service_account
from engine.constants import GOOGLE_API_KEY, MAX_TOKENS, TEMPERATURE, NUM_COMPLETIONS

class GoogleClient:
    def __init__(self, model_name='text-bison-001', cache="google_cache.json"):
        self.cache_file = cache
        self.model_name = model_name

        # Load cache
        if os.path.exists(cache):
            while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
                time.sleep(0.1)
            with open(cache, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

        # Initialize Google API client
        try:
            credentials = service_account.Credentials.from_service_account_info(
                json.loads(GOOGLE_API_KEY),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self.client = build("generativelanguage", "v1beta2", credentials=credentials)
        except json.JSONDecodeError:
            raise ValueError("Invalid GOOGLE_API_KEY. Please ensure it is set correctly and contains valid JSON.")
        except Exception as e:
            raise ValueError(f"Failed to initialize Google API client: {str(e)}")

    def generate(self, user_prompt, system_prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop_sequences=None, verbose=False,
                 num_completions=NUM_COMPLETIONS, skip_cache_completions=0, skip_cache=False):

        print(f'[INFO] Google: querying for {num_completions=}, {skip_cache_completions=}')
        if skip_cache:
            print(f'[INFO] Google: Skipping cache')
        if verbose:
            print(user_prompt)
            print("-----")

        # Prepare messages for the API request
        messages = [{"role": "user", "content": user_prompt}]

        cache_key = None
        results = []
        if not skip_cache:
            cache_key = str((user_prompt, system_prompt, max_tokens, temperature, stop_sequences, 'google'))

            num_completions = skip_cache_completions + num_completions
            if cache_key in self.cache:
                print(f'[INFO] Google: cache hit {len(self.cache[cache_key])}')
                if len(self.cache[cache_key]) < num_completions:
                    num_completions -= len(self.cache[cache_key])
                    results = self.cache[cache_key]
                else:
                    return cache_key, self.cache[cache_key][skip_cache_completions:num_completions]

        while num_completions > 0:
            response = self.client.text().generate(
                model=self.model_name,
                body={
                    "prompt": {
                        "text": user_prompt
                    },
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                    "candidateCount": 1
                }
            ).execute()

            num_completions -= 1

            content = response.get("candidates", [{}])[0].get("output", "")
            results.append(content.split('\n'))

        if not skip_cache:
            self.update_cache(cache_key, results)

        return cache_key, results[skip_cache_completions:]

    def update_cache(self, cache_key, results):
        while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
            time.sleep(0.1)
        with open(self.cache_file + ".lock", "w") as f:
            pass
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)
        self.cache[cache_key] = results
        with open(self.cache_file + ".tmp", "w") as f:
            json.dump(self.cache, f)
        os.rename(self.cache_file + ".tmp", self.cache_file)
        os.remove(self.cache_file + ".lock")

def setup_google():
    model = GoogleClient(cache='google_cache.json')
    return model
