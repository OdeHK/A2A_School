
    def __init__(self, api_token, model="google/flan-t5-large"):
        self.api_token = api_token
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    def __call__(self, prompt, **kwargs):
        # Nếu prompt không phải string, cố gắng chuyển về string
        if not isinstance(prompt, str):
            if hasattr(prompt, "to_string"):
                prompt = prompt.to_string()
            else:
                prompt = str(prompt)
        payload = {"inputs": prompt}
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            # Tùy model, có thể trả về khác nhau
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"]
                elif "summary_text" in result[0]:
                    return result[0]["summary_text"]
                elif isinstance(result[0], str):
                    return result[0]
            return str(result)
        else:
            return f"[Lỗi HuggingFace API]: {response.status_code} - {response.text}"