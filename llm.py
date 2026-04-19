import google.genai as genai




class TrafficLLM:

    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def generate_report(self, speeds, status, trend, user_query):
       

        prompt = f"""
        You are a smart and human-like traffic assistant.

        Traffic:
        - Status: {status}
        - Trend: {trend}

        User Question:
        {user_query}

        Instructions:
        - Answer the user's question directly
        - Do NOT repeat the same sentence structure every time
        - Match tone to question:
        - If emotional ("regret", "worth") → reassuring tone
        - If advice ("should I", "would you") → recommendation tone
        - If casual → conversational tone
        - Use varied phrasing each time
        - Keep it natural and short (1–2 lines)

        Examples:
        Q: Will I regret leaving now?
        A: You should be fine — traffic is smooth at the moment.

        Q: Should I go now?
        A: Yes, it’s a good time to head out.

        Q: Would you go out right now?
        A: I would — traffic conditions look clear.

        Now answer:
        """

        response = self.model.generate_content(prompt)
        return response.text