import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(override=True)

NAME = "Larry Kubende"


class Me:
    """Digital twin of Larry Kubende - answers questions about his career, skills, and experience."""

    def __init__(self):
        self.openai = OpenAI()
        self.model = "gpt-4o-mini"
        self.summary = open("data/summary.txt").read()
        self.system_prompt = self._build_system_prompt()

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "record_unknown_question",
                    "description": "Record a question that you cannot answer based on the available information about Larry.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question that could not be answered",
                            }
                        },
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "record_user_details",
                    "description": "Record details of someone who wants to get in touch with Larry. Use this when the user shares their name or email.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": "The user's email address",
                            },
                            "name": {
                                "type": "string",
                                "description": "The user's name",
                            },
                            "notes": {
                                "type": "string",
                                "description": "Any additional context about what they want to discuss",
                            },
                        },
                        "required": ["email", "name"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def _build_system_prompt(self):
        return f"""You are acting as {NAME}, a Software Engineer based in Nairobi, Kenya.
You are answering questions on {NAME}'s behalf, representing him in a professional capacity.

Your responsibilities:
- Answer questions about {NAME}'s career, skills, experience, projects, and professional background faithfully and accurately.
- Use a professional, confident tone. Be concise but thorough.
- Only answer based on the information provided below. Do not fabricate details.
- If you don't know the answer to a question, say so honestly and use your record_unknown_question tool to log it.
- Do NOT answer questions about {NAME}'s personal life, relationships, family, or anything outside his professional profile. Politely redirect to professional topics.
- If someone wants to get in touch with {NAME}, encourage them to share their name and email, then use the record_user_details tool.
- You may suggest they reach out to {NAME} at karanilarry@gmail.com for professional inquiries.

Here is {NAME}'s professional summary and background:

{self.summary}
"""

    def _handle_tool_call(self, tool_call):
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if name == "record_unknown_question":
            print(f"[Unknown question logged]: {args.get('question')}")
            return json.dumps({"status": "recorded", "message": "Question has been logged for Larry to review."})

        elif name == "record_user_details":
            print(f"[Contact recorded]: {args.get('name')} - {args.get('email')}")
            return json.dumps({"status": "recorded", "message": f"Thanks! Larry will be in touch with {args.get('name')} at {args.get('email')}."})

        return json.dumps({"error": "Unknown tool"})

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt}]

        for entry in history:
            if entry["role"] == "user":
                messages.append({"role": "user", "content": entry["content"]})
            elif entry["role"] == "assistant":
                messages.append({"role": "assistant", "content": entry["content"]})

        messages.append({"role": "user", "content": message})

        response = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
        )

        # Handle tool calls in a loop
        while response.choices[0].finish_reason == "tool_calls":
            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                result = self._handle_tool_call(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            response = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
            )

        return response.choices[0].message.content


if __name__ == "__main__":
    me = Me()

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .gradio-container { max-width: 800px !important; margin: auto; }
        .header { text-align: center; padding: 20px 0; }
        """
    ) as ui:
        gr.HTML("""
        <div class="header">
            <h1>Larry Kubende</h1>
            <h3>Software Engineer | Nairobi, Kenya</h3>
            <p>Ask me about my career, skills, projects, or experience.</p>
        </div>
        """)

        chatbot = gr.ChatInterface(
            fn=me.chat,
            type="messages",
            examples=[
                "What is your current role?",
                "What technologies do you work with?",
                "Tell me about your experience with insurance technology.",
                "What projects have you worked on?",
                "How can I get in touch with you?",
            ],
        )

    ui.launch(inbrowser=True)
