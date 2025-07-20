import json
import time
import openai


def run_agentic_example(question: str = "What is 7 plus 5?"):
    """Demonstrate a simple agent using the OpenAI Assistant API."""
    client = openai.OpenAI()

    assistant = client.beta.assistants.create(
        name="Calculator Assistant",
        instructions="You answer math questions by adding numbers using a tool.",
        tools=[{
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                }
            }
        }]
    )

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=question)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)

    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status == "completed":
            break
        if run.status == "requires_action":
            for call in run.required_action["submit_tool_outputs"]["tool_calls"]:
                args = json.loads(call["function"]["arguments"])
                result = args["a"] + args["b"]
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=[{"tool_call_id": call["id"], "output": str(result)}],
                )
        else:
            time.sleep(1)

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    response = messages.data[0].content[0].text.value
    print(response)


if __name__ == "__main__":
    run_agentic_example()
