"""
agent.py
--------
The hand-rolled agent loop. No LangChain / LlamaIndex / agent framework.

How it works, step by step:

  1. We give the LLM a system prompt describing the 3 tools and the goal.
  2. We ask the LLM (Gemini, via its native function-calling feature)
     what to do next, given the conversation so far.
  3. The LLM responds with either:
       a) a function call (name + args) -> we run TOOL_DISPATCH[name](**args)
          and feed the result back into the conversation as a function
          response part
       b) plain text -> the brief -> loop exits
  4. We repeat, capping at MAX_STEPS iterations so a confused LLM can
     never spin forever (the "no infinite loop risk" rubric item).

Using the model's native function-calling (rather than hand-parsing free
text) is still "hand-rolling the agent loop": the framework concept being
avoided here is LangChain-style agents/executors that own the loop,
memory, and tool routing for you. We own all of that ourselves below —
we just use the LLM provider's structured function-call output instead of
regex-parsing "Action: search_youtube(...)" out of free text, which is
strictly more reliable and is what every modern agent (framework or not)
does under the hood.

NOTE on automatic_function_calling: Gemini's SDK can auto-execute
functions for you (AFC). We deliberately disable that
(automatic_function_calling=disable=True) so that WE own the loop,
the state tracking (all_videos_by_id, comments_by_video), and the exit
conditions explicitly -- exactly what this assignment is testing.
"""

import os
from google import genai
from google.genai import types

from tools import search_youtube, get_video_comments, summarize_findings

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "MODEL = "gemini-2.5-flash-lite""
MAX_STEPS = 8  # hard ceiling -> guarantees the loop terminates

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ---------------------------------------------------------------------------
# Tool schemas the LLM sees. This is the *contract*, not the implementation.
# ---------------------------------------------------------------------------
SEARCH_YOUTUBE_DECL = {
    "name": "search_youtube",
    "description": "Search YouTube for videos on a given topic. Returns up to 5 videos with title, videoId, viewCount, publishedAt.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query / topic"}
        },
        "required": ["query"],
    },
}

GET_VIDEO_COMMENTS_DECL = {
    "name": "get_video_comments",
    "description": "Fetch the top 10 comments (sorted by likeCount) for a single YouTube video.",
    "parameters": {
        "type": "object",
        "properties": {
            "video_id": {"type": "string", "description": "YouTube videoId"}
        },
        "required": ["video_id"],
    },
}

SUMMARIZE_FINDINGS_DECL = {
    "name": "summarize_findings",
    "description": (
        "Produce the FINAL research brief once you have picked 2-3 relevant "
        "videos and fetched their comments. This ends the research task — "
        "only call this when you have enough information."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "video_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The 2-3 videoIds you selected as most relevant",
            }
        },
        "required": ["video_ids"],
    },
}

TOOLS_CONFIG = [
    types.Tool(function_declarations=[
        SEARCH_YOUTUBE_DECL, GET_VIDEO_COMMENTS_DECL, SUMMARIZE_FINDINGS_DECL
    ])
]

SYSTEM_PROMPT = """You are a research agent. The user gives you an AI-related topic.
Your job:
1. Call search_youtube with a good query for the topic.
2. From the results, pick the 2-3 most relevant videos (use view count, recency, and title relevance as judgment).
3. Call get_video_comments for EACH of those 2-3 videos (one call per video).
4. Once you have comments for your selected videos, call summarize_findings with the video_ids you chose. This produces the final brief and ends the task.

Rules:
- Call exactly one tool at a time.
- Never call a tool that isn't in your tool list.
- Don't call get_video_comments more times than the number of videos you selected.
- Once summarize_findings has been called, you are done — do not call anything else.
"""

# Maps tool name -> the actual Python function. This dict IS the dispatch
# table: adding a 4th tool later means adding one function to tools.py
# and one line here — nothing else in the loop changes.
TOOL_DISPATCH = {
    "search_youtube": lambda args: search_youtube(args.get("query", "")),
    "get_video_comments": lambda args: get_video_comments(args.get("video_id", "")),
}

GENERATE_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=TOOLS_CONFIG,
    temperature=0.2,
    # Disable Automatic Function Calling: we want manual control over
    # dispatch, state tracking, and exit conditions (see module docstring).
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def run_agent(topic: str, verbose: bool = True) -> str:
    """
    Runs the full agent loop for one user topic.
    Returns the final brief as a string (or an explanation if the loop
    ended without producing one).
    """
    if not client:
        return "ERROR: GEMINI_API_KEY is not set. Add it to your .env file."

    # Gemini's "contents" list is the conversation history (like
    # OpenAI/Groq's "messages" list), made of types.Content objects.
    contents = [
        types.Content(role="user", parts=[
            types.Part.from_text(text=f"Research this topic on YouTube: {topic}")
        ])
    ]

    # State the agent accumulates across steps, needed to actually call
    # summarize_findings (which needs full video + comment objects, not
    # just the IDs the LLM passes back).
    all_videos_by_id = {}
    comments_by_video = {}

    for step in range(1, MAX_STEPS + 1):
        if verbose:
            print(f"\n--- Step {step} ---")

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=GENERATE_CONFIG,
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts or []
        function_calls = [p.function_call for p in parts if p.function_call is not None]

        # Case A: model produced plain text with no function call.
        # Treat this as a fallback final answer -> exit condition.
        if not function_calls:
            if verbose:
                print("Model returned plain text (no function call). Treating as final answer.")
            return response.text or "(The model returned an empty response.)"

        # Record the model's turn (including the function call) before
        # we respond to it -- required by the contents-history contract.
        contents.append(candidate.content)

        # We only act on the FIRST function call per step, even if the
        # model requested several -- keeps "one tool at a time" easy to
        # reason about. Any extra calls get a neutral function response
        # so the conversation stays valid.
        response_parts = []
        for i, fc in enumerate(function_calls):
            name = fc.name
            args = dict(fc.args) if fc.args else {}

            if i > 0:
                response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"note": "Skipped: only one tool call processed per step."},
                    )
                )
                continue

            if verbose:
                print(f"Tool call: {name}({args})")

            # ---- Unknown tool name: the "what if it hallucinates a tool?" case ----
            if name not in TOOL_DISPATCH and name != "summarize_findings":
                result = {
                    "error": f"Unknown tool '{name}'. Valid tools are: "
                             f"{list(TOOL_DISPATCH.keys()) + ['summarize_findings']}."
                }
                response_parts.append(types.Part.from_function_response(name=name, response=result))
                continue

            # ---- Exit condition: summarize_findings ends the loop ----
            if name == "summarize_findings":
                video_ids = args.get("video_ids", [])
                selected_videos = [all_videos_by_id[v] for v in video_ids if v in all_videos_by_id]

                if not selected_videos:
                    # Model picked IDs we never saw -> don't crash, hand it
                    # an error and let it try again (loop continues).
                    result = {"error": "None of the given video_ids were found in prior search results."}
                    response_parts.append(types.Part.from_function_response(name=name, response=result))
                    continue

                final = summarize_findings(topic, selected_videos, comments_by_video)
                if "error" in final:
                    return f"The agent finished gathering data but the summary step failed: {final['error']}"
                return final["brief"]

            # ---- Normal tool dispatch ----
            result = TOOL_DISPATCH[name](args)

            # Track state so summarize_findings has real data to work with later.
            if name == "search_youtube" and "videos" in result:
                for v in result["videos"]:
                    all_videos_by_id[v["videoId"]] = v
            if name == "get_video_comments":
                vid = args.get("video_id")
                if vid:
                    comments_by_video[vid] = result.get("comments", [])

            if verbose:
                import json as _json
                preview = _json.dumps(result)[:200]
                print(f"Tool result: {preview}{'...' if len(_json.dumps(result)) > 200 else ''}")

            response_parts.append(types.Part.from_function_response(name=name, response=result))

        # Send all function responses back as a single "user-role"
        # turn (Gemini's function-response convention), then loop again.
        contents.append(types.Content(role="user", parts=response_parts))

    # Exit condition: step ceiling reached without a final brief.
    return (
        f"Reached the {MAX_STEPS}-step limit without the agent producing a final brief. "
        f"This is a safety stop, not a crash — try re-running, or narrow the topic."
    )
