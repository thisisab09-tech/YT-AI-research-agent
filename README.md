# YouTube AI Research Agent

A hand-rolled agentic assistant that takes an AI-related topic, autonomously
searches YouTube, reads comments, and synthesizes a brief on what the
YouTube audience is currently saying about that topic.

No agent framework (LangChain / LlamaIndex / etc.) is used. The loop —
tool dispatch, exit conditions, error handling — is written from scratch
in `agent.py`.

## How it works (high level)

1. User enters a topic.
2. 2. The LLM (Gemini, `gemini-2.5-flash-lite`) decides to call `search_youtube`.
3. The LLM picks 2–3 relevant videos from the results.
4. The LLM calls `get_video_comments` for each chosen video.
5. The LLM calls `summarize_findings`, which triggers one final LLM call
   that writes the brief and ends the loop.

The loop is capped at 8 steps so a confused model can never spin forever.

## Setup

### 1. Get a YouTube Data API v3 key (free)

1. Go to https://console.cloud.google.com/
2. Create a new project (or select an existing one).
3. Go to **APIs & Services → Library**, search "YouTube Data API v3", click **Enable**.
4. Go to **APIs & Services → Credentials → Create Credentials → API key**.
5. Copy the key. (Optional but recommended: restrict it to "YouTube Data API v3" under API restrictions.)

Free quota: 10,000 units/day. A `search` call costs 100 units, `commentThreads` costs 1 unit — plenty for testing.

### 2. Get a Gemini API key (free)

1. Go to https://aistudio.google.com/app/apikey
2. Sign in with your normal Google account.
3. Click **Create API key**.
4. Copy the key (starts with `AIza...`).
5. Recommended: on the same page, restrict the key to the Generative Language API.

### 3. Install and configure

```bash
git clone <your-repo-url>
cd youtube-ai-agent
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste in your two keys
```

### 4. Run

```bash
python main.py
```

Then type a topic at the prompt, e.g.:

```
Topic> Claude vs GPT-4
```

## Tech used

- **Language:** Python
- **LLM:** Gemini, `gemini-2.5-flash-lite` (free tier, native function calling)
- **APIs:** YouTube Data API v3 (`/search`, `/videos`, `/commentThreads`)
- **No agent framework** — `agent.py` hand-rolls the loop using Gemini's native function-calling response format (the model returns structured `function_call` parts, which we dispatch via a plain Python dict lookup). Automatic Function Calling is explicitly disabled so the loop, state tracking, and exit conditions are all owned by our own code, not the SDK.

## Project structure

```
tools.py    — the 3 tools (search_youtube, get_video_comments, summarize_findings)
agent.py    — the hand-rolled agent loop: tool schemas, dispatch, exit conditions
main.py     — CLI entry point
```

Agent logic and tool logic are intentionally separate files: `tools.py` knows
nothing about the LLM-decision loop, and `agent.py` contains no direct
`requests` calls to YouTube — it only talks to `tools.py`'s functions and to
Gemini for the next-step decision.

## Known limitations

- **Comment ranking is an approximation.** YouTube's `commentThreads`
  endpoint doesn't support true "sort by likeCount" server-side — we request
  `order=relevance` and re-sort the returned page client-side. This is "top
  10 of what the API handed back," not "top 10 of all comments on the video."
- **No persistent memory across runs.** Each topic is a fresh conversation;
  there's no chat history or caching between sessions.
- **Single LLM provider.** Only tested against Gemini's `gemini-2.5-flash-lite`. Swapping
  providers would mean updating the function-calling schema/response handling
  in `agent.py` (OpenAI/Groq use a different tool-calling response shape:
  `choices[0].message.tool_calls` vs Gemini's `candidates[0].content.parts[].function_call`).
- **No automatic retries.** API failures are caught and surfaced as an
  `{"error": ...}` tool result so the LLM can see and react to them, but
  there's no exponential-backoff retry layer.
- **English-centric.** Comment text isn't language-filtered; non-English
  comments are passed to the summarizer as-is.

## Answers to likely walkthrough questions

**What happens if `get_video_comments` returns 0 comments?**
It's not treated as an error — `tools.py` returns `{"comments": []}` (plus a
note if comments are disabled). The agent loop passes that empty list into
the final summary step; `summarize_findings` will just say less about that
video rather than failing.

**What if the LLM tries to call a tool that doesn't exist?**
`agent.py` checks the tool name against `TOOL_DISPATCH` before calling
anything. An unknown name produces an `{"error": "Unknown tool '...'"}`
message fed back to the LLM as a tool result — the loop continues instead
of crashing, and the model gets a chance to self-correct.

**If you had to add a 4th tool, what would it be?**
`get_channel_info(channel_id)` — pulling subscriber count / channel
authority would let the agent weigh a comment/video by source credibility,
not just by view count and like count.
