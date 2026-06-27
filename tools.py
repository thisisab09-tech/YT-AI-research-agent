"""
tools.py
--------
The three tools the agent can call. Each tool:
  - has exactly one job
  - has a well-defined input/output contract (see docstrings)
  - never raises an uncaught exception — on failure it returns a dict
    with an "error" key so the agent loop can decide what to do next,
    instead of crashing the whole process.

No agent-framework code lives here. This file knows nothing about the
LLM or the loop — it just wraps two YouTube Data API v3 endpoints and
one Groq chat completion call.
"""

import os
import requests
from groq import Groq

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def search_youtube(query: str) -> dict:
    """
    Tool 1: search_youtube(query)

    Calls YouTube Data API v3 /search.

    Input:
        query (str): a search term, e.g. "Claude vs GPT-4"

    Output (success):
        {
            "videos": [
                {"videoId": str, "title": str, "viewCount": int, "publishedAt": str},
                ...  up to 5 videos
            ]
        }
    Output (failure):
        {"error": "<human readable reason>"}
    """
    if not YOUTUBE_API_KEY:
        return {"error": "YOUTUBE_API_KEY is not set."}
    if not query or not query.strip():
        return {"error": "Empty query passed to search_youtube."}

    try:
        # Step 1: search returns videoIds but NOT view counts.
        search_resp = requests.get(
            YOUTUBE_SEARCH_URL,
            params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": 5,
                "order": "relevance",
                "key": YOUTUBE_API_KEY,
            },
            timeout=10,
        )
        search_resp.raise_for_status()
        items = search_resp.json().get("items", [])

        if not items:
            return {"videos": []}  # valid empty result, not an error

        video_ids = [item["id"]["videoId"] for item in items]

        # Step 2: videos.list gives us statistics (viewCount) in one batched call.
        stats_resp = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={
                "part": "statistics",
                "id": ",".join(video_ids),
                "key": YOUTUBE_API_KEY,
            },
            timeout=10,
        )
        stats_resp.raise_for_status()
        stats_by_id = {
            v["id"]: int(v.get("statistics", {}).get("viewCount", 0))
            for v in stats_resp.json().get("items", [])
        }

        videos = []
        for item in items:
            vid = item["id"]["videoId"]
            videos.append(
                {
                    "videoId": vid,
                    "title": item["snippet"]["title"],
                    "publishedAt": item["snippet"]["publishedAt"],
                    "viewCount": stats_by_id.get(vid, 0),
                }
            )

        return {"videos": videos}

    except requests.exceptions.Timeout:
        return {"error": "YouTube search timed out."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"YouTube search API error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error in search_youtube: {e}"}


def get_video_comments(video_id: str) -> dict:
    """
    Tool 2: get_video_comments(video_id)

    Calls YouTube Data API v3 /commentThreads.

    Input:
        video_id (str): a YouTube video ID, e.g. "dQw4w9WgXcQ"

    Output (success):
        {"comments": [{"text": str, "likeCount": int, "author": str}, ...]}
        (list may be EMPTY if comments are disabled or there are none —
        this is a valid, non-error result)
    Output (failure):
        {"error": "<human readable reason>"}
    """
    if not YOUTUBE_API_KEY:
        return {"error": "YOUTUBE_API_KEY is not set."}
    if not video_id:
        return {"error": "Empty video_id passed to get_video_comments."}

    try:
        resp = requests.get(
            YOUTUBE_COMMENTS_URL,
            params={
                "part": "snippet",
                "videoId": video_id,
                "order": "relevance",  # API's proxy for "top" comments
                "maxResults": 10,
                "textFormat": "plainText",
                "key": YOUTUBE_API_KEY,
            },
            timeout=10,
        )

        # Comments disabled on a video -> YouTube returns 403 with a specific reason.
        # This is an EXPECTED case, not a crash: surface it as zero comments.
        if resp.status_code == 403:
            return {"comments": [], "note": "Comments are disabled for this video."}

        resp.raise_for_status()
        items = resp.json().get("items", [])

        comments = []
        for item in items:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                {
                    "text": snippet.get("textDisplay", ""),
                    "likeCount": snippet.get("likeCount", 0),
                    "author": snippet.get("authorDisplayName", "unknown"),
                }
            )

        # Sort by likeCount desc and keep top 10 (API already caps at 10, but
        # "relevance" order isn't strictly like-count order, so we re-sort
        # to honor the spec: "top 10 comments sorted by likeCount").
        comments.sort(key=lambda c: c["likeCount"], reverse=True)
        return {"comments": comments[:10]}

    except requests.exceptions.Timeout:
        return {"error": "YouTube comments request timed out."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"YouTube comments API error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error in get_video_comments: {e}"}


def summarize_findings(topic: str, videos: list, comments_by_video: dict) -> dict:
    """
    Tool 3: summarize_findings(videos, comments)

    An LLM call (Groq) with a structured prompt. This is the only tool
    that talks to the LLM for *content generation* — separate from the
    LLM call the agent loop uses to *decide which tool to call next*.

    Input:
        topic (str): the original user topic
        videos (list[dict]): video metadata from search_youtube
        comments_by_video (dict): {videoId: [comment dicts]} from get_video_comments

    Output (success):
        {"brief": "<final natural language brief>"}
    Output (failure):
        {"error": "<human readable reason>"}
    """
    if not videos:
        return {"brief": f"No YouTube videos were found for '{topic}'. "
                          f"Try a broader or differently-worded query."}

    if not _groq_client:
        return {"error": "GROQ_API_KEY is not set."}

    # Build a compact, structured context block for the LLM rather than
    # dumping raw JSON — keeps the prompt token-efficient and readable.
    context_lines = [f"Topic: {topic}\n"]
    for v in videos:
        context_lines.append(
            f"- Video: \"{v['title']}\" (views: {v['viewCount']}, published: {v['publishedAt']})"
        )
        vid_comments = comments_by_video.get(v["videoId"], [])
        if not vid_comments:
            context_lines.append("  Comments: none available")
        else:
            for c in vid_comments[:10]:
                context_lines.append(f"  Comment ({c['likeCount']} likes): {c['text'][:200]}")
    context = "\n".join(context_lines)

    system_prompt = (
        "You are a research analyst. You are given a topic, a short list of "
        "YouTube videos about it, and top comments from those videos. "
        "Write a concise brief (150-250 words) answering: "
        "'What is the YouTube audience saying about this topic right now?' "
        "Ground every claim in the videos/comments given. Note any disagreement "
        "or mixed sentiment among commenters. Do not invent facts not present "
        "in the context. Plain prose, no markdown headers."
    )

    try:
        completion = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            temperature=0.4,
            max_tokens=500,
        )
        brief = completion.choices[0].message.content.strip()
        return {"brief": brief}

    except Exception as e:
        return {"error": f"Unexpected error in summarize_findings: {e}"}
