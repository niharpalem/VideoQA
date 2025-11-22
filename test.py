import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from e2b import AsyncSandbox
import yt_dlp
import tempfile
import hashlib
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from datetime import timedelta
import json

load_dotenv()

st.set_page_config(page_title="Video Frame Analyzer with Notion", page_icon="🎬", layout="wide")

# ----------- GLOBAL ASYNC LOOP (STREAMLIT-SAFE) -----------

if "asyncio_loop" not in st.session_state:
    st.session_state.asyncio_loop = asyncio.new_event_loop()


def run_async(coro):
    """
    Run async function on a persistent event loop.
    Do NOT create/close loops per call – that breaks Streamlit.
    """
    loop = st.session_state.asyncio_loop
    if loop.is_closed():
        st.session_state.asyncio_loop = asyncio.new_event_loop()
        loop = st.session_state.asyncio_loop
    return loop.run_until_complete(coro)


# ----------- UTILS -----------

def download_video_locally(url):
    """Download YouTube video"""
    with st.spinner("📥 Downloading video..."):
        output_path = tempfile.mkdtemp()
        ydl_opts = {
            "format": "best[ext=mp4]",
            "outtmpl": os.path.join(output_path, "video.mp4"),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_title = info.get("title", "Unknown")
        video_path = os.path.join(output_path, "video.mp4")
        st.success("✅ Downloaded!")
        return video_path, video_title


# ----------- TAB 1 (PROCESS VIDEO) -----------

async def process_in_e2b_with_mcp(
    video_path,
    notion_page_id,
    video_source,
    source_id,
    video_url,
    video_title,
    notion_token,
):
    """Process video in E2B sandbox with MCP and save to Notion"""

    sandbox = await AsyncSandbox.create(
        api_key=os.getenv("E2B_API_KEY"),
        timeout=600,
        mcp={"notion": {"internalIntegrationToken": notion_token}},
    )

    with open(video_path, "rb") as f:
        await sandbox.files.write("/tmp/video.mp4", f.read())

    await sandbox.commands.run(
        "pip install --no-cache-dir opencv-python-headless groq python-dotenv pillow",
        timeout=180,
    )

    script = f"""
import cv2
import base64
from groq import Groq
import json
import os
from PIL import Image
import io

os.environ['GROQ_API_KEY'] = '{os.getenv("GROQ_API_KEY")}'
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

def read_frame_at_index(cap, frame_idx, target_size=(384, 384)):
    \"\"\"Read a single frame at specific index using seeking\"\"\"
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        if target_size:
            frame_pil = frame_pil.resize(target_size, Image.LANCZOS)
        return frame_pil
    return None

def pil_to_base64(pil_image):
    \"\"\"Convert PIL image to base64 string\"\"\"
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

cap = cv2.VideoCapture("/tmp/video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Sampling parameters
sample_interval_seconds = 1.5
temporal_offset_seconds = 0.8

sample_frame_interval = int(sample_interval_seconds * fps)
temporal_frame_offset = int(temporal_offset_seconds * fps)

frames_data = []
sample_count = 0

# Process frames with temporal context
for i in range(0, total_frames, sample_frame_interval):
    sample_count += 1
    
    # Get temporal context frames (before, current, after)
    before_idx = max(0, i - temporal_frame_offset)
    current_idx = i
    after_idx = min(total_frames - 1, i + temporal_frame_offset)
    
    frame_before = read_frame_at_index(cap, before_idx)
    frame_current = read_frame_at_index(cap, current_idx)
    frame_after = read_frame_at_index(cap, after_idx)
    
    if frame_before is None or frame_current is None or frame_after is None:
        continue
    
    # Convert frames to base64
    b64_before = pil_to_base64(frame_before)
    b64_current = pil_to_base64(frame_current)
    b64_after = pil_to_base64(frame_after)
    
    # Multi-frame temporal prompt
    prompt = "You are viewing 3 consecutive frames from a video (before, current, after).Describe this frame in 1–2 sentences."
    
    result = groq.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{{
            "role": "user",
            "content": [
                {{"type": "text", "text": "Frame BEFORE:"}},
                {{"type": "image_url", "image_url": {{"url": f"data:image/jpeg;base64,{{b64_before}}"}}}},
                {{"type": "text", "text": "Frame CURRENT (describe this one):"}},
                {{"type": "image_url", "image_url": {{"url": f"data:image/jpeg;base64,{{b64_current}}"}}}},
                {{"type": "text", "text": "Frame AFTER:"}},
                {{"type": "image_url", "image_url": {{"url": f"data:image/jpeg;base64,{{b64_after}}"}}}},
                {{"type": "text", "text": prompt}}
            ]
        }}],
        temperature=0.08,
        max_tokens=128
    )
    
    description = result.choices[0].message.content
    timestamp_seconds = i / fps
    
    minutes = int(timestamp_seconds // 60)
    seconds = int(timestamp_seconds % 60)
    timestamp_str = f"{{minutes:02d}}:{{seconds:02d}}"
    
    frames_data.append({{
        "frame_number": sample_count,
        "actual_frame_index": i,
        "text": description,
        "timestamp": timestamp_str,
        "timestamp_seconds": timestamp_seconds
    }})

cap.release()

output_data = {{
    "video_source": "{video_source}",
    "video_id": "{source_id}",
    "video_title": "{video_title}",
    "video_url": "{video_url}",
    "total_frames_analyzed": len(frames_data),
    "frames": frames_data
}}

filename = f"/tmp/video_{{'{source_id}'}}.json"
with open(filename, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"SUCCESS {{filename}}")
"""

    await sandbox.files.write("/tmp/process.py", script)
    result = await sandbox.commands.run("python3 /tmp/process.py", timeout=600)
    processing_output = result.stdout

    # MCP writing logic 
    mcp_url = sandbox.get_mcp_url()
    mcp_token = await sandbox.get_mcp_token()

    async with streamablehttp_client(
        url=mcp_url,
        headers={"Authorization": f"Bearer {mcp_token}"},
        timeout=timedelta(seconds=600),
    ) as (read_stream, write_stream, _):

        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            json_content = await sandbox.files.read(f"/tmp/video_{source_id}.json")
            data = json.loads(json_content)

            notion_blocks = []
            notion_blocks.append(
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": f"📹 {data['video_title']}"},
                            }
                        ]
                    },
                }
            )

            for frame in data["frames"]:
                frame_text = frame["text"][:1900]
                notion_blocks.extend(
                    [
                        {
                            "object": "block",
                            "type": "heading_3",
                            "heading_3": {
                                "rich_text": [
                                    {
                                        "type": "text",
                                        "text": {
                                            "content": f"Frame {frame['frame_number']} @ {frame['timestamp']}"
                                        },
                                    }
                                ]
                            },
                        },
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [
                                    {
                                        "type": "text",
                                        "text": {"content": frame_text},
                                    }
                                ]
                            },
                        },
                    ]
                )

            # NOTE: We use the 'auto-find' approach here as well to be safe, 
            # or rely on the previously working assumption. 
            # For stability, I'll stick to a hardcoded attempt but wrap it.
            # Ideally, we should use list_tools here too, but Tab 1 was reported working.
            for i in range(0, len(notion_blocks), 50):
                batch = notion_blocks[i : i + 50]
                # Try standard name first, then prefixed
                try:
                    await session.call_tool(
                        "API-patch-block-children",
                        arguments={"block_id": notion_page_id, "children": batch},
                    )
                except:
                    await session.call_tool(
                        "notion-API-patch-block-children",
                        arguments={"block_id": notion_page_id, "children": batch},
                    )

    try:
        await sandbox.kill()
    except Exception:
        pass

    return processing_output


# ------------- TAB 2 — FETCH NOTION VIA LIGHT SANDBOX (FIXED) -------------

async def fetch_notion_context_via_sandbox(notion_page_id, notion_token):
    """
    Spins up a tiny E2B sandbox, connects to Notion MCP,
    and tries to fetch page content using automatic tool discovery loop.
    """
    
    sandbox = await AsyncSandbox.create(
        api_key=os.getenv("E2B_API_KEY"),
        timeout=60, 
        mcp={"notion": {"internalIntegrationToken": notion_token}},
    )

    mcp_url = sandbox.get_mcp_url()
    mcp_token = await sandbox.get_mcp_token()

    notion_context_data = []

    async with streamablehttp_client(
        url=mcp_url,
        headers={"Authorization": f"Bearer {mcp_token}"},
        timeout=timedelta(seconds=45),
    ) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # 1. Get the authoritative list of tools from the server
            try:
                tools_result = await session.list_tools()
                available_tools = [t.name for t in tools_result.tools]
                # print(f"Available tools: {available_tools}")
            except Exception as e:
                return f"Error listing tools: {e}"

            # 2. Define our strategies to get data
            # Each strategy has a list of 'possible name signatures' and the argument it needs.
            strategies = [
                {
                    "name": "Get Block Children (Content)",
                    "arg_key": "block_id",
                    "signatures": ["API-get-block-children", "get-block-children", "retrieve-block-children"]
                },
                {
                    "name": "Retrieve Page (Metadata)",
                    "arg_key": "page_id",
                    "signatures": ["API-retrieve-a-page", "retrieve-a-page", "get-page"]
                }
            ]

            data_found = False

            # 3. Loop through strategies
            for strategy in strategies:
                # Find the best matching tool for this strategy
                selected_tool = None
                
                # Check exact matches first
                for tool in available_tools:
                    if tool in strategy["signatures"]:
                        selected_tool = tool
                        break
                
                # If no exact match, check fuzzy match (contains string)
                if not selected_tool:
                    for tool in available_tools:
                        for sig in strategy["signatures"]:
                            if sig in tool:
                                selected_tool = tool
                                break
                        if selected_tool: break
                
                # Execute if we found a tool
                if selected_tool:
                    try:
                        result = await session.call_tool(
                            selected_tool,
                            arguments={strategy["arg_key"]: notion_page_id},
                        )
                        
                        if hasattr(result, "content"):
                            for item in result.content:
                                if hasattr(item, "text") and item.text:
                                    notion_context_data.append(f"--- {strategy['name']} ---\n{item.text}")
                                    data_found = True
                    except Exception as e:
                        notion_context_data.append(f"Failed calling {selected_tool}: {str(e)}")

    try:
        await sandbox.kill()
    except Exception:
        pass

    final_text = "\n\n".join(notion_context_data)
    
    if not final_text or len(final_text) < 50:
        error_msg = (
            f"⚠️ SYSTEM_ERROR: Could not retrieve valid data.\n"
            f"Tools Found on Server: {available_tools}\n\n"
            f"Logs:\n{final_text}\n"
            "--------------------------------------------------\n"
            "🔴 Troubleshooting:\n"
            "1. Check Page ID.\n"
            "2. Ensure Integration is invited to the page.\n"
        )
        return error_msg
        
    return final_text


# ----------- STREAMLIT UI -----------

st.title("🎬 Video Frame Analyzer with Notion")

with st.sidebar:
    st.header("⚙️ Settings")
    notion_page_id = st.text_input(
        "Notion Page ID", value=os.getenv("NOTION_PAGE_ID", "")
    )
    notion_token = st.text_input(
        "Notion Token",
        value=os.getenv("NOTION_INTEGRATION_TOKEN", ""),
        type="password",
    )

tab1, tab2 = st.tabs(["📹 Upload & Process", "💬 Chat"])


# -------- TAB 1 (PROCESS) --------

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Video", type=["mp4", "mov", "avi"]
        )
    with col2:
        youtube_url = st.text_input("Or YouTube URL")

    if st.button("🚀 Process Video", type="primary", use_container_width=True):
        if not notion_page_id or not notion_token:
            st.error("❌ Enter Notion credentials in sidebar")
        else:
            video_path = None
            video_source = None
            video_title = "Unknown"
            video_url = ""

            if uploaded_file:
                temp_path = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ).name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                video_path = temp_path
                video_source = "local"
                video_title = uploaded_file.name
                st.video(uploaded_file)

                source_id = hashlib.sha256(
                    f"{uploaded_file.name}{os.path.getsize(video_path)}".encode()
                ).hexdigest()[:16]

            elif youtube_url:
                video_path, video_title = download_video_locally(youtube_url)
                video_source = "youtube"
                video_url = youtube_url
                st.video(video_path)

                source_id = hashlib.sha256(youtube_url.encode()).hexdigest()[:16]

            else:
                st.error("❌ Provide a video")
                st.stop()

            try:
                with st.spinner("🔧 Processing in E2B..."):
                    output = run_async(
                        process_in_e2b_with_mcp(
                            video_path,
                            notion_page_id,
                            video_source,
                            source_id,
                            video_url,
                            video_title,
                            notion_token,
                        )
                    )
                st.success("✅ Done!")
                st.code(output)
                if "SUCCESS" in output:
                    st.balloons()
            except Exception as e:
                st.error(f"❌ {e}")


# -------- TAB 2 (CHAT) --------

with tab2:
    st.header("💬 Chat with Video Knowledge")

    if not notion_page_id or not notion_token:
        st.warning("⚠️ Enter Notion credentials in sidebar")
        st.stop()

    st.info("🤖 MCP via lightweight E2B sandbox + Local Groq LLM (fast)")

    # Chat history memory (C2 mode)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render chat history
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            st.write(turn["a"])

    user_query = st.chat_input("Ask anything about the analyzed video...")

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Reading Notion + Running LLM..."):
                # Fetch Notion context using LIGHT sandbox
                notion_context = run_async(
                    fetch_notion_context_via_sandbox(notion_page_id, notion_token)
                )

                # Local Groq LLM call
                from groq import Groq
                groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

                try:
                    # STEP 1: Split Notion context into chunks
                    # Each chunk should be ~5000 chars (~1250 tokens) to stay safe
                    chunk_size = 5000
                    
                    # Split context into chunks
                    chunks = []
                    for i in range(0, len(notion_context), chunk_size):
                        chunk = notion_context[i:i + chunk_size]
                        chunks.append(chunk)
                    
                    st.info(f"📦 Processing {len(chunks)} chunks of data...")
                    
                    # STEP 2: Process each chunk and extract relevant information
                    chunk_results = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, chunk in enumerate(chunks):
                        status_text.text(f"Processing chunk {idx + 1}/{len(chunks)}...")
                        progress_bar.progress((idx + 1) / len(chunks))
                        
                        # Ask each chunk: "Is there relevant info here?"
                        chunk_prompt = f"""You are analyzing video frames. Read this chunk of video data and answer:

Is there information relevant to this question: "{user_query}"?

If YES, extract the relevant information with timestamps.
If NO, respond with "NO_RELEVANT_INFO".

VIDEO DATA CHUNK:
{chunk}

Response (include timestamps if relevant):"""

                        try:
                            chunk_response = groq_client.chat.completions.create(
                                model="llama-3.1-8b-instant",
                                messages=[
                                    {"role": "user", "content": chunk_prompt}
                                ],
                                temperature=0.2,
                                max_tokens=300,
                            )
                            
                            result = chunk_response.choices[0].message.content
                            
                            # Only keep relevant chunks
                            if "NO_RELEVANT_INFO" not in result.upper():
                                chunk_results.append({
                                    'chunk_id': idx + 1,
                                    'content': result
                                })
                        
                        except Exception as e:
                            st.warning(f"⚠️ Error processing chunk {idx + 1}: {str(e)}")
                            continue
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # STEP 3: Combine all relevant chunk results
                    if not chunk_results:
                        st.warning("No relevant information found in the video data.")
                        final_answer = "I couldn't find information relevant to your question in the video analysis. Please try rephrasing your question or process a different video."
                    else:
                        st.success(f"✅ Found relevant info in {len(chunk_results)} chunk(s)")
                        
                        # Combine relevant findings
                        combined_info = "\n\n".join([
                            f"Section {r['chunk_id']}:\n{r['content']}" 
                            for r in chunk_results[:5]  # Limit to top 5 chunks
                        ])
                        
                        # Build chat history (minimal)
                        history_text = ""
                        if len(st.session_state.chat_history) > 0:
                            last_turn = st.session_state.chat_history[-1]
                            history_text = f"Previous Q&A:\nQ: {last_turn['q'][:80]}\nA: {last_turn['a'][:120]}\n\n"
                        
                        # STEP 4: Generate final answer from combined relevant info
                        final_prompt = f"""You are a video analysis assistant. Based on the relevant information extracted from the video, provide a comprehensive answer.

IMPORTANT: Always cite timestamps (MM:SS format) when describing what happens in the video.

{history_text}RELEVANT VIDEO INFORMATION:
{combined_info}

USER QUESTION:
{user_query}

Final Answer (cite timestamps):"""

                        # Safety check
                        if len(final_prompt) / 4 > 5000:  # Estimate tokens
                            combined_info = combined_info[:16000]
                            final_prompt = f"""Based on this video info, answer the question. Cite timestamps.

{combined_info}

Q: {user_query}

A:"""
                        
                        final_response = groq_client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Provide comprehensive answers based on video information. Always cite timestamps.",
                                },
                                {"role": "user", "content": final_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=600,
                        )
                        
                        final_answer = final_response.choices[0].message.content
                    
                    # Display and save answer
                    st.write(final_answer)
                    
                    st.session_state.chat_history.append(
                        {"q": user_query, "a": final_answer}
                    )

                except Exception as e:
                    st.error(f"⚠️ LLM Error: {str(e)}")
                    st.info("💡 Tip: If you see rate limit errors, wait a few seconds and try again.")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("---")
st.caption("Built with E2B • Notion MCP • Groq • Streamlit")