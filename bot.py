import logging
import os
import json
import random
import shutil # For /imagine
import base64
import mimetypes
import asyncio
import re # For sanitization and new markdown, and calculator tool
import traceback # For detailed error handler
import html    # For detailed error handler
import httpx   # For specific exception types
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Set, Union, Any

# New imports for additional tools
import math
from datetime import datetime, timedelta, timezone as dt_timezone # Added timezone for startup timestamp
import pytz
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote as url_unquote


# Telegram imports
from telegram import Update, InputMediaPhoto, Message as TelegramMessage, User as TelegramUser, Chat, Voice, Bot, BotCommand, ChatPermissions
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, Application
)
from telegram.constants import ChatAction, ParseMode
import telegram.error

# OpenAI imports
from openai import AsyncOpenAI, InternalServerError, APITimeoutError # Added InternalServerError, APITimeoutError
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_message import ChatCompletionMessage

# --- START OF MARKDOWNV2 ESCAPE CODE ---
_MD_SPECIAL_CHARS_TO_ESCAPE_GENERAL_LIST = r"_*\[\]()~`#+\-=|{}.!"
_MD_ESCAPE_REGEX_GENERAL = re.compile(r"([%s])" % re.escape(_MD_SPECIAL_CHARS_TO_ESCAPE_GENERAL_LIST))
_PLACEHOLDER_PREFIX = "zYzTgMdPhPrfxzYz"
_PLACEHOLDER_SUFFIX = "zYzTgMdPhSffxzYz"
_placeholder_regex_global = re.compile(f"{re.escape(_PLACEHOLDER_PREFIX)}(\\d+){re.escape(_PLACEHOLDER_SUFFIX)}")

def _escape_general_markdown_chars(text: str) -> str:
    return _MD_ESCAPE_REGEX_GENERAL.sub(r"\\\1", text)

_styled_text_definitions: List[Tuple[str, str, str, str]] = [
    ("spoiler", r"(\|\|)", r"(\|\|)", r"\|\|"),
    ("underline", r"(__)", r"(__)", r"__"),
    ("bold", r"(\*)", r"(\*)", r"\*"),
    ("italic", r"(_)", r"(_)(?!_)", r"_"),
    ("strikethrough", r"(~)", r"(~)(?!~)", r"~"),
]
_COMPILED_STYLED_TEXT_PATTERNS: List[Tuple[str, re.Pattern, re.Pattern, re.Pattern]] = []
for name, open_delim_re_str, close_delim_re_str, literal_delim_for_content in _styled_text_definitions:
    content_re_str = rf"((?!\s)(?:(?!(?<!\\){re.escape(literal_delim_for_content)}).)+?(?<!\s))"
    full_pattern = re.compile(open_delim_re_str + content_re_str + close_delim_re_str, re.DOTALL)
    _COMPILED_STYLED_TEXT_PATTERNS.append(
        (name, full_pattern, re.compile(open_delim_re_str), re.compile(close_delim_re_str))
    )
_CODE_BLOCK_LANG_RE = r"[\w_#+.-]+"

class MarkdownV2ParserContext:
    def __init__(self):
        self.placeholders_map: Dict[str, str] = {}
        self.placeholder_idx: int = 0
    def _get_next_placeholder_key(self) -> str:
        self.placeholder_idx += 1
        return f"{_PLACEHOLDER_PREFIX}{self.placeholder_idx}{_PLACEHOLDER_SUFFIX}"
    def add_placeholder(self, content_to_store: str) -> str:
        ph_key = self._get_next_placeholder_key()
        self.placeholders_map[ph_key] = content_to_store
        return ph_key

def _telegram_markdown_v2_escape_recursive(
    current_text: str, ctx: MarkdownV2ParserContext, active_style_type: Optional[str] = None
) -> str:
    if not current_text: return ""

    code_block_re = re.compile(rf"(?s)```(?:({_CODE_BLOCK_LANG_RE})\n)?(.*?)```")
    def replace_code_block(match: re.Match) -> str:
        language, code_content = match.group(1), match.group(2)
        escaped_code_for_telegram = code_content.replace("\\", "\\\\").replace("`", "\\`")
        placeholder_value = f"```{language}\n{escaped_code_for_telegram}```" if language else f"```{escaped_code_for_telegram}```"
        return ctx.add_placeholder(placeholder_value)
    current_text = code_block_re.sub(replace_code_block, current_text)

    inline_code_re = re.compile(r"`(.+?)`")
    def replace_inline_code(match: re.Match) -> str:
        content = match.group(1)
        escaped_content_for_telegram = content.replace("\\", "\\\\").replace("`", "\\`")
        return ctx.add_placeholder(f"`{escaped_content_for_telegram}`")
    current_text = inline_code_re.sub(replace_inline_code, current_text)

    link_re = re.compile(r"\[((?:[^\[\]]|\\\[|\\\])*?)\]\(((?:[^()\s\\]|\\.)*?)\)")
    processed_segments_for_links: List[str] = []
    last_end_idx_links = 0
    for m_link in link_re.finditer(current_text):
        processed_segments_for_links.append(current_text[last_end_idx_links:m_link.start()])
        link_text_raw, url_raw = m_link.group(1), m_link.group(2)
        if not url_raw:
             processed_segments_for_links.append(_escape_general_markdown_chars(m_link.group(0)))
        else:
            escaped_url_for_telegram = url_raw.replace("\\", "\\\\").replace(")", "\\)")
            recursively_escaped_link_text = _telegram_markdown_v2_escape_recursive(link_text_raw, ctx, None)
            processed_segments_for_links.append(ctx.add_placeholder(f"[{recursively_escaped_link_text}]({escaped_url_for_telegram})"))
        last_end_idx_links = m_link.end()
    processed_segments_for_links.append(current_text[last_end_idx_links:])
    current_text = "".join(processed_segments_for_links)

    for style_name, full_pattern_re, _, _ in _COMPILED_STYLED_TEXT_PATTERNS:
        if style_name == active_style_type: continue
        new_segments_styled: List[str] = []
        last_idx_styled = 0
        for m_style in full_pattern_re.finditer(current_text):
            new_segments_styled.append(current_text[last_idx_styled:m_style.start()])
            opening_delimiter, content_raw, closing_delimiter = m_style.group(1), m_style.group(2), m_style.group(3)
            recursively_escaped_content = _telegram_markdown_v2_escape_recursive(content_raw, ctx, active_style_type=style_name)
            new_segments_styled.append(ctx.add_placeholder(f"{opening_delimiter}{recursively_escaped_content}{closing_delimiter}"))
            last_idx_styled = m_style.end()
        new_segments_styled.append(current_text[last_idx_styled:])
        current_text = "".join(new_segments_styled)

    final_segments: List[str] = []
    last_idx_final = 0
    for m_placeholder in _placeholder_regex_global.finditer(current_text):
        final_segments.append(_escape_general_markdown_chars(current_text[last_idx_final:m_placeholder.start()]))
        final_segments.append(m_placeholder.group(0))
        last_idx_final = m_placeholder.end()
    final_segments.append(_escape_general_markdown_chars(current_text[last_idx_final:]))
    return "".join(final_segments)

def telegram_markdown_v2_escape(text: str) -> str:
    if not text: return ""
    text = re.sub(r'<a?:[a-zA-Z0-9_]+:[0-9]+>', '', text) # Remove custom emojis
    ctx = MarkdownV2ParserContext()
    processed_text = _telegram_markdown_v2_escape_recursive(text, ctx)

    max_restoration_loops = len(ctx.placeholders_map) * 2 + 10
    count = 0
    temp_text = processed_text
    while True:
        restored_text = _placeholder_regex_global.sub(lambda m: ctx.placeholders_map.get(m.group(0), m.group(0)), temp_text)
        if restored_text == temp_text: break
        temp_text = restored_text
        count += 1
        if count >= max_restoration_loops:
            if 'logger' in globals(): logger.warning(f"MarkdownV2: Max placeholder restoration loops ({max_restoration_loops}). Finalizing.")
            break
    return restored_text
# --- END OF MARKDOWNV2 ESCAPE CODE ---

# --- START OF INTELLIGENT SPLITTING CODE ---
_BALANCING_MARKDOWN_DELIMITERS = sorted(['||', '__', '*', '_', '~'], key=len, reverse=True)
_MAX_DELIMITER_SEQUENCE_LEN = sum(len(d) for d in _BALANCING_MARKDOWN_DELIMITERS)
_ATOMIC_ENTITY_REGEXES = [
    re.compile(r"(?s)```(?:[\w_#+.-]*\n)?.*?```"),
    re.compile(r"`(?:\\.|[^`\n])+?`"),
    re.compile(r"\[(?:[^\[\]]|\\\[|\\\])*?\]\((?:[^()\s\\]|\\.)*?\)")
]

def _is_char_escaped_at_pos(text: str, char_pos: int) -> bool:
    if char_pos == 0: return False
    num_backslashes = 0
    k = char_pos - 1
    while k >= 0 and text[k] == '\\':
        num_backslashes += 1; k -= 1
    return num_backslashes % 2 == 1

def _scan_segment_and_update_style_stack(initial_stack: List[str], content_segment: str, lgr: logging.Logger) -> List[str]:
    current_stack = list(initial_stack)
    idx = 0
    while idx < len(content_segment):
        found_atomic_block = False
        for atomic_re in _ATOMIC_ENTITY_REGEXES:
            match = atomic_re.match(content_segment, idx)
            if match:
                idx += len(match.group(0)); found_atomic_block = True; break
        if found_atomic_block: continue
        matched_delimiter = None
        if not _is_char_escaped_at_pos(content_segment, idx):
            for delim_str in _BALANCING_MARKDOWN_DELIMITERS:
                if content_segment.startswith(delim_str, idx):
                    if current_stack and current_stack[-1] == delim_str: current_stack.pop()
                    else: current_stack.append(delim_str)
                    matched_delimiter = delim_str; break
        if matched_delimiter: idx += len(matched_delimiter)
        else: idx += 1
    return current_stack

def get_safe_segment_len(full_text: str, segment_start_pos: int, desired_max_len: int, lgr: logging.Logger) -> int:
    if desired_max_len <= 0: return 0
    current_safe_len = desired_max_len
    for atomic_re in _ATOMIC_ENTITY_REGEXES:
        for match in atomic_re.finditer(full_text):
            entity_abs_start, entity_abs_end = match.start(), match.end()
            current_segment_abs_end = segment_start_pos + current_safe_len
            if entity_abs_start < current_segment_abs_end and entity_abs_end > current_segment_abs_end:
                current_safe_len = max(0, entity_abs_start - segment_start_pos)
            if entity_abs_start >= segment_start_pos + desired_max_len + 100: break
    return current_safe_len

def split_message_with_markdown_balancing(escaped_text: str, max_part_len: int, lgr: logging.Logger) -> List[str]:
    final_parts: List[str] = []
    current_pos = 0
    active_styles_at_part_start: List[str] = []
    if not escaped_text: return [""]

    while current_pos < len(escaped_text):
        prefix = "".join(active_styles_at_part_start)
        max_suffix_len_estimate = len(prefix) if len(prefix) <= _MAX_DELIMITER_SEQUENCE_LEN else _MAX_DELIMITER_SEQUENCE_LEN
        content_len_budget = max(20, max_part_len - len(prefix) - max_suffix_len_estimate)
        effective_content_len_budget = min(content_len_budget, len(escaped_text) - current_pos)
        actual_content_segment_len = get_safe_segment_len(escaped_text, current_pos, effective_content_len_budget, lgr)

        if actual_content_segment_len == 0 and effective_content_len_budget > 0 and current_pos < len(escaped_text):
            taken_fallback_len = 0
            for atomic_re in _ATOMIC_ENTITY_REGEXES:
                match = atomic_re.match(escaped_text, current_pos)
                if match: taken_fallback_len = len(match.group(0)); break
            if taken_fallback_len == 0 and current_pos < len(escaped_text): taken_fallback_len = 1
            actual_content_segment_len = taken_fallback_len

        current_content_segment = escaped_text[current_pos : current_pos + actual_content_segment_len]
        temp_current_content_segment = current_content_segment
        styles_after_segment_content: List[str] = []

        while True:
            styles_after_segment_content = _scan_segment_and_update_style_stack(active_styles_at_part_start, temp_current_content_segment, lgr)
            suffix = "".join(reversed(styles_after_segment_content))
            final_part_candidate = prefix + temp_current_content_segment + suffix
            if len(final_part_candidate) <= max_part_len:
                current_content_segment = temp_current_content_segment; break
            if not temp_current_content_segment:
                styles_after_segment_content = _scan_segment_and_update_style_stack(active_styles_at_part_start, current_content_segment, lgr)
                break

            prev_newline_idx = temp_current_content_segment.rfind('\n', 0, len(temp_current_content_segment) -1)
            if prev_newline_idx != -1:
                safe_trimmed_len = get_safe_segment_len(escaped_text, current_pos, prev_newline_idx + 1, lgr)
                if safe_trimmed_len < len(temp_current_content_segment):
                    temp_current_content_segment = escaped_text[current_pos : current_pos + safe_trimmed_len]; continue
            prev_space_idx = temp_current_content_segment.rfind(' ', 0, len(temp_current_content_segment) -1)
            if prev_space_idx != -1:
                safe_trimmed_len = get_safe_segment_len(escaped_text, current_pos, prev_space_idx + 1, lgr)
                if safe_trimmed_len < len(temp_current_content_segment):
                    temp_current_content_segment = escaped_text[current_pos : current_pos + safe_trimmed_len]; continue

            potential_new_len = len(temp_current_content_segment) - 1
            if potential_new_len <= 0:
                styles_after_segment_content = _scan_segment_and_update_style_stack(active_styles_at_part_start, current_content_segment, lgr); break
            safe_trimmed_len = get_safe_segment_len(escaped_text, current_pos, potential_new_len, lgr)
            if safe_trimmed_len < len(temp_current_content_segment):
                temp_current_content_segment = escaped_text[current_pos : current_pos + safe_trimmed_len]
            else:
                styles_after_segment_content = _scan_segment_and_update_style_stack(active_styles_at_part_start, current_content_segment, lgr); break

        final_part_to_add = prefix + current_content_segment + "".join(reversed(styles_after_segment_content))
        if final_part_to_add.strip(): final_parts.append(final_part_to_add)
        elif not final_parts and not escaped_text: final_parts.append("")

        active_styles_at_part_start = list(styles_after_segment_content)
        if len(current_content_segment) > 0: current_pos += len(current_content_segment)
        elif current_pos < len(escaped_text): lgr.error(f"SplitLoop PROGRESS: Empty segment. Advancing by 1."); current_pos += 1
    return [p for p in final_parts if p is not None]
# --- END OF INTELLIGENT SPLITTING CODE ---

# --- Global Config & Setup ---
try:
    from BingImageCreator import ImageGen
    BING_IMAGE_CREATOR_AVAILABLE = True
except ImportError:
    BING_IMAGE_CREATOR_AVAILABLE = False
    ImageGen = None

load_dotenv()
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
SHAPESINC_API_KEY = os.getenv("SHAPESINC_API_KEY")
SHAPESINC_SHAPE_USERNAME = os.getenv("SHAPESINC_SHAPE_USERNAME")
ALLOWED_USERS_STR = os.getenv("ALLOWED_USERS", "")
ALLOWED_CHATS_STR = os.getenv("ALLOWED_CHATS", "")
BING_AUTH_COOKIE = os.getenv("BING_AUTH_COOKIE") if BING_IMAGE_CREATOR_AVAILABLE else None
ENABLE_TOOL_USE = os.getenv("ENABLE_TOOL_USE", "false").lower() == "true"
SHAPES_API_BASE_URL = os.getenv("SHAPES_API_BASE_URL", "https://api.shapes.inc/v1/")

GROUP_FREE_WILL_ENABLED = os.getenv("GROUP_FREE_WILL_ENABLED", "false").lower() == "true"
GROUP_FREE_WILL_PROBABILITY_STR = os.getenv("GROUP_FREE_WILL_PROBABILITY", "0.0")
GROUP_FREE_WILL_CONTEXT_MESSAGES_STR = os.getenv("GROUP_FREE_WILL_CONTEXT_MESSAGES", "3")

IGNORE_OLD_MESSAGES_ON_STARTUP = os.getenv("IGNORE_OLD_MESSAGES_ON_STARTUP", "false").lower() == "true"
BOT_STARTUP_TIMESTAMP: Optional[datetime] = None # Will be set in main

try:
    GROUP_FREE_WILL_PROBABILITY = float(GROUP_FREE_WILL_PROBABILITY_STR)
    if not (0.0 <= GROUP_FREE_WILL_PROBABILITY <= 1.0):
        raise ValueError("GROUP_FREE_WILL_PROBABILITY must be between 0.0 and 1.0")
except ValueError as e:
    logging.error(f"Invalid GROUP_FREE_WILL_PROBABILITY: {e}. Disabling free will.")
    GROUP_FREE_WILL_PROBABILITY = 0.0
    GROUP_FREE_WILL_ENABLED = False

try:
    GROUP_FREE_WILL_CONTEXT_MESSAGES = int(GROUP_FREE_WILL_CONTEXT_MESSAGES_STR)
    if not (0 <= GROUP_FREE_WILL_CONTEXT_MESSAGES <= 20): # Max 20 for sanity
        raise ValueError("GROUP_FREE_WILL_CONTEXT_MESSAGES must be between 0 and 20.")
except ValueError as e:
    logging.warning(f"Invalid GROUP_FREE_WILL_CONTEXT_MESSAGES: {e}. Defaulting to 3.")
    GROUP_FREE_WILL_CONTEXT_MESSAGES = 3

SHAPES_API_CLIENT_TIMEOUT = httpx.Timeout(90.0, connect=5.0, read=85.0)
HTTP_CLIENT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

if not TELEGRAM_TOKEN: raise ValueError("BOT_TOKEN not set in environment.")
if not SHAPESINC_API_KEY: raise ValueError("SHAPESINC_API_KEY not set in environment.")
if not SHAPESINC_SHAPE_USERNAME: raise ValueError("SHAPESINC_SHAPE_USERNAME not set in environment.")

ALLOWED_USERS = [user.strip() for user in ALLOWED_USERS_STR.split(",") if user.strip()]
ALLOWED_CHATS = [chat.strip() for chat in ALLOWED_CHATS_STR.split(",") if chat.strip()]

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

try:
    aclient_shape = AsyncOpenAI(
        api_key=SHAPESINC_API_KEY,
        base_url=SHAPES_API_BASE_URL,
        timeout=SHAPES_API_CLIENT_TIMEOUT
    )
    logger.info(f"Shapes client init: {SHAPESINC_SHAPE_USERNAME} at {SHAPES_API_BASE_URL} with timeout {SHAPES_API_CLIENT_TIMEOUT}")
    logger.info(f"Tool use: {'ENABLED' if ENABLE_TOOL_USE else 'DISABLED'}.")
    if GROUP_FREE_WILL_ENABLED:
        logger.info(f"Group Free Will: ENABLED with probability {GROUP_FREE_WILL_PROBABILITY:.2%} and context of {GROUP_FREE_WILL_CONTEXT_MESSAGES} messages.")
    else:
        logger.info("Group Free Will: DISABLED.")
    logger.info(f"Ignore old messages on startup: {'ENABLED' if IGNORE_OLD_MESSAGES_ON_STARTUP else 'DISABLED'}")
except Exception as e:
    logger.error(f"Failed to init Shapes client: {e}"); raise

# Stores conversation history per chat_id and thread_id tuple
chat_histories: dict[tuple[int, Optional[int]], list[ChatCompletionMessageParam]] = {}
MAX_HISTORY_LENGTH = 10
# Stores raw messages (sender/text) per group chat_id and thread_id for free will context
group_raw_message_log: Dict[int, Dict[Optional[int], List[Dict[str, str]]]] = {}
MAX_RAW_LOG_PER_THREAD = 50 # Limit raw log size per topic/thread
# Stores chat_id/thread_id tuples where the bot should always be active
activated_chats_topics: Set[Tuple[int, Optional[int]]] = set()
# --- Topic Name Cache ---
topic_names_cache: Dict[Tuple[int, int], str] = {} # (chat_id, thread_id) -> topic_name
MAX_TOPIC_ENTRIES_PER_CHAT_IN_CACHE = 200 
chat_topic_cache_keys_order: Dict[int, List[Tuple[int, int]]] = {} 
# --- END OF Global Config & Setup ---

# --- NEW HELPER FUNCTION for sending messages with thread fallback ---
async def send_message_to_chat_or_general(
    bot_instance: Bot,
    chat_id: int,
    text: str,
    preferred_thread_id: Optional[int],
    parse_mode: Optional[str] = None,
    **kwargs  # For other send_message params like reply_markup, disable_web_page_preview
) -> Optional[TelegramMessage]:
    """
    Attempts to send a message to a preferred thread, falling back to the general chat
    if the preferred thread is not found.
    """
    try:
        return await bot_instance.send_message(
            chat_id=chat_id,
            text=text,
            message_thread_id=preferred_thread_id,
            parse_mode=parse_mode,
            **kwargs
        )
    except telegram.error.BadRequest as e:
        if "message thread not found" in e.message.lower() and preferred_thread_id is not None:
            logger.warning(
                f"Chat {chat_id}: Preferred thread {preferred_thread_id} not found for message. "
                f"Attempting to send to general chat instead. Error: {e}"
            )
            try:
                # Fallback: send to general chat (message_thread_id=None)
                return await bot_instance.send_message(
                    chat_id=chat_id,
                    text=text,
                    message_thread_id=None, # Explicitly None for general chat
                    parse_mode=parse_mode,
                    **kwargs
                )
            except telegram.error.BadRequest as e2:
                # If sending to general also fails with BadRequest (could be for other reasons now)
                logger.error(f"Chat {chat_id}: Sending to general chat also failed with BadRequest. Error: {e2}")
                raise e2 # Re-raise the second error
            except Exception as e_general_unexpected:
                logger.error(f"Chat {chat_id}: Unexpected error sending to general chat. Error: {e_general_unexpected}")
                raise e_general_unexpected # Re-raise
        else:
            # Different BadRequest error, or preferred_thread_id was already None
            raise e # Re-raise original error
    except Exception as e_other_send:
        # Catch any other non-BadRequest errors during send
        logger.error(f"Chat {chat_id} (thread {preferred_thread_id}): Unexpected error during send_message. Error: {e_other_send}")
        raise e_other_send
# --- END OF NEW HELPER FUNCTION ---

# Add helper functions for media with fallback logic
async def send_photo_to_chat_or_general(
    bot_instance: Bot,
    chat_id: int,
    photo: Union[str, bytes],
    preferred_thread_id: Optional[int],
    **kwargs
) -> Optional[TelegramMessage]:
    try:
        return await bot_instance.send_photo(
            chat_id=chat_id,
            photo=photo,
            message_thread_id=preferred_thread_id,
            **kwargs
        )
    except telegram.error.BadRequest as e:
        if "message thread not found" in e.message.lower() and preferred_thread_id is not None:
            logger.warning(
                f"Chat {chat_id}: Preferred thread {preferred_thread_id} not found for photo. "
                f"Attempting to send to general chat instead. Error: {e}"
            )
            try:
                # Fallback: send to general chat (message_thread_id=None)
                return await bot_instance.send_photo(
                    chat_id=chat_id,
                    photo=photo,
                    message_thread_id=None,
                    **kwargs
                )
            except Exception as e2:
                logger.error(f"Chat {chat_id}: Sending photo to general chat also failed. Error: {e2}")
                raise e2
        else:
            raise e
    except Exception as e_other:
        logger.error(f"Chat {chat_id} (thread {preferred_thread_id}): Unexpected error during send_photo. Error: {e_other}")
        raise e_other

async def send_audio_to_chat_or_general(
    bot_instance: Bot,
    chat_id: int,
    audio: Union[str, bytes],
    preferred_thread_id: Optional[int],
    **kwargs
) -> Optional[TelegramMessage]:
    try:
        return await bot_instance.send_audio(
            chat_id=chat_id,
            audio=audio,
            message_thread_id=preferred_thread_id,
            **kwargs
        )
    except telegram.error.BadRequest as e:
        if "message thread not found" in e.message.lower() and preferred_thread_id is not None:
            logger.warning(
                f"Chat {chat_id}: Preferred thread {preferred_thread_id} not found for audio. "
                f"Attempting to send to general chat instead. Error: {e}"
            )
            try:
                # Fallback: send to general chat (message_thread_id=None)
                return await bot_instance.send_audio(
                    chat_id=chat_id,
                    audio=audio,
                    message_thread_id=None,
                    **kwargs
                )
            except Exception as e2:
                logger.error(f"Chat {chat_id}: Sending audio to general chat also failed. Error: {e2}")
                raise e2
        else:
            raise e
    except Exception as e_other:
        logger.error(f"Chat {chat_id} (thread {preferred_thread_id}): Unexpected error during send_audio. Error: {e_other}")
        raise e_other

# --- TOOL IMPLEMENTATIONS ---
async def create_poll_tool(
    question: str,
    options: List[str],
    is_anonymous: bool = True,
    allows_multiple_answers: bool = False,
    # Parameters to be passed by the main handler:
    telegram_bot_context: Optional[ContextTypes.DEFAULT_TYPE] = None,
    current_chat_id: Optional[int] = None,
    current_message_thread_id: Optional[int] = None # Added thread ID
) -> str:
    logger.info(f"TOOL: create_poll_tool for chat_id {current_chat_id} (thread_id: {current_message_thread_id}) with question='{question}', options={options}")

    if not telegram_bot_context or not current_chat_id:
        err_msg = "Telegram context or chat ID not provided to create_poll_tool."
        logger.error(err_msg)
        return json.dumps({"error": err_msg, "details": "This tool requires internal context to function."})

    # Validation
    if not question or not isinstance(question, str) or len(question.strip()) == 0:
        return json.dumps({"error": "Poll question cannot be empty."})
    if not options or not isinstance(options, list) or len(options) < 2 or len(options) > 10:
        return json.dumps({"error": "Poll must have between 2 and 10 options."})
    if not all(isinstance(opt, str) and opt.strip() for opt in options):
        return json.dumps({"error": "All poll options must be non-empty strings."})

    try:
        # Ensure all options are unique, Telegram might enforce this
        unique_options = list(dict.fromkeys(options)) # Preserves order while making unique
        if len(unique_options) < len(options):
            logger.warning(f"Poll options for '{question}' had duplicates, using unique set: {unique_options}")
        
        # Telegram API limits for question (1-300 chars) and options (1-100 chars)
        if len(question) > 300:
            return json.dumps({"error": "Poll question is too long (max 300 characters)."})
        for opt_idx, opt_val in enumerate(unique_options):
            if len(opt_val) > 100:
                 return json.dumps({"error": f"Poll option #{opt_idx+1} ('{opt_val[:20]}...') is too long (max 100 characters)." })
        
        # Attempt to send poll to the specific thread
        await telegram_bot_context.bot.send_poll(
            chat_id=current_chat_id,
            question=question,
            options=unique_options,
            is_anonymous=is_anonymous,
            allows_multiple_answers=allows_multiple_answers,
            message_thread_id=current_message_thread_id # Pass the thread ID
        )
        logger.info(f"Poll sent to chat {current_chat_id} (thread {current_message_thread_id}): '{question}'")
        return json.dumps({
            "status": "poll_created_successfully",
            "question": question,
            "options_sent": unique_options,
            "chat_id": current_chat_id,
            "message_thread_id": current_message_thread_id
        })
    except telegram.error.BadRequest as e:
        # Check if this BadRequest is due to thread not found for polls
        if "message thread not found" in e.message.lower() and current_message_thread_id is not None:
            logger.warning(f"Poll for chat {current_chat_id}, thread {current_message_thread_id} failed (thread not found). Retrying in general chat.")
            try:
                # Fallback: Send poll to general chat
                await telegram_bot_context.bot.send_poll(
                    chat_id=current_chat_id,
                    question=question,
                    options=unique_options,
                    is_anonymous=is_anonymous,
                    allows_multiple_answers=allows_multiple_answers,
                    message_thread_id=None # Fallback to general
                )
                logger.info(f"Poll sent to chat {current_chat_id} (GENERAL after thread fail): '{question}'")
                return json.dumps({
                    "status": "poll_created_successfully_in_general_after_thread_fail",
                    "question": question,
                    "options_sent": unique_options,
                    "chat_id": current_chat_id,
                    "message_thread_id": None
                })
            except Exception as e2:
                logger.error(f"Error creating poll in general chat (fallback) for {current_chat_id}: {e2}", exc_info=True)
                return json.dumps({"error": "Failed to create poll in thread and general.", "details": f"Telegram API error: {e2.message if hasattr(e2, 'message') else str(e2)}"})
        else:
            # Different BadRequest error
            logger.error(f"Telegram BadRequest creating poll in chat {current_chat_id} (thread {current_message_thread_id}): {e}", exc_info=True)
            return json.dumps({"error": "Failed to create poll.", "details": f"Telegram API error: {e.message}"})
    except Exception as e:
        logger.error(f"Unexpected error in create_poll_tool for chat {current_chat_id} (thread {current_message_thread_id}): {e}", exc_info=True)
        return json.dumps({"error": "An unexpected error occurred while trying to create the poll.", "details": str(e)})

async def calculator_tool(expression: str) -> str:
    logger.info(f"TOOL: calculator_tool with expression: '{expression}'")
    processed_expression = expression.replace('^', '**')
    if re.search(r"[a-zA-Z]", processed_expression):
        logger.warning(f"Calculator: Forbidden characters (letters) in expression: '{processed_expression}'")
        return json.dumps({"error": "Invalid expression: letters are not allowed."})
    temp_expr = processed_expression
    temp_expr = re.sub(r"[0-9\.\s]", "", temp_expr) # Keep dots for floats, remove numbers and whitespace
    temp_expr = temp_expr.replace('**', '').replace('*', '').replace('/', '').replace('+', '').replace('-', '').replace('(', '').replace(')', '')
    if temp_expr:
        logger.warning(f"Calculator: Invalid characters remain: '{temp_expr}' from '{processed_expression}'")
        return json.dumps({"error": "Invalid expression: contains disallowed characters."})
    try:
        result = eval(processed_expression, {"__builtins__": {}}, {})
        if not isinstance(result, (int, float)):
            return json.dumps({"error": "Expression did not evaluate to a numerical result."})
        if isinstance(result, float):
            result_str = f"{result:.10f}".rstrip('0').rstrip('.')
            if result_str == "-0": result_str = "0"
        else:
            result_str = str(result)
        logger.info(f"Calculator: Expression '{processed_expression}' evaluated to: {result_str}")
        return json.dumps({"result": result_str})
    except ZeroDivisionError: return json.dumps({"error": "Error: Division by zero."})
    except SyntaxError: return json.dumps({"error": "Error: Invalid mathematical expression syntax."})
    except TypeError: return json.dumps({"error": "Error: Type error in expression."})
    except OverflowError: return json.dumps({"error": "Error: Calculation result is too large."})
    except Exception as e:
        logger.error(f"Calculator: Unexpected error: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {type(e).__name__}."})

_WEATHER_CODES = {
    0: 'Clear', 1: 'Mostly Clear', 2: 'Partly Cloudy', 3: 'Overcast',
    45: 'Fog', 48: 'Freezing Fog',
    51: 'Light Drizzle', 53: 'Drizzle', 55: 'Heavy Drizzle',
    56: 'Freezing Drizzle', 57: 'Heavy Freezing Drizzle',
    61: 'Light Rain', 63: 'Rain', 65: 'Heavy Rain',
    66: 'Freezing Rain', 67: 'Heavy Freezing Rain',
    71: 'Light Snow', 73: 'Snow', 75: 'Heavy Snow',
    77: 'Snow Grains', 80: 'Light Showers', 81: 'Showers', 82: 'Violent Showers',
    85: 'Light Snow Showers', 86: 'Heavy Snow Showers',
    95: 'Thunderstorm', 96: 'Thunderstorm with Hail', 99: 'Severe Thunderstorm'
}

def _weather_code_to_text(code: int) -> str:
    return _WEATHER_CODES.get(code, 'Unknown')

def _format_current_weather(data: Dict[str, Any], unit: str, location_name: str) -> Dict[str, Any]:
    u = unit[0].upper()
    return {
        "location": location_name,
        "current": {
            "temperature": f"{data.get('temperature_2m')}°{u}",
            "feels_like": f"{data.get('apparent_temperature')}°{u}",
            "humidity": f"{data.get('relative_humidity_2m')}%",
            "precipitation": f"{data.get('precipitation')} mm",
            "wind": f"{data.get('wind_speed_10m')} km/h", # Open-Meteo default is km/h for wind_speed_10m
            "condition": _weather_code_to_text(data.get('weather_code', -1))
        }
    }

def _get_specific_hour_forecast(
    hourly_data: Dict[str, List[Any]],
    hours_ahead_target: int,
    timezone_str: str,
    unit_char: str
) -> Optional[Dict[str, Any]]:
    try:
        target_timezone = pytz.timezone(timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        logger.error(f"Weather: Unknown timezone provided by Open-Meteo: {timezone_str}")
        return {"error": f"Internal error: Invalid timezone '{timezone_str}' from weather service."}

    if hours_ahead_target < 0 or hours_ahead_target >= len(hourly_data.get('time', [])):
        logger.warning(f"Weather: hours_ahead_target {hours_ahead_target} is out of range for available forecast hours ({len(hourly_data.get('time', []))}).")
        return {"error": f"Requested hour {hours_ahead_target} is beyond forecast range."}

    idx = hours_ahead_target
    try:
        timestamp_str = hourly_data['time'][idx]
        dt_obj = datetime.fromisoformat(timestamp_str)
        if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
            dt_obj = target_timezone.localize(dt_obj)
        else:
            dt_obj = dt_obj.astimezone(target_timezone)

        return {
            "time": timestamp_str,
            "local_time": dt_obj.strftime('%a, %H:%M %Z'),
            "temperature": f"{hourly_data['temperature_2m'][idx]}°{unit_char}",
            "feels_like": f"{hourly_data['apparent_temperature'][idx]}°{unit_char}",
            "precipitation_chance": f"{hourly_data['precipitation_probability'][idx]}%",
            "condition": _weather_code_to_text(hourly_data['weather_code'][idx])
        }
    except (IndexError, KeyError, ValueError) as e:
        logger.error(f"Weather: Error processing specific hour forecast at index {idx}: {e}")
        return {"error": "Could not retrieve forecast for the specific hour."}

def _format_hourly_weather(
    data: Dict[str, Any], unit: str, location_name: str,
    specific_hour: Optional[int], timezone_str: str, forecast_days: int
) -> Dict[str, Any]:
    u = unit[0].upper()
    if specific_hour is not None:
        forecast_data = _get_specific_hour_forecast(data, specific_hour, timezone_str, u)
        if forecast_data and "error" in forecast_data:
             return {"location": location_name, "timeframe": "hourly", **forecast_data}
        return {"location": location_name, "timeframe": "hourly", "forecast_at_specific_hour": forecast_data}

    forecasts = []
    num_entries_to_show = 24 * forecast_days
    try:
        target_timezone = pytz.timezone(timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        logger.error(f"Weather: Unknown timezone for hourly formatting: {timezone_str}")
        return {"error": f"Internal error: Invalid timezone '{timezone_str}' for hourly display."}

    for i, timestamp_str in enumerate(data.get('time', [])):
        if i >= num_entries_to_show : break
        try:
            dt_obj = datetime.fromisoformat(timestamp_str)
            if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
                dt_obj = target_timezone.localize(dt_obj)
            else:
                dt_obj = dt_obj.astimezone(target_timezone)

            forecasts.append({
                "time_utc_iso": timestamp_str,
                "local_time": dt_obj.strftime('%a, %H:%M %Z'),
                "temperature": f"{data['temperature_2m'][i]}°{u}",
                "feels_like": f"{data['apparent_temperature'][i]}°{u}",
                "precipitation_chance": f"{data['precipitation_probability'][i]}%",
                "condition": _weather_code_to_text(data['weather_code'][i])
            })
        except (IndexError, KeyError, ValueError) as e_item:
            logger.warning(f"Weather: Skipping hourly item at index {i} due to error: {e_item}")
            continue
    return {"location": location_name, "timeframe": "hourly", "forecasts": forecasts}

def _format_daily_weather(data: Dict[str, Any], unit: str, location_name: str) -> Dict[str, Any]:
    u = unit[0].upper()
    forecasts = []
    for i, date_str in enumerate(data.get('time', [])):
        try:
            dt_obj = datetime.strptime(date_str, '%Y-%m-%d')
            forecasts.append({
                "date": dt_obj.strftime('%Y-%m-%d (%a)'),
                "high": f"{data['temperature_2m_max'][i]}°{u}",
                "low": f"{data['temperature_2m_min'][i]}°{u}",
                "feels_like_max": f"{data.get('apparent_temperature_max', ['N/A'])[i]}°{u}",
                "precipitation_chance": f"{data['precipitation_probability_max'][i]}%",
                "condition": _weather_code_to_text(data['weather_code'][i])
            })
        except (IndexError, KeyError, ValueError) as e_item:
            logger.warning(f"Weather: Skipping daily item at index {i} due to error: {e_item}")
            continue
    return {"location": location_name, "timeframe": "daily", "forecasts": forecasts}

async def get_weather_tool(
    location: str,
    timeframe: str = 'current',
    hours_ahead: Optional[int] = None,
    forecast_days: int = 1,
    unit: str = 'celsius'
) -> str:
    logger.info(f"TOOL: get_weather_tool for {location}, timeframe: {timeframe}, hours: {hours_ahead}, days: {forecast_days}, unit: {unit}")
    async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
        try:
            geo_params = {'q': location, 'format': 'json', 'limit': 1}
            geo_headers = {'User-Agent': 'TelegramAIBot/1.0'}
            geo_response = await client.get('https://nominatim.openstreetmap.org/search', params=geo_params, headers=geo_headers)
            geo_response.raise_for_status()
            geo_data = geo_response.json()

            if not geo_data:
                return json.dumps({"error": "Location not found", "details": f"Could not geocode: {location}"})

            lat = geo_data[0]['lat']
            lon = geo_data[0]['lon']
            display_name = geo_data[0]['display_name']

            max_api_forecast_days = 16
            effective_forecast_days = forecast_days
            if timeframe == 'hourly' and hours_ahead is not None:
                days_needed_for_hourly = math.ceil((hours_ahead + 1) / 24)
                effective_forecast_days = max(days_needed_for_hourly, forecast_days)

            effective_forecast_days = min(effective_forecast_days, max_api_forecast_days)
            effective_forecast_days = max(1, effective_forecast_days)

            weather_params: Dict[str, Any] = {
                'latitude': lat,
                'longitude': lon,
                'temperature_unit': unit,
                'timezone': 'auto',
                'forecast_days': effective_forecast_days
            }

            if timeframe == 'current':
                weather_params['current'] = 'temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m'
            elif timeframe == 'hourly':
                weather_params['hourly'] = 'temperature_2m,precipitation_probability,weather_code,apparent_temperature,wind_speed_10m'
            elif timeframe == 'daily':
                 weather_params['daily'] = 'temperature_2m_max,temperature_2m_min,weather_code,apparent_temperature_max,precipitation_probability_max'

            weather_response = await client.get('https://api.open-meteo.com/v1/forecast', params=weather_params)
            weather_response.raise_for_status()
            weather_data = weather_response.json()

            result_data = {}
            timezone_returned = weather_data.get('timezone', 'UTC')

            if timeframe == 'current' and 'current' in weather_data:
                result_data = _format_current_weather(weather_data['current'], unit, display_name)
            elif timeframe == 'hourly' and 'hourly' in weather_data:
                result_data = _format_hourly_weather(weather_data['hourly'], unit, display_name, hours_ahead, timezone_returned, effective_forecast_days)
            elif timeframe == 'daily' and 'daily' in weather_data:
                result_data = _format_daily_weather(weather_data['daily'], unit, display_name)
            else:
                return json.dumps({"error": f"No data for timeframe '{timeframe}' or data missing in response."})

            result_data['coordinates'] = {'latitude': float(lat), 'longitude': float(lon)}
            return json.dumps(result_data)
        except httpx.HTTPStatusError as e:
            err_detail = f"HTTP error {e.response.status_code} calling {e.request.url}."
            try:
                api_err = e.response.json()
                if 'reason' in api_err: err_detail += f" API Reason: {api_err['reason']}"
            except ValueError: pass
            logger.error(f"Weather Tool Error: {err_detail}", exc_info=True)
            return json.dumps({"error": "Weather service request failed.", "details": err_detail})
        except (httpx.RequestError, json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Weather Tool Error: {e}", exc_info=True)
            return json.dumps({"error": "Weather service unavailable or data processing error.", "details": str(e)})
        except Exception as e:
            logger.error(f"Unexpected Weather Tool Error: {e}", exc_info=True)
            return json.dumps({"error": "An unexpected error occurred in weather tool.", "details": str(e)})

async def web_search_tool(query: str, site: str = '', region: str = '', date_filter: str = '') -> str:
    logger.info(f"TOOL: web_search for query='{query}', site='{site}', region='{region}', date_filter='{date_filter}'")

    # Construct the search query string
    if site:
        query = f"{query} site:{site}"

    params = {
        'q': query,
        'kl': region,
        'df': date_filter,
        'kp': '-2' # No safe search
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT, follow_redirects=True) as client:
            response = await client.get('https://duckduckgo.com/html/', params=params)
            response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        results_list = []
        processed_results_count = 0
        for element_a in soup.select('.result__title a'):
            if processed_results_count >= 5: break # Limit to 5 results
            
            title = element_a.text.strip()
            raw_url = element_a.get('href')
            if not raw_url:
                logger.warning(f"Web Search: Found a result title ('{title[:50]}...') without an href. Skipping.")
                continue

            # Decode DDG's 'uddg' parameter if present
            parsed_url = urlparse(raw_url)
            uddg_list = parse_qs(parsed_url.query).get('uddg')
            if uddg_list and uddg_list[0]:
                clean_url = url_unquote(uddg_list[0])
            else:
                clean_url = raw_url # Fallback if 'uddg' is not found
                logger.warning(f"Web Search: Could not find 'uddg' param for URL {raw_url} (title: '{title[:50]}...'). Using raw href.")

            snippet_text = "Snippet not available."
            # Try to find the snippet more robustly
            result_item_container = element_a.find_parent(class_=re.compile(r'\bresult\b'))
            if result_item_container:
                snippet_tag = result_item_container.select_one('.result__snippet')
                if snippet_tag:
                    snippet_text = snippet_tag.text.strip()
                else: # Fallback if typical snippet class not found directly under result container
                    header_candidate = element_a.find_parent(class_='result__title')
                    if header_candidate: header_candidate = header_candidate.parent # up to result__header
                    if header_candidate:
                        snippet_tag_fallback = header_candidate.find_next_sibling(class_='result__snippet')
                        if snippet_tag_fallback: snippet_text = snippet_tag_fallback.text.strip()
            else:
                 logger.warning(f"Web Search: Could not determine result item container for title '{title[:50]}...'. Snippet may be missing.")

            results_list.append({
                'searchResult': f'Web Search Result #{processed_results_count + 1}',
                'title': title,
                'searchResultSourceUrl': clean_url,
                'snippet': snippet_text
            })
            processed_results_count += 1

        if not results_list:
            return json.dumps({'error': 'No results found'})

        # Format as a list of dictionaries as per common tool output patterns
        final_payload_list = [f'Results for search query "{query}":']
        final_payload_list.extend(results_list) # Add the list of result dicts
        return json.dumps({'results': final_payload_list})
    except httpx.HTTPStatusError as e:
        logger.error(f"Web Search: HTTP error {e.response.status_code} for '{query}'. Resp: {e.response.text[:200]}", exc_info=False)
        return json.dumps({'error': f'Unable to perform web search: HTTP {e.response.status_code}'})
    except httpx.RequestError as e:
        logger.error(f"Web Search: Request error for '{query}': {e}", exc_info=False)
        return json.dumps({'error': f'Unable to perform web search: Request Error - {type(e).__name__}'})
    except Exception as e:
        logger.error(f"Web Search: Unexpected error during web search for '{query}': {e}", exc_info=True)
        return json.dumps({'error': 'An unexpected error occurred during web search.'})

async def get_game_deals_tool(
    deal_id: Optional[int] = None,
    fetch_worth: bool = False,
    platform: Optional[str] = None, 
    type: Optional[str] = None, 
    sort_by: Optional[str] = None
) -> str:
    """
    Fetches information about free game giveaways. Can fetch a list of deals, a single deal by ID, or the total worth of all deals.
    - To get a list of current giveaways, use the 'platform', 'type', and 'sort_by' parameters.
    - To get the total number and value of giveaways, set 'fetch_worth' to true.
    - To get details for a specific giveaway, provide its 'deal_id' (The deal_id's are unknown until you fetch the deals first).
    """
    logger.info(
        f"TOOL: get_game_deals_tool with id={deal_id}, fetch_worth={fetch_worth}, "
        f"platform={platform}, type={type}, sort_by={sort_by}"
    )
    
    base_url = "https://www.gamerpower.com/api"
    params = {}

    # Determine the correct API endpoint based on the parameters provided
    if deal_id is not None:
        api_url = f"{base_url}/giveaway"
        params["id"] = deal_id
    elif fetch_worth:
        api_url = f"{base_url}/worth"
        if platform: params["platform"] = platform
        if type: params["type"] = type
    else:
        api_url = f"{base_url}/giveaways"
        if platform: params["platform"] = platform
        if type: params["type"] = type
        if sort_by: params["sort-by"] = sort_by

    try:
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
            response = await client.get(api_url, params=params)
            
            # GamerPower API returns 404 for a non-existent ID, which is a valid "not found" case
            if response.status_code == 404:
                return json.dumps({"result": "No giveaway found with the specified ID."})

            response.raise_for_status()
            data = response.json()

        if not data:
            return json.dumps({"result": "No data returned from the API for the specified criteria."})

        # The 'worth' endpoint returns a single object, not a list
        if fetch_worth:
            return json.dumps({"worth_estimation": data})

        # A single giveaway ID returns a single object
        if deal_id:
            return json.dumps({"giveaway_details": data})

        # Otherwise, we have a list of giveaways
        results_limit = 7 # Limit to a reasonable number for chat
        formatted_results = []
        for item in data[:results_limit]:
            formatted_results.append({
                "id": item.get("id"),
                "title": item.get("title"),
                "platforms": item.get("platforms"),
                "end_date": item.get("end_date"),
                "url": item.get("open_giveaway_url")
            })
        return json.dumps({"giveaways": formatted_results})

    except httpx.HTTPStatusError as e:
        error_details = f"API request failed with status code {e.response.status_code}."
        logger.error(f"GamerPower API Error: {error_details}", exc_info=True)
        return json.dumps({"error": "Failed to fetch data from the GamerPower API.", "details": error_details})
    except Exception as e:
        logger.error(f"Unexpected error in get_game_deals_tool: {e}", exc_info=True)
        return json.dumps({"error": "An unexpected error occurred."})

# --- ADD THIS HELPER FUNCTION FOR PARSING DURATION ---
def _parse_duration(duration_str: str) -> timedelta:
    """Parses a simple duration string (e.g., '30m', '2h', '1d') into a timedelta."""
    duration_str = duration_str.lower().strip()
    match = re.match(r"^(\d+)([mhd])$", duration_str)
    if not match:
        raise ValueError("Invalid duration format. Use 'm' for minutes, 'h' for hours, 'd' for days. E.g., '30m', '2h'.")
    
    value, unit = int(match.group(1)), match.group(2)
    if unit == 'm':
        return timedelta(minutes=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    return timedelta() # Should not be reached

# --- THE MODERATION TOOL IMPLEMENTATION ---
async def restrict_user_in_chat_tool(
    user_id: int,
    duration: str = '1h',
    reason: Optional[str] = None,
    # Parameters to be passed by the main handler:
    telegram_bot_context: Optional[ContextTypes.DEFAULT_TYPE] = None,
    current_chat_id: Optional[int] = None
) -> str:
    """Restricts a user in the current chat."""
    if not telegram_bot_context or not current_chat_id:
        return json.dumps({"error": "Internal error: Context or Chat ID not provided."})
    
    bot = telegram_bot_context.bot
    logger.info(f"TOOL: restrict_user_in_chat for user {user_id} in chat {current_chat_id} for {duration}. Reason: {reason}")

    # --- SAFETY CHECK 1: Bot cannot restrict itself ---
    if user_id == bot.id:
        return json.dumps({"status": "failed", "error": "Action failed: I cannot restrict myself."})

    try:
        # --- SAFETY CHECK 2: Get bot's status to ensure it's an admin ---
        bot_member = await bot.get_chat_member(current_chat_id, bot.id)
        if not isinstance(bot_member, (telegram.ChatMemberAdministrator, telegram.ChatMemberOwner)):
            return json.dumps({"status": "failed", "error": "Action failed: I am not an administrator in this chat."})
        if not bot_member.can_restrict_members:
            return json.dumps({"status": "failed", "error": "Action failed: I am an admin, but I don't have permission to restrict members."})
            
        # --- SAFETY CHECK 3: Get target user's status to ensure they are not an admin ---
        target_member = await bot.get_chat_member(current_chat_id, user_id)
        if target_member.status in [target_member.ADMINISTRATOR, target_member.CREATOR]:
             return json.dumps({"status": "failed", "error": f"Action failed: I cannot restrict another administrator or the chat owner ('{target_member.user.full_name}')."})

        # --- Parse duration and set restriction end time ---
        try:
            mute_timedelta = _parse_duration(duration)
            # Telegram expects a Unix timestamp for `until_date`
            until_date = datetime.now() + mute_timedelta
            logger.info(f"User {user_id} will be restricted until {until_date.isoformat()}")
        except ValueError as e:
            return json.dumps({"status": "failed", "error": str(e)})

        # --- Perform the action ---
        # To "mute", we set can_send_messages to False. All other perms remain default (None = Unchanged).
        await bot.restrict_chat_member(
            chat_id=current_chat_id,
            user_id=user_id,
            permissions=ChatPermissions(can_send_messages=False),
            until_date=until_date
        )
        
        success_message = f"User {user_id} ({target_member.user.full_name}) has been successfully muted for {duration}."
        if reason:
            success_message += f" Reason: {reason}"
        
        return json.dumps({"status": "success", "details": success_message})

    except telegram.error.BadRequest as e:
        logger.error(f"Error restricting user {user_id} in chat {current_chat_id}: {e.message}")
        if "user not found" in e.message.lower():
            return json.dumps({"status": "failed", "error": "User with the specified ID was not found in this chat."})
        return json.dumps({"status": "failed", "error": f"Telegram API error: {e.message}"})
    except Exception as e:
        logger.error(f"Unexpected error in restrict_user_in_chat_tool: {e}", exc_info=True)
        return json.dumps({"error": "An unexpected internal error occurred."})

async def get_user_info_tool(
    user_id: int,
    fetch_profile_photos: bool = False,
    # Parameters to be passed by the main handler:
    telegram_bot_context: Optional[ContextTypes.DEFAULT_TYPE] = None,
    current_chat_id: Optional[int] = None
) -> str:
    """Gets comprehensive information about a member of the current chat."""
    if not telegram_bot_context or not current_chat_id:
        return json.dumps({"error": "Internal error: Context or Chat ID not provided."})
    
    bot = telegram_bot_context.bot
    logger.info(f"TOOL: Comprehensive get_user_info for user {user_id} in chat {current_chat_id}")
    
    try:
        # --- Step 1: Get the ChatMember object ---
        # This is the most important call, as it confirms the user is in the chat
        # and gives us both their User object and their chat-specific status.
        member = await bot.get_chat_member(current_chat_id, user_id)
        user = member.user

        # --- Step 2: Assemble the result dictionary ---
        result = {
            "status": "success",
            "user_profile": {
                "id": user.id,
                "full_name": user.full_name,
                "first_name": user.first_name,
                "last_name": user.last_name or "N/A",
                "username": f"@{user.username}" if user.username else "N/A",
                "is_bot": user.is_bot,
                "is_premium": user.is_premium or False,
                "language_code": user.language_code or "N/A"
            },
            "chat_specific_info": {
                "status": member.status,
            }
        }

        # --- Step 3: Add context-specific details based on status ---
        chat_info = result["chat_specific_info"]
        
        if isinstance(member, (telegram.ChatMemberRestricted, telegram.ChatMemberBanned)):
            # If user is muted or banned, show until when
            if member.until_date:
                chat_info["restricted_until_utc"] = member.until_date.isoformat()
            else:
                chat_info["restricted_until_utc"] = "Permanent"
            # You can also add the specific permissions they are denied
            chat_info["permissions_denied"] = {
                "can_send_messages": member.can_send_messages,
                "can_send_media_messages": member.can_send_media_messages,
                "can_send_other_messages": member.can_send_other_messages,
                "can_add_web_page_previews": member.can_add_web_page_previews,
            }

        elif isinstance(member, telegram.ChatMemberAdministrator):
            chat_info["custom_title"] = member.custom_title or "N/A"
            chat_info["admin_permissions"] = {
                "can_manage_chat": member.can_manage_chat,
                "can_delete_messages": member.can_delete_messages,
                "can_manage_video_chats": member.can_manage_video_chats,
                "can_restrict_members": member.can_restrict_members,
                "can_promote_members": member.can_promote_members,
                "can_change_info": member.can_change_info,
                "can_invite_users": member.can_invite_users,
                "can_post_messages": member.can_post_messages,
                "can_edit_messages": member.can_edit_messages,
                "can_pin_messages": member.can_pin_messages,
                "can_manage_topics": member.can_manage_topics,
            }
        
        elif isinstance(member, telegram.ChatMemberOwner):
            chat_info["custom_title"] = member.custom_title or "N/A"
            chat_info["admin_permissions"] = "All (Owner)"

        # --- Step 4: Optionally fetch profile photos ---
        if fetch_profile_photos:
            try:
                profile_photos = await bot.get_user_profile_photos(user_id, limit=3) # Limit to 3 to keep response size sane
                result["user_profile"]["profile_photos"] = {
                    "total_count": profile_photos.total_count,
                    "fetched_photos": [
                        {
                            "file_id": p[-1].file_id, # Get the file_id of the largest photo
                            "file_unique_id": p[-1].file_unique_id,
                            "width": p[-1].width,
                            "height": p[-1].height
                        } for p in profile_photos.photos
                    ]
                }
            except Exception as e:
                logger.warning(f"Could not fetch profile photos for user {user_id}: {e}")
                result["user_profile"]["profile_photos"] = {"error": "Could not fetch profile photos."}

        return json.dumps(result, indent=2)

    except telegram.error.BadRequest as e:
        if "user not found" in e.message.lower():
            return json.dumps({"status": "not_found", "error": "User with the specified ID was not found in this chat."})
        return json.dumps({"status": "failed", "error": f"Telegram API error: {e.message}"})
    except Exception as e:
        logger.error(f"Unexpected error in comprehensive get_user_info_tool: {e}", exc_info=True)
        return json.dumps({"error": "An unexpected internal error occurred."})

AVAILABLE_TOOLS_PYTHON_FUNCTIONS = {
    "calculator": calculator_tool,
    "get_weather": get_weather_tool,
    "web_search": web_search_tool,
    "create_poll_in_chat": create_poll_tool,
    "get_game_deals": get_game_deals_tool,
    "restrict_user_in_chat": restrict_user_in_chat_tool,
    "get_user_info": get_user_info_tool,
}

TOOL_DEFINITIONS_FOR_API: list[ChatCompletionToolParam] = [
   {
      "type":"function",
      "function":{
         "name":"calculator",
         "description":"Evaluates a mathematical expression. Supports +, -, *, /, ** (exponentiation), and parentheses.",
         "parameters":{
            "type":"object",
            "properties":{
               "expression":{
                  "type":"string",
                  "description":"The mathematical expression to evaluate (e.g., '2+2', '(5*8-3)/2', '2^10')."
               }
            },
            "required":[
               "expression"
            ]
         }
      }
   },
   {
      "type":"function",
      "function":{
         "name":"get_weather",
         "description":"Get weather data for a location with flexible timeframes.",
         "parameters":{
            "type":"object",
            "properties":{
               "location":{
                  "type":"string",
                  "description":"The city and country, e.g., London, UK or a specific address."
               },
               "timeframe":{
                  "type":"string",
                  "enum":[
                     "current",
                     "hourly",
                     "daily"
                  ],
                  "default":"current",
                  "description":"Time resolution of the forecast: 'current' for current conditions, 'hourly' for hour-by-hour, 'daily' for day summaries."
               },
               "hours_ahead":{
                  "type":"number",
                  "description":"Optional. For 'hourly' timeframe, specifies the number of hours from now for which to get a single forecast point (e.g., 0 for current hour, 1 for next hour). Max 167."
               },
               "forecast_days":{
                  "type":"number",
                  "minimum":1,
                  "maximum":14,
                  "default":1,
                  "description":"Number of days to forecast (for daily or hourly if hours_ahead is not specified). E.g., 1 for today, 7 for a week."
               },
               "unit":{
                  "type":"string",
                  "enum":[
                     "celsius",
                     "fahrenheit"
                  ],
                  "default":"celsius",
                  "description":"Temperature unit."
               }
            },
            "required":[
               "location"
            ]
         }
      }
   },
   {
      "type":"function",
      "function":{
         "name":"web_search",
         "description":"Search the web/online/on the internet for information on a given topic. Fetches only the first page of search results from DuckDuckGo. Use when you require information you are unsure or unaware of.",
         "parameters":{
            "type":"object",
            "properties":{
               "query":{
                  "type":"string",
                  "description":"The search query, e.g., 'What is the capital of France?'"
               },
               "site":{
                  "type":"string",
                  "description":"Optional. Limit search to a specific website (e.g., 'wikipedia.org') or use a DuckDuckGo bang (e.g., '!w' for Wikipedia). This is passed to DuckDuckGo's 'b' (bang) parameter. Leave empty for general search."
               },
               "region":{
                  "type":"string",
                  "description":"Optional. Limit search to results from a specific region/language (e.g., 'us-en' for US English, 'de-de' for Germany German). This is a DuckDuckGo region code passed to 'kl' parameter. Leave empty for global results."
               },
               "date_filter":{
                  "type":"string",
                  "description":"Optional. Filter search results by date: 'd' (past day), 'w' (past week), 'm' (past month), 'y' (past year). Passed to DuckDuckGo's 'df' parameter. Leave empty for no date filter.",
                  "enum":[
                     "",
                     "d",
                     "w",
                     "m",
                     "y"
                  ]
               }
            },
            "required":[
               "query"
            ]
         }
      }
   },
   {
      "type":"function",
      "function":{
         "name":"create_poll_in_chat",
         "description":"Creates a new poll in the current Telegram chat. This is useful for asking questions with multiple choice answers to the group or user.",
         "parameters":{
            "type":"object",
            "properties":{
               "question":{
                  "type":"string",
                  "description":"The question for the poll. Max 300 characters."
               },
               "options":{
                  "type":"array",
                  "items":{
                     "type":"string"
                  },
                  "minItems":2,
                  "maxItems":10,
                  "description":"A list of 2 to 10 answer options for the poll. Each option max 100 characters."
               },
               "is_anonymous":{
                  "type":"boolean",
                  "default":True,
                  "description":"Optional. If true, the poll is anonymous. Defaults to true."
               },
               "allows_multiple_answers":{
                  "type":"boolean",
                  "default":False,
                  "description":"Optional. If true, users can select multiple answers. Defaults to false."
               }
            },
            "required":[
               "question",
               "options"
            ]
         }
      }
   },
   {
      "type":"function",
      "function":{
         "name":"get_game_deals",
         "description":"Fetches information about free game giveaways. Can fetch a list of deals, a single deal by ID, or the total worth of all active deals.",
         "parameters":{
            "type":"object",
            "properties":{
               "deal_id":{
                  "type":"integer",
                  "description":"The unique ID of a specific giveaway to fetch details for."
               },
               "fetch_worth":{
                  "type":"boolean",
                  "description":"Set to true to get the total number and estimated USD value of all active giveaways instead of a list."
               },
               "platform":{
                  "type":"string",
                  "description":"The platform to filter giveaways for. e.g., 'pc', 'steam', 'epic-games-store'.",
                  "enum":[
                     "pc",
                     "steam",
                     "epic-games-store",
                     "ubisoft",
                     "gog",
                     "itchio",
                     "ps4",
                     "ps5",
                     "xbox-one",
                     "xbox-series-xs",
                     "switch",
                     "android",
                     "ios",
                     "vr",
                     "battlenet",
                     "origin",
                     "drm-free",
                     "xbox-360"
                  ]
               },
               "type":{
                  "type":"string",
                  "description":"The type of giveaway to filter for.",
                  "enum":[
                     "game",
                     "loot",
                     "beta"
                  ]
               },
               "sort_by":{
                  "type":"string",
                  "description":"How to sort the results when fetching a list.",
                  "enum":[
                     "date",
                     "value",
                     "popularity"
                  ]
               }
            },
            "required":[
               
            ]
         }
      }
   },
   {
      "type":"function",
      "function":{
         "name":"restrict_user_in_chat",
         "description":"Temporarily mutes a user in the telegram chat, preventing them from sending messages. Requires the user's telegram ID. You must be an admin with permission to restrict members. Do not abuse this, keep mute durations low unless the offense truly warrants more.",
         "parameters":{
            "type":"object",
            "properties":{
               "user_id":{
                  "type":"integer",
                  "description":"The unique integer ID of the user to mute. You can get this from a user's context or the get_user_info tool."
               },
               "duration":{
                  "type":"string",
                  "description":"The duration for the mute. Examples: '30m' for 30 minutes, '2h' for 2 hours, '1d' for 1 day. If not provided, defaults to 1 hour."
               },
               "reason":{
                  "type":"string",
                  "description":"Optional. The reason for the restriction, which will be logged."
               }
            },
            "required":[
               "user_id"
            ]
         }
      }
   },
   {
      "type":"function",
      "function":{
         "name":"get_user_info",
         "description":"Retrieves comprehensive information about a user, including their global profile, chat-specific status (like admin rights or restrictions), and optionally their profile pictures.",
         "parameters":{
            "type":"object",
            "properties":{
               "user_id":{
                  "type":"integer",
                  "description":"The unique integer ID of the user to look up. You can get this from a user's context."
               },
               "fetch_profile_photos":{
                  "type":"boolean",
                  "default":False,
                  "description":"Set to true to also fetch the user's profile pictures. This is an extra step and may not always be necessary."
               }
            },
            "required":[
               "user_id"
            ]
         }
      }
   }
]
# --- END OF TOOL DEFINITIONS ---

# --- UTILITY FUNCTIONS ---
def get_display_name(user: Optional[TelegramUser]) -> str:
    if not user: return "Unknown User"
    name = user.full_name
    if user.username:
        name = f"{name} (@{user.username})" if name and user.username else user.username
    if not name:
        name = f"User_{user.id}"
    return name

def is_user_or_chat_allowed(user_id: int, chat_id: int) -> bool:
    if not ALLOWED_USERS and not ALLOWED_CHATS: return True
    if ALLOWED_USERS and str(user_id) in ALLOWED_USERS: return True
    if ALLOWED_CHATS and str(chat_id) in ALLOWED_CHATS: return True
    return False

async def handle_permission_denied(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat:
        message_thread_id: Optional[int] = None
        if update.message and update.message.message_thread_id is not None:
            message_thread_id = update.message.message_thread_id
        # Use the new helper for sending
        try:
            await send_message_to_chat_or_general(
                context.bot,
                update.effective_chat.id,
                "Sorry, you are not authorized to use this command or interact with me.",
                preferred_thread_id=message_thread_id
            )
        except Exception as e:
            logger.error(f"Failed to send permission denied message to {update.effective_chat.id} (thread {message_thread_id}): {e}")


def get_llm_chat_history(chat_id: int, message_thread_id: Optional[int]) -> list[ChatCompletionMessageParam]:
    # (Code adjusted for thread_id, comments from old version not directly applicable)
    history_key = (chat_id, message_thread_id)
    if history_key not in chat_histories:
        chat_histories[history_key] = []
    
    current_thread_history = chat_histories[history_key]
    # Trim history if it gets excessively long (more aggressive than just MAX_HISTORY_LENGTH)
    if len(current_thread_history) > MAX_HISTORY_LENGTH * 7: 
        trimmed_length = MAX_HISTORY_LENGTH * 5 # Trim down significantly
        logger.info(f"Trimming LLM chat history for chat {chat_id} (thread {message_thread_id}) from {len(current_thread_history)} to {trimmed_length}")
        chat_histories[history_key] = current_thread_history[-trimmed_length:]
    return chat_histories[history_key]

def add_to_raw_group_log(chat_id: int, message_thread_id: Optional[int], sender_name: str, text: str):
    # (Code adjusted for thread_id, comments from old version not directly applicable)
    if chat_id not in group_raw_message_log:
        group_raw_message_log[chat_id] = {}
    if message_thread_id not in group_raw_message_log[chat_id]:
        group_raw_message_log[chat_id][message_thread_id] = []
    
    log_entry = {"sender_name": sender_name, "text": text if text else "[empty message content]"}
    current_thread_log = group_raw_message_log[chat_id][message_thread_id]
    current_thread_log.append(log_entry)
    
    # Trim the log for the specific thread if it exceeds the limit
    if len(current_thread_log) > MAX_RAW_LOG_PER_THREAD:
        group_raw_message_log[chat_id][message_thread_id] = current_thread_log[-MAX_RAW_LOG_PER_THREAD:]
    
    topic_desc = f"topic ID {message_thread_id}" if message_thread_id is not None else "general chat"
    # Reduced log level for this frequent message
    logger.debug(f"Raw message log for group {chat_id} ({topic_desc}) updated. Length: {len(group_raw_message_log[chat_id][message_thread_id])} messages.")


def format_freewill_context_from_raw_log(
    chat_id: int,
    message_thread_id: Optional[int], # Added thread ID
    num_messages_to_include: int,
    bot_name: str,
    current_chat_title: Optional[str],
    current_topic_name_if_known: Optional[str],
    replied_to_message_context: Optional[Dict[str, str]] = None
) -> str:
    # Build a descriptive location string
    location_description_parts = []
    if current_chat_title:
        location_description_parts.append(f"group '{current_chat_title}'")
    else: # Fallback if title not passed or None (should ideally always be passed for groups)
        location_description_parts.append(f"a group chat (ID: {chat_id})")

    if message_thread_id is not None: # This is the topic ID for the raw log
        if current_topic_name_if_known: # If resolved and passed from process_message_entrypoint
            location_description_parts.append(f"topic '{current_topic_name_if_known}'")
        # No need to check cache here again if current_topic_name_if_known is the primary way to pass it
        else: # Fallback if not explicitly passed (e.g. cache miss in process_message_entrypoint)
            # Try to get from cache as a last resort for free will prompt
            cached_topic_name = topic_names_cache.get((chat_id, message_thread_id))
            if cached_topic_name:
                location_description_parts.append(f"topic '{cached_topic_name}'")
            else:
                location_description_parts.append(f"topic ID {message_thread_id} (name not resolved)")
    elif current_chat_title : # In a group, but not a specific topic (general area)
        location_description_parts.append("the general chat area")
    # For private chats, message_thread_id will be None, current_chat_title might be None or user's name

    topic_desc_log = "this chat" # Default for private chats or if somehow no group info
    if location_description_parts:
        topic_desc_log = " ".join(location_description_parts)
    
    # Check if log exists for the chat AND the specific thread
    if chat_id not in group_raw_message_log or \
       message_thread_id not in group_raw_message_log.get(chat_id, {}) or \
       not group_raw_message_log[chat_id].get(message_thread_id) or \
       num_messages_to_include <= 0:
        return f"[Recent conversation context for this group's {topic_desc_log} is minimal or unavailable.]\n"
    
    log = group_raw_message_log[chat_id][message_thread_id] # Get log for the specific thread
    start_index = max(0, len(log) - num_messages_to_include)
    context_messages_to_format = log[start_index:]

    if not context_messages_to_format:
         return f"[No prior messages in raw log for this group's {topic_desc_log} to form context.]\n"

    formatted_context_parts = [f"[Recent conversation excerpt from {topic_desc_log}:]"]
    triggering_user_name = "Unknown User" # Default value
    # Initialize triggering_message_text with the original text of the last message in context
    # This will be updated if the last message text gets neutralized.
    if context_messages_to_format: # Ensure list is not empty
        triggering_message_text = context_messages_to_format[-1].get("text", "[message content not available]")
    else: # Should not happen due to earlier check, but for safety:
        triggering_message_text = "[message content not available]"


    for i, msg_data in enumerate(context_messages_to_format):
        sender = msg_data.get("sender_name", "Unknown User")
        text = msg_data.get("text", "[message content not available]")
        original_text_for_this_message = text # Keep a copy of the original for this message

        # --- START OF MODIFICATION TO ADDRESS PREFIX COMMANDS IN FREE WILL CONTEXT ---
        # If this message is part of free will context and contains a known prefix command,
        # neutralize it to prevent the Shapes API's built-in commands from triggering.
        # We replace the command with a version that's still readable for AI context
        # but shouldn't be directly executable by the Shapes API.
        # (Original issue: "The shapes api which I use has some builtin prefix commands like '!wack' '!dashboard' etc...")
        # (Original issue: "...these also get triggered when the bot uses free will if the command was sent in one of the messages taken into free will context...")
        # (Refinement: Neutralize any command starting with '!' to '!_' regardless of where it is in the text, for ALL such commands)

        # Find all words starting with '!' (potential commands)
        # A "word" starting with '!' means '!' followed by one or more non-whitespace characters.
        # We use a regex that captures the '!' and the command word.
        # The replacement function will then prepend '_' to the captured command word.
        def neutralize_command(match: re.Match) -> str:
            # match.group(0) is the full matched command, e.g., "!wack"
            # We want to return "!_wack"
            return f"!_{match.group(0)[1:]}" # Prepend "_" after the "!"

        # Apply this to all occurrences in the text.
        # The regex (?<=\s|^) ensures the '!' is at the beginning of a word (preceded by space or start of string).
        # (\!\w+) captures the '!' and the subsequent word characters.
        # Using re.sub with a function allows dynamic replacement.
        # This regex will match "!command" and replace it with "!_command".
        # It handles multiple commands in a single message text.
        modified_text = re.sub(r"(?<!\S)(!\w+)", neutralize_command, text)
        
        if modified_text != text: # Log if any command was actually neutralized in this message
            logger.debug(f"Free will context: Neutralized prefix commands in message from '{sender}'. Original: '{text[:100]}...', Modified: '{modified_text[:100]}...'")
            text = modified_text # Use the modified text
        # --- END OF MODIFICATION TO ADDRESS PREFIX COMMANDS IN FREE WILL CONTEXT ---
            
        max_len_per_msg_in_context = 4096
        if len(text) > max_len_per_msg_in_context:
            text = text[:max_len_per_msg_in_context].strip() + "..."
        formatted_context_parts.append(f"- '{sender}' said: \"{text}\"")
        
        # Capture the last message details for the final prompt part.
        # If the last message in the context was modified, use its modified form for `triggering_message_text`.
        if i == len(context_messages_to_format) - 1: 
            triggering_user_name = sender
            # If the text of the *last message* was modified by neutralization,
            # `triggering_message_text` should reflect this modified version.
            # Otherwise, it keeps its initially assigned value (original text of the last message).
            if modified_text != original_text_for_this_message :
                triggering_message_text = text # Use the modified, potentially truncated text
            # else, triggering_message_text remains the original full text of the last message.
            # (No need for an `else` here as it's pre-assigned and only updated if modified)
            
            # Truncate if the (potentially modified) triggering message text itself is very long
            if len(triggering_message_text) > 4096: 
                triggering_message_text = triggering_message_text[:4096].strip() + "..."
    
    # Construct and add the reply context to the final prompt
    reply_context_addon = ""
    if replied_to_message_context:
        replied_author = replied_to_message_context.get("author", "someone")
        replied_content = replied_to_message_context.get("content", "[their message]")
        reply_context_addon = f" (in reply to '{replied_author}' who said: \"{replied_content}\")"
    
    formatted_context_parts.append(
        f"\n[You are '{bot_name}', chatting on Telegram in {topic_desc_log}. Based on the excerpt above, where '{triggering_user_name}' "
        f"just said: \"{triggering_message_text}\"{reply_context_addon}, "
        "make a relevant and in character interjection or comment. Be concise and natural.]"
    )
    return "\n".join(formatted_context_parts) + "\n\n"

async def _keep_typing_loop(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_thread_id: Optional[int], action: str = ChatAction.TYPING, interval: float = 4.5):
    while True:
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=action, message_thread_id=message_thread_id)
            await asyncio.sleep(interval)
        except asyncio.CancelledError: break
        except telegram.error.BadRequest as e:
            # Handle case where the thread might have been deleted while typing
            if "message thread not found" in e.message.lower() and message_thread_id is not None:
                logger.warning(f"Typing loop: Thread {message_thread_id} not found for chat {chat_id}. Stopping typing for this thread.")
                break # Stop trying for this specific (now invalid) thread
            logger.warning(f"Error sending {action} action in loop for chat {chat_id} (thread {message_thread_id}): {e}")
            await asyncio.sleep(interval) # Continue loop even on other BadRequest errors, but log it
        except Exception as e:
            logger.warning(f"Error sending {action} action in loop for chat {chat_id} (thread {message_thread_id}): {e}")
            await asyncio.sleep(interval) # Continue loop even on other errors, but log it
# --- END OF UTILITY FUNCTIONS ---

# --- Status Update Handler for Forum Topics (to populate cache) ---
async def handle_forum_topic_updates(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat:
        logger.debug("Forum topic update lacks message or effective_chat. Skipping.")
        return
        
    # Ignore old status updates if feature is enabled
    if IGNORE_OLD_MESSAGES_ON_STARTUP and BOT_STARTUP_TIMESTAMP:
        message_date_utc = update.message.date # Status update messages also have a date
        if message_date_utc.tzinfo is None:
            message_date_utc = message_date_utc.replace(tzinfo=dt_timezone.utc)
        else:
            message_date_utc = message_date_utc.astimezone(dt_timezone.utc)
        if message_date_utc < BOT_STARTUP_TIMESTAMP:
            logger.info(f"Ignoring old forum topic update (MsgID: {update.message.message_id}) from before bot startup.")
            return

    chat_id = update.effective_chat.id
    # Topic events are always tied to a specific message_thread_id for created/edited
    thread_id: Optional[int] = update.message.message_thread_id 
    
    topic_name_to_cache: Optional[str] = None
    action_taken: Optional[str] = None

    if update.message.forum_topic_created and thread_id is not None:
        topic_name_to_cache = update.message.forum_topic_created.name
        action_taken = "created"
    elif update.message.forum_topic_edited and thread_id is not None:
        # The `name` field in `forum_topic_edited` is the *new* name
        topic_name_to_cache = update.message.forum_topic_edited.name
        action_taken = "edited"
    
    if topic_name_to_cache is not None and thread_id is not None:
        key = (chat_id, thread_id)
        old_name = topic_names_cache.get(key)
        
        topic_names_cache[key] = topic_name_to_cache
        
        # Manage cache order for pruning
        if chat_id not in chat_topic_cache_keys_order:
            chat_topic_cache_keys_order[chat_id] = []
        if key in chat_topic_cache_keys_order[chat_id]:
            chat_topic_cache_keys_order[chat_id].remove(key) # Move to end if re-edited
        chat_topic_cache_keys_order[chat_id].append(key)
        
        # Prune if cache for this chat_id exceeds limit
        if len(chat_topic_cache_keys_order[chat_id]) > MAX_TOPIC_ENTRIES_PER_CHAT_IN_CACHE:
            key_to_remove = chat_topic_cache_keys_order[chat_id].pop(0) # Remove oldest
            if key_to_remove in topic_names_cache:
                removed_name = topic_names_cache.pop(key_to_remove)
                logger.info(f"Pruned oldest topic '{removed_name}' (Key: {key_to_remove}) from cache for chat {chat_id} due to size limit.")

        if action_taken == "created":
            logger.info(f"Topic '{topic_name_to_cache}' (ID: {thread_id}) {action_taken} in chat {chat_id}. Cached.")
        elif action_taken == "edited": # Log edit only if name changed or for debug
            if old_name != topic_name_to_cache :
                 logger.info(f"Topic ID {thread_id} in chat {chat_id} renamed from '{old_name}' to '{topic_name_to_cache}'. Cache updated.")
            else:
                 logger.debug(f"Topic ID {thread_id} in chat {chat_id} edited but name ('{topic_name_to_cache}') unchanged. Cache refreshed.")

# --- COMMAND HANDLERS ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat or not update.effective_user: return
    
    # Check if the command is specifically for this bot in group chats
    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(' ', 1)
        command_with_potential_mention = command_parts[0]
        if "@" in command_with_potential_mention and f"@{context.bot.username}" not in command_with_potential_mention:
            logger.info(f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring.")
            return

    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        await handle_permission_denied(update, context); return
    
    message_thread_id: Optional[int] = None
    if update.message and update.message.message_thread_id is not None:
        message_thread_id = update.message.message_thread_id

    start_message = (
        f"Hello! I am {SHAPESINC_SHAPE_USERNAME}, chatting here on Telegram! "
        "I can chat, use tools, and even understand images and voice messages.\n\n"
        "Type /help to see a list of commands."
    )
    try:
        # Use helper to send, respecting thread ID
        await send_message_to_chat_or_general(
            context.bot, update.effective_chat.id, start_message, preferred_thread_id=message_thread_id
        )
    except Exception as e:
        logger.error(f"Failed to send start message to {update.effective_chat.id} (thread {message_thread_id}): {e}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat or not update.effective_user: return

    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(' ', 1)
        command_with_potential_mention = command_parts[0]
        if "@" in command_with_potential_mention and f"@{context.bot.username}" not in command_with_potential_mention:
            logger.info(f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring.")
            return
            
    message_thread_id: Optional[int] = None
    if update.message and update.message.message_thread_id is not None:
        message_thread_id = update.message.message_thread_id
        
    help_text_parts = [
        "Here are the available commands:",
        "/start - Display the welcome message.",
        "/help - Show this help message.",
        "/newchat - Clear the current conversation history (for this topic/chat) and start fresh.",
        "/activate - (Groups/Topics only) Make me respond to every message in this specific group/topic.",
        "/deactivate - (Groups/Topics only) Stop me from responding to every message (revert to mentions/replies/free will)."
    ]
    if BING_IMAGE_CREATOR_AVAILABLE and BING_AUTH_COOKIE:
        help_text_parts.append("/imagine <prompt> - Generate images based on your prompt using Bing.")
        if ALLOWED_USERS and str(update.effective_user.id) in ALLOWED_USERS:
             help_text_parts.append("/setbingcookie <cookie_value> - (Admin) Update the Bing authentication cookie.")

    help_text_parts.append("\nSimply send me a message, an image (with or without a caption), or a voice message to start chatting!") 
    if ENABLE_TOOL_USE and TOOL_DEFINITIONS_FOR_API:
        help_text_parts.append("\nI can also use tools like:")
        for tool_def in TOOL_DEFINITIONS_FOR_API:
            if tool_def['type'] == 'function':
                func_info = tool_def['function']
                desc_first_sentence = func_info['description'].split('.')[0] + "."
                help_text_parts.append(f"  - `{func_info['name']}`: {desc_first_sentence}")

    if GROUP_FREE_WILL_ENABLED and update.effective_chat and update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        # Adjusted comment for thread awareness
        help_text_parts.append(f"\nGroup Free Will is enabled! I might respond randomly about {GROUP_FREE_WILL_PROBABILITY:.1%} of the time, considering the last ~{GROUP_FREE_WILL_CONTEXT_MESSAGES} messages in this specific topic/chat.")

    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        help_text_parts.append("\n\nNote: Your access to interact with me is currently restricted.")

    escaped_help_text = telegram_markdown_v2_escape("\n".join(help_text_parts))
    try:
        # Use helper to send, respecting thread ID
        await send_message_to_chat_or_general(
            context.bot, update.effective_chat.id, escaped_help_text,
            preferred_thread_id=message_thread_id, parse_mode=ParseMode.MARKDOWN_V2
        )
    except Exception as e:
         logger.error(f"Failed to send help message to {update.effective_chat.id} (thread {message_thread_id}): {e}")
         # Fallback to plain text if MDv2 fails for reasons other than thread not found (handled by helper)
         try:
            # Use helper for plain text fallback as well
            await send_message_to_chat_or_general(
                context.bot, update.effective_chat.id, "\n".join(help_text_parts), # Send unescaped for plain
                preferred_thread_id=message_thread_id
            )
         except Exception as e2:
            logger.error(f"Failed to send plain text help fallback to {update.effective_chat.id} (thread {message_thread_id}): {e2}")


async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat or not update.effective_user: return

    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(' ', 1)
        command_with_potential_mention = command_parts[0]
        if "@" in command_with_potential_mention and f"@{context.bot.username}" not in command_with_potential_mention:
            logger.info(f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring.")
            return

    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        await handle_permission_denied(update, context); return

    chat_id = update.effective_chat.id
    message_thread_id: Optional[int] = None
    if update.message.message_thread_id is not None:
        message_thread_id = update.message.message_thread_id
    
    # Key for history/log now includes thread_id
    history_key = (chat_id, message_thread_id)
    topic_desc = f"topic ID {message_thread_id}" if message_thread_id is not None else "general chat"
    cleared_any = False

    # Clear LLM history for this specific thread/chat
    if history_key in chat_histories and chat_histories[history_key]:
        chat_histories[history_key] = []
        logger.info(f"LLM Conversation history cleared for chat ID: {chat_id} ({topic_desc})")
        cleared_any = True
    
    # Also clear raw log if it exists for the chat and thread
    if chat_id in group_raw_message_log:
        if message_thread_id in group_raw_message_log.get(chat_id, {}):
            if group_raw_message_log[chat_id][message_thread_id]: # Check if the thread log is non-empty
                group_raw_message_log[chat_id][message_thread_id] = []
                logger.info(f"Raw group message log cleared for chat ID: {chat_id} ({topic_desc})")
                cleared_any = True
    
    # Use topic-aware message
    reply_text = "✨ Conversation history for this topic/chat cleared! Let's start a new topic." if cleared_any else "There's no conversation history for this topic/chat to clear yet."
    # update.message.reply_text automatically handles the thread_id
    await update.message.reply_text(reply_text)


async def imagine_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_chat or not update.effective_user: return

    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(' ', 1)
        command_with_potential_mention = command_parts[0]
        if "@" in command_with_potential_mention and f"@{context.bot.username}" not in command_with_potential_mention:
            logger.info(f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring.")
            return
            
    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        await handle_permission_denied(update, context); return
    
    # reply_text handles thread_id automatically
    if not (BING_IMAGE_CREATOR_AVAILABLE and ImageGen and BING_AUTH_COOKIE):
        await update.message.reply_text("The /imagine command is currently unavailable or not configured. Please contact an admin."); return 
    if not context.args: 
        await update.message.reply_text("Please provide a prompt for the image. Usage: /imagine <your image prompt>"); return

    prompt = " ".join(context.args)
    chat_id=update.effective_chat.id
    message_thread_id: Optional[int] = None
    if update.message.message_thread_id is not None:
        message_thread_id = update.message.message_thread_id

    typing_task: Optional[asyncio.Task] = None
    status_msg: Optional[TelegramMessage] = None
    temp_dir = f"temp_bing_images_{chat_id}_{random.randint(1000,9999)}" # Unique temp dir

    try:
        # Send initial action respecting thread_id
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO, message_thread_id=message_thread_id)
        typing_task = asyncio.create_task(_keep_typing_loop(context, chat_id, message_thread_id, action=ChatAction.UPLOAD_PHOTO, interval=5.0))
        # Reply handles thread_id automatically
        status_msg = await update.message.reply_text(f"🎨 Working on your vision: \"{prompt[:50]}...\" (using Bing)") 

        image_gen = ImageGen(auth_cookie=BING_AUTH_COOKIE) # Uses the global BING_AUTH_COOKIE
        image_links = await asyncio.to_thread(image_gen.get_images, prompt)

        if not image_links:
            if status_msg: await status_msg.edit_text("Sorry, Bing couldn't generate images for that prompt, or no images were returned. Try rephrasing!")
            return

        os.makedirs(temp_dir, exist_ok=True)
        # Download images to the temporary directory
        await asyncio.to_thread(image_gen.save_images, image_links, temp_dir, download_count=len(image_links))

        media_photos: List[InputMediaPhoto] = []
        image_files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
        image_files.sort() # Ensure some order if multiple images

        for filename in image_files:
            if len(media_photos) >= 10: break # Telegram media group limit
            image_path = os.path.join(temp_dir, filename)
            try:
                with open(image_path, "rb") as img_f:
                    img_bytes = img_f.read()
                    media_photos.append(InputMediaPhoto(media=img_bytes))
            except Exception as e_img: logger.error(f"Error processing image file {filename} for media group: {e_img}")

        if typing_task: typing_task.cancel(); typing_task=None # Cancel before sending media

        if media_photos:
            # send_media_group needs explicit thread_id
            await context.bot.send_media_group(chat_id=chat_id, media=media_photos, message_thread_id=message_thread_id)
            if status_msg: await status_msg.delete() # Clean up status message
        else:
            err_msg_no_proc = "Sorry, no images could be processed or sent from Bing."
            if status_msg: await status_msg.edit_text(err_msg_no_proc)
            else: await update.message.reply_text(err_msg_no_proc) # Fallback reply

    except Exception as e:
        logger.error(f"Error during /imagine command for prompt '{prompt}': {e}", exc_info=True)
        err_text = "An error occurred while generating images with Bing. Please try again later."
        try:
            if status_msg: await status_msg.edit_text(err_text)
            # Use helper for direct send fallback
            else: await send_message_to_chat_or_general(context.bot, chat_id, err_text, preferred_thread_id=message_thread_id) 
        except Exception: # Ultimate fallback if editing/sending fails
            await send_message_to_chat_or_general(context.bot, chat_id, err_text, preferred_thread_id=message_thread_id)
    finally:
        if typing_task and not typing_task.done(): typing_task.cancel()
        if os.path.exists(temp_dir): # Cleanup temp directory
            try: shutil.rmtree(temp_dir)
            except Exception as e_clean: logger.error(f"Error cleaning up temporary directory {temp_dir}: {e_clean}")

async def set_bing_cookie_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_user: return

    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(' ', 1)
        command_with_potential_mention = command_parts[0]
        if "@" in command_with_potential_mention and f"@{context.bot.username}" not in command_with_potential_mention:
            logger.info(f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring.")
            return

    user_id_str = str(update.effective_user.id)
    # Check admin and feature availability
    if not (ALLOWED_USERS and user_id_str in ALLOWED_USERS and BING_IMAGE_CREATOR_AVAILABLE): 
        await update.message.reply_text("This command is restricted to authorized administrators or is currently unavailable."); return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("Usage: /setbingcookie <new_cookie_value>"); return

    new_cookie = context.args[0]
    global BING_AUTH_COOKIE
    BING_AUTH_COOKIE = new_cookie
    logger.info(f"BING_AUTH_COOKIE updated by admin: {user_id_str}")
    # Reply handles thread ID
    await update.message.reply_text("Bing authentication cookie has been updated for the /imagine command.")

async def activate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat or not update.effective_user: return

    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(' ', 1)
        command_with_potential_mention = command_parts[0]
        if "@" in command_with_potential_mention and f"@{context.bot.username}" not in command_with_potential_mention:
            logger.info(f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring.")
            return
            
    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        await handle_permission_denied(update, context); return

    if update.effective_chat.type not in [Chat.GROUP, Chat.SUPERGROUP]:
        await update.message.reply_text("The /activate command can only be used in groups or supergroups.")
        return

    chat_id = update.effective_chat.id
    message_thread_id: Optional[int] = update.message.message_thread_id # Can be None for general group chat

    chat_topic_key = (chat_id, message_thread_id)
    topic_desc = f"this topic (ID: {message_thread_id})" if message_thread_id is not None else "this group's general chat"

    if chat_topic_key in activated_chats_topics:
        reply_text = f"I am already actively listening in {topic_desc}."
    else:
        activated_chats_topics.add(chat_topic_key)
        reply_text = f"✅ Activated! I will now respond to all messages in {topic_desc}."
        logger.info(f"Bot activated for chat {chat_id} ({topic_desc}) by user {update.effective_user.id}")
    
    await update.message.reply_text(reply_text)

async def deactivate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat or not update.effective_user: return

    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(' ', 1)
        command_with_potential_mention = command_parts[0]
        if "@" in command_with_potential_mention and f"@{context.bot.username}" not in command_with_potential_mention:
            logger.info(f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring.")
            return

    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        await handle_permission_denied(update, context); return

    if update.effective_chat.type not in [Chat.GROUP, Chat.SUPERGROUP]:
        await update.message.reply_text("The /deactivate command can only be used in groups or supergroups.")
        return

    chat_id = update.effective_chat.id
    message_thread_id: Optional[int] = update.message.message_thread_id

    chat_topic_key = (chat_id, message_thread_id)
    topic_desc = f"this topic (ID: {message_thread_id})" if message_thread_id is not None else "this group's general chat"

    if chat_topic_key in activated_chats_topics:
        activated_chats_topics.remove(chat_topic_key)
        reply_text = f"💤 Deactivated. I will no longer respond to all messages in {topic_desc}. (I'll still respond to mentions, replies, or free will)."
        logger.info(f"Bot deactivated for chat {chat_id} ({topic_desc}) by user {update.effective_user.id}")
    else:
        reply_text = f"I was not actively listening to all messages in {topic_desc} anyway."
        
    await update.message.reply_text(reply_text)
# --- END OF COMMAND HANDLERS ---

# --- Main Message Handler ---
async def process_message_entrypoint(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat or not update.effective_user:
        logger.debug("Update is missing essential message, chat, or user. Ignoring.")
        return

    # --- Ignore old messages if feature is enabled ---
    if IGNORE_OLD_MESSAGES_ON_STARTUP and BOT_STARTUP_TIMESTAMP:
        # Ensure message date is UTC for comparison
        message_date_utc = update.message.date
        if message_date_utc.tzinfo is None: # Should not happen with Telegram messages
            message_date_utc = message_date_utc.replace(tzinfo=dt_timezone.utc)
        else:
            message_date_utc = message_date_utc.astimezone(dt_timezone.utc)

        if message_date_utc < BOT_STARTUP_TIMESTAMP:
            logger.info(f"Ignoring old message (ID: {update.message.message_id}, Date: {message_date_utc}) from chat {update.effective_chat.id} received before bot startup ({BOT_STARTUP_TIMESTAMP}).")
            return
    # --- End of ignore old messages ---

    current_user = update.effective_user
    chat_id = update.effective_chat.id
    chat_type = update.effective_chat.type
    
    # Get message thread ID if it exists
    current_message_thread_id: Optional[int] = None
    if update.message.message_thread_id is not None:
        current_message_thread_id = update.message.message_thread_id
    
    user_message_text_original = update.message.text or ""
    user_message_caption_original = update.message.caption or ""
    
    # Determine content for raw log entry
    current_message_content_for_raw_log = user_message_text_original or \
                                        user_message_caption_original or \
                                        ("[Image]" if update.message.photo else \
                                         ("[Voice Message]" if update.message.voice else "[Unsupported Message Type]"))

    # Log raw message if in a group/supergroup context, respecting thread ID
    if chat_type in [Chat.GROUP, Chat.SUPERGROUP]:
        add_to_raw_group_log(chat_id, current_message_thread_id, get_display_name(current_user), current_message_content_for_raw_log)

    # Check permissions
    if not is_user_or_chat_allowed(current_user.id, chat_id):
        logger.warning(f"Ignoring message from non-allowed user ID {current_user.id} in chat ID {chat_id} (thread {current_message_thread_id})")
        return

    # Get conversation history specific to this chat and thread
    llm_history = get_llm_chat_history(chat_id, current_message_thread_id)
    user_content_parts_for_llm: List[Dict[str, Any]] = []
    
    # Context and trigger flags
    speaker_context_prefix = ""
    reply_context_prefix = ""
    should_process_message = False
    is_direct_reply_to_bot = False
    is_mention_to_bot = False
    is_free_will_triggered = False
    is_activated_chat_topic = False # New flag
    
    bot_username_at = f"@{context.bot.username}"
    text_for_trigger_check = user_message_text_original or user_message_caption_original

    # Determine if the bot should process this message based on chat type and content
    if chat_type in [Chat.GROUP, Chat.SUPERGROUP]:
        current_chat_topic_key = (chat_id, current_message_thread_id)
        # Check if this specific chat/topic is activated
        if current_chat_topic_key in activated_chats_topics:
            is_activated_chat_topic = True
            should_process_message = True
            logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Processing message because this chat/topic is activated.")

        # Check if it's a direct reply to the bot (even if not activated)
        if not should_process_message and update.message.reply_to_message and \
           update.message.reply_to_message.from_user and \
           update.message.reply_to_message.from_user.id == context.bot.id:
            is_direct_reply_to_bot = True; should_process_message = True
            logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Message is a direct reply to the bot.")

        # Check if the bot is mentioned (even if not activated or replied to)
        if not should_process_message and bot_username_at in text_for_trigger_check:
            is_mention_to_bot = True; should_process_message = True
            logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Bot was mentioned.")

        # Check for free will trigger if enabled AND not already set to process by other means
        if not should_process_message and GROUP_FREE_WILL_ENABLED and GROUP_FREE_WILL_PROBABILITY > 0:
            if random.random() < GROUP_FREE_WILL_PROBABILITY:
                is_free_will_triggered = True; should_process_message = True
                logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Free will triggered! (Prob: {GROUP_FREE_WILL_PROBABILITY:.2%}, Context: {GROUP_FREE_WILL_CONTEXT_MESSAGES} msgs)")
            else:
                logger.debug(f"Chat {chat_id} (thread {current_message_thread_id}): Free will not triggered this time.")
        
        # If none of the above, ignore the message in groups
        if not should_process_message:
            logger.debug(f"Message in group {chat_id} (thread {current_message_thread_id}) not for bot. Ignoring.")
            return
            
    elif chat_type == Chat.PRIVATE: # Always process messages in private chats
        should_process_message = True
    else: # Ignore other chat types like channels
        logger.debug(f"Message in unhandled chat_type '{chat_type}'. Ignoring.")
        return

    # --- Start building the user message content for the LLM ---
    current_speaker_display_name = get_display_name(current_user)
    # --- Get known topic name for context for both free will and direct interaction ---
    known_topic_name_for_context: Optional[str] = None
    if current_message_thread_id is not None and chat_type in [Chat.GROUP, Chat.SUPERGROUP]: # Only relevant for group topics
        known_topic_name_for_context = topic_names_cache.get((chat_id, current_message_thread_id))
    # --- END MODIFICATION ---

    if is_free_will_triggered:
        # Capture reply context specifically for free will
        replied_to_info_for_freewill: Optional[Dict[str, str]] = None
        if update.message.reply_to_message:
            replied_msg = update.message.reply_to_message
            author = get_display_name(replied_msg.from_user)
            content = replied_msg.text or replied_msg.caption or "[Media or other non-text content]"
            replied_to_info_for_freewill = {"author": author, "content": content}
            logger.info(f"Free Will trigger is a reply to '{author}'. Adding context.")
        
        # Generate context prompt based on raw log for the specific thread
        free_will_prompt = format_freewill_context_from_raw_log(
            chat_id,
            current_message_thread_id, # Pass thread ID
            GROUP_FREE_WILL_CONTEXT_MESSAGES,
            context.bot.username or SHAPESINC_SHAPE_USERNAME,
            update.effective_chat.title, # Pass current chat title
            known_topic_name_for_context, # Pass known topic name resolved above
            replied_to_message_context=replied_to_info_for_freewill
        )
        if free_will_prompt.strip():
             user_content_parts_for_llm.append({"type": "text", "text": free_will_prompt})
    else:
        # Direct interaction (DM, reply, mention, activated chat) or voice/image message
        # Add speaker context in groups (unless it's a free will, which is handled above)
        # --- Build location_addon_for_speaker_prefix ---
        location_addon_for_speaker_prefix = ""
        if chat_type in [Chat.GROUP, Chat.SUPERGROUP]:
            chat_location_description_parts = []
            group_title = update.effective_chat.title 
            if group_title:
                chat_location_description_parts.append(f"in group '{group_title}'")
        
            if update.message.is_topic_message and current_message_thread_id is not None:
                # Use the 'known_topic_name_for_context' resolved earlier
                if known_topic_name_for_context:
                    chat_location_description_parts.append(f"with topic name '{known_topic_name_for_context}'")
                else: # Fallback if topic name not in cache
                    chat_location_description_parts.append(f"with topic ID '{current_message_thread_id}' (failed to get topic name)") # Use ID as fallback
            elif group_title : 
                 chat_location_description_parts.append("the general chat area")

            if chat_location_description_parts:
                location_addon_for_speaker_prefix = f" {' '.join(chat_location_description_parts)}"
        elif chat_type == Chat.PRIVATE:
            location_addon_for_speaker_prefix = " in a private chat"
        # --- END MODIFICATION ---

        # Apply the location addon to the speaker prefix
        speaker_context_prefix = f"[Person with display name '{current_speaker_display_name}' (ID: {current_user.id}) on Telegram{location_addon_for_speaker_prefix} says:]\n"

        # Add reply context if applicable
        replied_msg: Optional[TelegramMessage] = None 
        if update.message.reply_to_message:
            replied_msg = update.message.reply_to_message
            original_author_of_replied_msg = replied_msg.from_user # Can be None for some service messages
            generate_this_reply_context = False # Default to False

            is_explicit_reply_to_our_bot = original_author_of_replied_msg and original_author_of_replied_msg.id == context.bot.id

            if is_explicit_reply_to_our_bot:
                generate_this_reply_context = True
            elif (is_mention_to_bot or is_activated_chat_topic) and replied_msg:
                # This branch means:
                #   - The bot was mentioned OR it's an activated chat/topic.
                #   - AND the current message is a reply to *some* previous message.

                # Check if the replied-to message is a type of service message we want to ignore
                # for context generation *specifically when the bot was pinged*.
                # For activated chats, we might still want context even if replying to a service message.
                is_ignorable_service_message = False
                if replied_msg:  # Ensure replied_msg is not None
                    # Store attribute names as strings, NOT direct references to replied_msg attributes
                    service_message_attributes = [
                    "forum_topic_created", "forum_topic_reopened", "forum_topic_edited", "forum_topic_closed",
                    "general_forum_topic_hidden", "general_forum_topic_unhidden", "write_access_allowed",
                    "group_chat_created", "supergroup_chat_created", "message_auto_delete_timer_changed",
                    "migrate_to_chat_id", "migrate_from_chat_id", "pinned_message", "new_chat_members",
                    "left_chat_member", "new_chat_title", "new_chat_photo", "delete_chat_photo",
                    "video_chat_scheduled", "video_chat_started", "video_chat_ended",
                    "video_chat_participants_invited", "web_app_data"
                    ]

                    # Check if any attribute exists and has a truthy value
                    if any(getattr(replied_msg, attr, False) for attr in service_message_attributes):
                        is_ignorable_service_message = True

                
                if is_mention_to_bot and is_ignorable_service_message:
                    # This is a PING, and the auto-reply (from Telegram client, or user explicitly replying to service msg)
                    # is to an ignorable service message. We don't want to generate reply context based on this service message
                    # for a simple ping, as it's usually not relevant to the ping's intent.
                    logger.info(
                        f"Chat {chat_id} (thread {current_message_thread_id}): Bot pinged. "
                        f"'reply_to_message' (ID: {replied_msg.message_id if replied_msg else 'N/A'}) "
                        f"is an ignorable service message. Suppressing this from reply context for the ping."
                    )
                    generate_this_reply_context = False # Explicitly keep it false for this scenario
                else:
                    # It's either:
                    # 1. A reply to a normal user/bot message (and bot is mentioned/activated)
                    # 2. An activated chat message (not a ping) that happens to be replying to any message (service or not)
                    # 3. A ping replying to a normal user/bot message (not an ignorable service message)
                    generate_this_reply_context = True
            
            if generate_this_reply_context and replied_msg: # Ensure replied_msg is not None again
                original_author_display_name = get_display_name(original_author_of_replied_msg)
                # Use the user's original structure for description
                original_message_content_description = replied_msg.text or replied_msg.caption or \
                                                       ("[Image]" if replied_msg.photo else \
                                                        ("[Voice Message]" if replied_msg.voice else "[replied to non-text/photo/voice message]"))
                
                max_original_content_len = 4096 # Allow longer context for replied messages
                if len(original_message_content_description) > max_original_content_len:
                    original_message_content_description = original_message_content_description[:max_original_content_len].strip() + "..."
                
                original_author_identifier = f"'{original_author_display_name}'"
                if original_author_of_replied_msg and original_author_of_replied_msg.is_bot and original_author_of_replied_msg.id == context.bot.id:
                    original_author_identifier = "your previous message"
                
                reply_context_prefix = (
                    f"[Replying to {original_author_identifier} which said: \"{original_message_content_description}\"]\n"
                )
        
        # Process the actual text/caption from the user's message
        actual_user_text = ""
        if user_message_text_original: actual_user_text = user_message_text_original
        elif user_message_caption_original: actual_user_text = user_message_caption_original
        
        # Clean mention if present (and not an activated chat processing a non-mention message)
        if is_mention_to_bot and bot_username_at in actual_user_text:
            # Replace only the first occurrence of the bot's username, case-insensitively, then strip.
            # This handles cases like "@botname some text" or "some text @botname more text".
            # Using re.IGNORECASE ensures it works regardless of how user types the bot name.
            # Using count=1 ensures only the first explicit mention is removed if there are multiple.
            cleaned_actual_text = re.sub(r'\s*' + re.escape(bot_username_at) + r'\s*', ' ', actual_user_text, count=1, flags=re.IGNORECASE).strip()
            
            # If after removing the mention, the text is empty or just whitespace,
            # AND the original message had text/caption (meaning it wasn't just an image/voice with mention),
            # then use a placeholder.
            if not cleaned_actual_text.strip() and (user_message_text_original or user_message_caption_original):
                actual_user_text = "(You were addressed directly)"
            else:
                actual_user_text = cleaned_actual_text

        # Combine context prefixes and user text
        full_text_for_llm = ""
        if speaker_context_prefix: full_text_for_llm += speaker_context_prefix
        if reply_context_prefix: full_text_for_llm += reply_context_prefix
        
        if actual_user_text: # Append the (potentially cleaned) user text
            full_text_for_llm += actual_user_text
        # Add placeholder if context exists but no user text/media, e.g., user replied with just a sticker
        elif (speaker_context_prefix or reply_context_prefix) and \
             not (update.message.photo or update.message.voice or actual_user_text.strip()): # Check actual_user_text.strip() here
            # This means some context (speaker or reply) was generated, but the current message itself
            # provided no new text, caption, photo, or voice.
            if replied_msg and not (user_message_text_original or user_message_caption_original): # Check original text for this part
                 # This specific placeholder is from your original code.
                 full_text_for_llm += "(This reply was without new text/caption)"
            # else: # Other cases of empty message with context are less common or might not need a placeholder.
            #     pass 
        
        # Add the combined text part if it's not empty
        if full_text_for_llm.strip():
            user_content_parts_for_llm.append({"type": "text", "text": full_text_for_llm.strip()})

        # --- Handle Media (Image/Voice) ---
        has_image_from_current_message = False
        has_voice_from_current_message = False

        # Add image from the current message
        if update.message.photo:
            has_image_from_current_message = True
            photo_file = await update.message.photo[-1].get_file()
            file_bytes = await photo_file.download_as_bytearray()
            base64_image = base64.b64encode(file_bytes).decode('utf-8')
            mime_type = mimetypes.guess_type(photo_file.file_path or "img.jpg")[0] or 'image/jpeg'
            user_content_parts_for_llm.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
            logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Added image_url from current message to LLM content.")

        # Add voice message URL from the current message
        if update.message.voice:
            has_voice_from_current_message = True
            voice: Voice = update.message.voice
            try:
                voice_file = await voice.get_file()
                # voice_file.file_path should be the full HTTPS URL from Telegram
                # e.g., https://api.telegram.org/file/bot<TOKEN>/path/to/file.oga
                if voice_file.file_path:
                    user_content_parts_for_llm.append({
                        "type": "audio_url",
                        "audio_url": {"url": voice_file.file_path}
                    })
                    logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Added audio_url from current message to LLM content: {voice_file.file_path}")
                else:
                    logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): Could not get file_path for current voice message.")
            except Exception as e_voice:
                logger.error(f"Chat {chat_id} (thread {current_message_thread_id}): Error processing current voice message: {e_voice}", exc_info=True)
        
        # If replying to a message (and interacting with bot OR in activated chat), add media from replied message if current message lacks it
        if replied_msg and not is_free_will_triggered and (is_mention_to_bot or is_direct_reply_to_bot or is_activated_chat_topic):
            # Add replied image if current message has no image
            if replied_msg.photo and not has_image_from_current_message:
                try:
                    photo_file_replied = await replied_msg.photo[-1].get_file()
                    file_bytes_replied = await photo_file_replied.download_as_bytearray()
                    base64_image_replied = base64.b64encode(file_bytes_replied).decode('utf-8')
                    mime_type_replied = mimetypes.guess_type(photo_file_replied.file_path or "img.jpg")[0] or 'image/jpeg'
                    # Avoid adding duplicate media types
                    if not any(p.get("type") == "image_url" for p in user_content_parts_for_llm):
                        user_content_parts_for_llm.append({"type": "image_url", "image_url": {"url": f"data:{mime_type_replied};base64,{base64_image_replied}"}})
                        logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Added image_url from replied message to LLM content.")
                        # Add placeholder text if no text content exists yet
                        # This is from your original code:
                        if not any(p.get("type") == "text" and p.get("text","").strip() for p in user_content_parts_for_llm):
                            # Construct placeholder considering existing prefixes to avoid duplicating them if they were empty.
                            current_placeholder_text = ""
                            if speaker_context_prefix.strip() and speaker_context_prefix not in (user_content_parts_for_llm[0].get("text", "") if user_content_parts_for_llm and user_content_parts_for_llm[0].get("type")=="text" else ""):
                                current_placeholder_text += speaker_context_prefix
                            if reply_context_prefix.strip() and reply_context_prefix not in (user_content_parts_for_llm[0].get("text", "") if user_content_parts_for_llm and user_content_parts_for_llm[0].get("type")=="text" else ""):
                                current_placeholder_text += reply_context_prefix
                            current_placeholder_text += "(Note: Image sent in the replied message)"
                            user_content_parts_for_llm.insert(0, {"type": "text", "text": current_placeholder_text.strip()})
                except Exception as e_img_replied:
                    logger.error(f"Chat {chat_id} (thread {current_message_thread_id}): Error processing replied image: {e_img_replied}", exc_info=True)

            # Add replied voice if current message has no voice
            if replied_msg.voice and not has_voice_from_current_message:
                try:
                    voice_replied: Voice = replied_msg.voice
                    voice_file_replied = await voice_replied.get_file()
                    if voice_file_replied.file_path:
                        # Avoid adding duplicate media types
                        if not any(p.get("type") == "audio_url" for p in user_content_parts_for_llm):
                            user_content_parts_for_llm.append({
                                "type": "audio_url",
                                "audio_url": {"url": voice_file_replied.file_path}
                            })
                            logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Added audio_url from replied message to LLM content: {voice_file_replied.file_path}")
                            # Add placeholder text if no text content exists yet
                            # This is from your original code:
                            if not any(p.get("type") == "text" and p.get("text","").strip() for p in user_content_parts_for_llm):
                                current_placeholder_text = ""
                                if speaker_context_prefix.strip() and speaker_context_prefix not in (user_content_parts_for_llm[0].get("text", "") if user_content_parts_for_llm and user_content_parts_for_llm[0].get("type")=="text" else ""):
                                    current_placeholder_text += speaker_context_prefix
                                if reply_context_prefix.strip() and reply_context_prefix not in (user_content_parts_for_llm[0].get("text", "") if user_content_parts_for_llm and user_content_parts_for_llm[0].get("type")=="text" else ""):
                                    current_placeholder_text += reply_context_prefix
                                current_placeholder_text += "(Note: Audio sent in the replied message)"
                                user_content_parts_for_llm.insert(0, {"type": "text", "text": current_placeholder_text.strip()})
                    else:
                        logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): Could not get file_path for replied voice message.")
                except Exception as e_voice_replied:
                    logger.error(f"Chat {chat_id} (thread {current_message_thread_id}): Error processing replied voice message: {e_voice_replied}", exc_info=True)
            
        # Final check: If message contains media but no text part was generated, add a placeholder text part.
        # This is your original logic block, retained as requested.
        final_has_any_media = any(p.get("type") in ["image_url", "audio_url"] for p in user_content_parts_for_llm)
        final_has_any_text = any(p.get("type") == "text" and p.get("text","").strip() for p in user_content_parts_for_llm)

        if final_has_any_media and not final_has_any_text:
            placeholder_base_text = "(Note: Media present in message)" 
            # Check if primarily audio or image to tailor the note slightly
            is_primarily_audio = any(p.get("type") == "audio_url" for p in user_content_parts_for_llm) and \
                                 not any(p.get("type") == "image_url" for p in user_content_parts_for_llm)
            is_primarily_image = any(p.get("type") == "image_url" for p in user_content_parts_for_llm) and \
                                 not any(p.get("type") == "audio_url" for p in user_content_parts_for_llm)

            if is_primarily_audio: placeholder_base_text = "(Note: An audio file is sent in this message)"
            elif is_primarily_image: placeholder_base_text = "(Note: An image is sent in this message)"
            
            # Combine with speaker/reply context if present
            final_placeholder_text = (speaker_context_prefix or "") + (reply_context_prefix or "") + placeholder_base_text
            # Ensure we don't insert an empty text part if prefixes were also empty.
            if final_placeholder_text.strip():
                user_content_parts_for_llm.insert(0, {"type": "text", "text": final_placeholder_text.strip()})
                logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Inserted placeholder for media-only: '{final_placeholder_text.strip()[:100]}'")


    # --- Final preparation before sending to LLM ---
    # Ensure there's actually content to send
    if not user_content_parts_for_llm:
        logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): No content parts generated for LLM. Trigger: reply_bot={is_direct_reply_to_bot}, mention={is_mention_to_bot}, freewill={is_free_will_triggered}, activated={is_activated_chat_topic}")
        # Only reply with error if it wasn't a silent free will attempt
        if should_process_message and (not is_free_will_triggered):
            # update.message.reply_text handles thread ID
            await update.message.reply_text("I'm not sure how to respond to that.") 
        return

    # Determine final content format (string or list of parts)
    final_llm_content: Union[str, List[Dict[str, Any]]]
    if len(user_content_parts_for_llm) == 1 and user_content_parts_for_llm[0].get("type") == "text":
        # If only one text part, send as a simple string
        final_llm_content = user_content_parts_for_llm[0]["text"]
        if not final_llm_content.strip(): # Check if the string content itself is empty
            logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): Final LLM content string is empty after processing. Not sending.")
            if should_process_message and (not is_free_will_triggered):
                await update.message.reply_text("I didn't get any text to process.") 
            return
    else:
        # Otherwise, send as a list of parts, filtering out any empty text parts that might have snuck in
        # or parts that are not dictionaries (though this should not happen with current logic)
        final_llm_content = [
            p for p in user_content_parts_for_llm 
            if isinstance(p, dict) and not (p.get("type") == "text" and not p.get("text","").strip())
        ]
        if not final_llm_content: # If list is empty after filtering
            logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): Final LLM multi-part content is empty after filtering. Not sending.")
            if should_process_message and (not is_free_will_triggered):
                 await update.message.reply_text("I didn't get any content to process.") 
            return
        
        # If only media remains after filtering, re-add a placeholder text if one wasn't added before
        # This check is more nuanced now with multiple media types.
        # The previous logic to add placeholder if (has_image or has_voice) and no text should cover this.
        # However, a final check if only media part(s) exist and no text part:
        if all(p.get("type") in ["image_url", "audio_url"] for p in final_llm_content if isinstance(p, dict)) and \
           not any(p.get("type") == "text" for p in final_llm_content if isinstance(p, dict)):
            placeholder_text_for_lone_media = "Regarding the attached media:"
            # Tailor placeholder if only audio
            if any(p.get("type") == "audio_url" for p in final_llm_content if isinstance(p,dict)) and \
               not any(p.get("type") == "image_url" for p in final_llm_content if isinstance(p,dict)):
                placeholder_text_for_lone_media = "Regarding the attached audio message:"
            
            # Combine with context prefixes if they existed but didn't form part of a text message
            final_placeholder_text_lone = ""
            if speaker_context_prefix: final_placeholder_text_lone += speaker_context_prefix
            if reply_context_prefix: final_placeholder_text_lone += reply_context_prefix 
            final_placeholder_text_lone += placeholder_text_for_lone_media
            
            if final_placeholder_text_lone.strip(): # Ensure we insert non-empty text
                final_llm_content.insert(0, {"type": "text", "text": final_placeholder_text_lone.strip()})
                logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Inserted placeholder for lone media after filtering: '{final_placeholder_text_lone.strip()[:100]}'")


    # Append the final user message (string or list) to the history
    llm_history.append({"role": "user", "content": final_llm_content}) # final_llm_content is now guaranteed to be non-empty if we reach here
    
    # Log the content being sent (with truncation)
    log_content_summary = ""
    if isinstance(final_llm_content, str):
        log_content_summary = f"Content (string): '{final_llm_content[:150].replace(chr(10), '/N')}...'"
    elif isinstance(final_llm_content, list):
        part_summaries = []
        for p_idx, p_content in enumerate(final_llm_content):
            p_type = p_content.get("type", "unknown") if isinstance(p_content, dict) else "unknown_item_type"
            p_summary_text = ""
            if isinstance(p_content, dict):
                if p_type == "text":
                    p_summary_text = p_content.get("text", "")[:200]
                elif p_type == "image_url":
                    p_summary_text = "[Base64 Image Data]" 
                elif p_type == "audio_url":
                    p_summary_text = p_content.get("audio_url", {}).get("url", "")[:200] 
                else:
                    p_summary_text = str(p_content)[:200]
            else: # Should not happen if filtering was correct
                p_summary_text = str(p_content)[:200]

            part_summaries.append(f"{p_type.capitalize()}[{p_idx}]: {p_summary_text.replace(chr(10), '/N')}...")
        log_content_summary = f"Content (multi-part): {part_summaries}"
    logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Appended user message to LLM history. {log_content_summary}. Processed due to: reply_to_bot={is_direct_reply_to_bot}, mention={is_mention_to_bot}, free_will={is_free_will_triggered}, DM={chat_type==Chat.PRIVATE}, activated={is_activated_chat_topic}")

    MAX_TOOL_ITERATIONS, current_iteration = 5, 0
    typing_task: Optional[asyncio.Task] = None
    tool_status_msg: Optional[TelegramMessage] = None
    final_text_from_llm_before_media_extraction = "I encountered an issue and couldn't generate a response. Please try again."
    escaped_text_for_splitting = "" # Will hold the text part after media extraction


    # --- START: Robust Thread ID Correction and Initial Indicator ---
    effective_send_thread_id = current_message_thread_id
    if (
        update.effective_chat.is_forum
        and update.message.reply_to_message
        and update.message.reply_to_message.message_thread_id is None
    ):
        if effective_send_thread_id is not None:
            logger.warning(
                f"Correcting invalid thread ID {effective_send_thread_id} for a reply in the General forum topic. Setting to None."
            )
            effective_send_thread_id = None

    # Start a generic "typing" indicator immediately.
    # It will run during the AI call. We may switch it later if media is detected.
    typing_task = asyncio.create_task(
        _keep_typing_loop(context, chat_id, effective_send_thread_id, action=ChatAction.TYPING)
    )
    # --- END: Robust Thread ID Correction and Initial Indicator ---

    try:
        ai_msg_obj: Optional[ChatCompletionMessage] = None
        while current_iteration < MAX_TOOL_ITERATIONS:
            current_iteration += 1
            messages_for_this_api_call = list(llm_history) # Use a copy for the API call

            # Define the Gemini-specific safety settings.
            #gemini_safety_settings = [
            #    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            #    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            #    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            #    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            #]
            
            # Put the custom Gemini parameters into the `extra_body` dictionary.
            #extra_body_params = {
            #    "safety_settings": gemini_safety_settings
            #}

            # Prepare API parameters
            api_params: Dict[str, Any] = {
                "model": f"shapesinc/{SHAPESINC_SHAPE_USERNAME}",
                "messages": messages_for_this_api_call
            }
            # Add tools if enabled and not a free-will trigger
            if ENABLE_TOOL_USE and TOOL_DEFINITIONS_FOR_API and not is_free_will_triggered:
                # Decide tool_choice: 'none' if last message was tool result, else 'auto'
                last_message_in_llm_history = llm_history[-1] if llm_history else None
                if last_message_in_llm_history and last_message_in_llm_history.get("role") == "tool":
                    api_params.update({"tools": TOOL_DEFINITIONS_FOR_API, "tool_choice": "none"})
                else:
                    api_params.update({"tools": TOOL_DEFINITIONS_FOR_API, "tool_choice": "auto"})
            elif is_free_will_triggered: # Explicitly disable tools for free will
                api_params.pop("tools", None)
                api_params.pop("tool_choice", None)
            
            # Log API call details (more robust logging)
            try: 
                logged_msgs_sample_parts = []
                num_messages_to_log_sample = min(3, len(api_params["messages"]))
                for msg_param_idx, msg_param_any_type in enumerate(api_params["messages"][-num_messages_to_log_sample:]): 
                    # Handle potential non-dict items (though unlikely with current structure)
                    if not isinstance(msg_param_any_type, dict):
                        logged_msgs_sample_parts.append(f"Msg {len(api_params['messages']) - num_messages_to_log_sample + msg_param_idx}: Non-dict item - {str(msg_param_any_type)[:100]}") 
                        continue
                    msg_dict_for_log = dict(msg_param_any_type) # Work with a copy
                    # Summarize content
                    if "content" in msg_dict_for_log:
                        content_val = msg_dict_for_log["content"]
                        if isinstance(content_val, str):
                            msg_dict_for_log["content"] = content_val[:500] + ('...' if len(content_val) > 500 else '')
                        elif isinstance(content_val, list): # Summarize multi-part content
                            summarized_parts = []
                            for part_content_idx, part_content in enumerate(content_val):
                                if isinstance(part_content, dict):
                                    part_type_str = part_content.get("type", "unknown_part_type")
                                    summary_text_val = ""
                                    if part_type_str == "text":
                                        summary_text_val = part_content.get('text','')[:200].replace(chr(10),'/N')
                                    elif part_type_str == "image_url":
                                        summary_text_val = "[Base64 Image]" 
                                    elif part_type_str == "audio_url":
                                        summary_text_val = part_content.get('audio_url',{}).get('url','')[:200].replace(chr(10),'/N')
                                    else: 
                                        summary_text_val = str(part_content)[:200].replace(chr(10),'/N')
                                    summarized_parts.append(f"{part_type_str.capitalize()}[{part_content_idx}]: {summary_text_val}...")
                                else: # if part_content is not a dict
                                    summarized_parts.append(f"Part[{part_content_idx}]: {str(part_content)[:200]}...")
                            msg_dict_for_log["content"] = summarized_parts
                    # Summarize tool calls
                    if "tool_calls" in msg_dict_for_log and isinstance(msg_dict_for_log.get("tool_calls"), list):
                        tool_calls_list = msg_dict_for_log["tool_calls"]
                        num_tc = len(tool_calls_list)
                        tc_names_sample = [
                            (tc.get('function', {}).get('name', '?') if isinstance(tc, dict) else '?')
                            for tc in tool_calls_list[:2]
                        ]
                        msg_dict_for_log["tool_calls"] = f"<{num_tc} tool_calls: {tc_names_sample}...>"
                    logged_msgs_sample_parts.append(f"Msg {len(api_params['messages']) - num_messages_to_log_sample + msg_param_idx}: {msg_dict_for_log}") 
                messages_log_str = "\n".join(logged_msgs_sample_parts)
            except Exception as log_e_inner: # Fallback logging if summary fails
                raw_sample_str = [str(m)[:150] for m in api_params["messages"][-2:]] 
                messages_log_str = f"Error in detailed message logging: {log_e_inner}. Raw sample: {raw_sample_str}"

            # Add custom headers for API call
            custom_headers_for_api = {
                "X-User-Id": str(current_user.id),
                "X-Channel-Id": str(chat_id) 
            }
            # Include thread ID in headers if present
            if current_message_thread_id is not None: 
                 custom_headers_for_api["X-Thread-Id"] = str(current_message_thread_id)

            logger.info(
                f"API Call (iter {current_iteration}/{MAX_TOOL_ITERATIONS}, FreeWill={is_free_will_triggered}) to {api_params['model']} for chat {chat_id} (thread {current_message_thread_id}). "
                f"Tool choice: {api_params.get('tool_choice', 'N/A')}. LLM History len: {len(llm_history)}. "
                f"Custom Headers: {custom_headers_for_api}. \n"
                f"API Messages (sample):\n{messages_log_str}"
            )

            # Execute the API call
            response_from_ai = await aclient_shape.chat.completions.create(
                model=api_params["model"],
                messages=api_params["messages"],
                tools=api_params.get("tools"),
                tool_choice=api_params.get("tool_choice"),
                #extra_body=extra_body_params,
                extra_headers=custom_headers_for_api
            )

            ai_msg_obj = response_from_ai.choices[0].message
            # Append AI's response (potentially including tool calls) to history
            llm_history.append(ai_msg_obj.model_dump(exclude_none=True)) 
            logger.debug(f"Chat {chat_id} (thread {current_message_thread_id}): Appended assistant's response to LLM history. Last item: {str(llm_history[-1])[:250].replace(chr(10), '/N')}...")

            # --- Handle Tool Calls ---
            if ai_msg_obj.tool_calls:
                # Check if tools are actually enabled globally
                if not ENABLE_TOOL_USE: 
                    logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): AI attempted tool use, but ENABLE_TOOL_USE is false. Tool calls: {ai_msg_obj.tool_calls}")
                    final_text_from_llm_before_media_extraction = "I tried to use a special tool, but it's currently disabled. Please ask in a different way."
                    llm_history[-1] = {"role": "assistant", "content": final_text_from_llm_before_media_extraction} # Overwrite tool call message
                    break # Exit tool loop
                if is_free_will_triggered: # Check if tools should be disabled for this specific call (free will)
                    logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): AI attempted tool use during free will, but tools are disabled for free will. Ignoring tool call.")
                    final_text_from_llm_before_media_extraction = ai_msg_obj.content or "I had a thought but it involved a tool I can't use for spontaneous comments. Never mind!"
                    llm_history[-1] = {"role": "assistant", "content": final_text_from_llm_before_media_extraction} # Overwrite tool call
                    break # Exit tool loop

                # Send a "using tools" status message if not already sent
                if not tool_status_msg and chat_id:
                    tool_names_str = ", ".join(sorted(list(set(tc.function.name for tc in ai_msg_obj.tool_calls if tc.function and tc.function.name))))
                    status_text = f"🛠️ Activating tools: {tool_names_str}..."
                    try: 
                        # Use helper to send status message, respecting thread
                        tool_status_msg = await send_message_to_chat_or_general(
                            context.bot, chat_id, status_text, preferred_thread_id=current_message_thread_id
                        )
                    except Exception as e_send_status: 
                        logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): Failed to send tool status message: {e_send_status}")

                # Execute each tool call
                tool_results_for_history: list[ChatCompletionMessageParam] = []
                for tool_call in ai_msg_obj.tool_calls:
                    func_name, tool_call_id, args_str = tool_call.function.name, tool_call.id, tool_call.function.arguments
                    logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): AI requests tool: '{func_name}' (ID: {tool_call_id}) with raw args: {args_str}")
                    tool_content_result = f"Error: Tool '{func_name}' execution failed or tool is unknown." # Default error
                    # Check if the requested tool function exists
                    if func_name in AVAILABLE_TOOLS_PYTHON_FUNCTIONS:
                        try:
                            py_func = AVAILABLE_TOOLS_PYTHON_FUNCTIONS[func_name]
                            # Parse arguments safely
                            parsed_args: Dict[str, Any] = {}
                            if args_str and args_str.strip(): 
                                parsed_args = json.loads(args_str)
                            else: 
                                logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): Tool '{func_name}' called with empty/null arguments string. Raw: '{args_str}'. Proceeding with empty dict if function allows.")
                            # Ensure parsed args are a dictionary
                            if not isinstance(parsed_args, dict): 
                                raise TypeError(f"Parsed arguments for tool '{func_name}' are not a dictionary. Got {type(parsed_args)} from '{args_str}'")
                            
                            # Inject necessary context for specific tools
                            kwargs_for_tool = parsed_args.copy()
                            if func_name == "create_poll_in_chat":
                                kwargs_for_tool["telegram_bot_context"] = context
                                kwargs_for_tool["current_chat_id"] = chat_id
                                kwargs_for_tool["current_message_thread_id"] = current_message_thread_id

                            elif func_name in ["restrict_user_in_chat", "get_user_info"]:
                                kwargs_for_tool["telegram_bot_context"] = context
                                kwargs_for_tool["current_chat_id"] = chat_id
                            # Add similar blocks if other tools need context in the future
                            
                            # Execute the tool function (async or sync)
                            output = await py_func(**kwargs_for_tool) if asyncio.iscoroutinefunction(py_func) else await asyncio.to_thread(py_func, **kwargs_for_tool)
                            tool_content_result = str(output)
                            logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Tool '{func_name}' executed. Output snippet: {tool_content_result[:200].replace(chr(10), ' ')}")
                        except (json.JSONDecodeError, TypeError, ValueError) as e_parse_args:
                            # Handle errors during argument parsing or calling with wrong types
                            err_msg = f"Error parsing arguments or calling tool '{func_name}': {e_parse_args}. Raw args: '{args_str}'"
                            logger.error(f"Chat {chat_id} (thread {current_message_thread_id}): {err_msg}", exc_info=True); tool_content_result = err_msg
                        except Exception as e_tool_exec: 
                            # Handle unexpected errors during tool execution
                            err_msg = f"Unexpected error executing tool '{func_name}': {e_tool_exec}"
                            logger.error(f"Chat {chat_id} (thread {current_message_thread_id}): {err_msg}", exc_info=True); tool_content_result = err_msg
                    else: 
                        # Handle case where AI hallucinated a tool name
                        logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): AI requested unknown tool: '{func_name}'")
                        tool_content_result = f"Error: Tool '{func_name}' is not available."
                    
                    # Prepare the tool result message for the LLM history
                    tool_results_for_history.append({"tool_call_id": tool_call_id, "role": "tool", "name": func_name, "content": tool_content_result}) 
                
                # Add all tool results to the history for the next API call
                llm_history.extend(tool_results_for_history)
                logger.debug(f"Chat {chat_id} (thread {current_message_thread_id}): Extended LLM history with {len(tool_results_for_history)} tool results.")
                # Loop continues for the next API call with tool results provided
            # --- Handle Final Text Response ---
            elif ai_msg_obj.content is not None:
                # If the AI provided text content directly, this is the final answer
                final_text_from_llm_before_media_extraction = str(ai_msg_obj.content)
                logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): AI final text response (iter {current_iteration}): '{final_text_from_llm_before_media_extraction[:120].replace(chr(10), ' ')}...'")
                break # Exit the tool loop, we have the final response
            # --- Handle Empty/Unusual Response ---
            else:
                # If the AI response has neither tool calls nor content
                logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): AI response (iter {current_iteration}) had no tool_calls and content was None. Response: {ai_msg_obj.model_dump_json(indent=2)}")
                final_text_from_llm_before_media_extraction = "AI provided an empty or unusual response. Please try rephrasing."
                llm_history[-1] = {"role": "assistant", "content": final_text_from_llm_before_media_extraction} # Overwrite empty response
                break # Exit loop with error message
        
        # --- Handle Max Iterations Reached ---
        # If loop finished without a final text response (e.g., max tool calls)
        if current_iteration >= MAX_TOOL_ITERATIONS and not (ai_msg_obj and ai_msg_obj.content is not None):
            logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): Max tool iterations ({MAX_TOOL_ITERATIONS}) reached without final content from AI.")
            final_text_from_llm_before_media_extraction = "I tried using my tools multiple times but couldn't get a final answer. Could you try rephrasing your request or ask in a different way?"
            # Append this error message to history if it's not already the last assistant message
            if not (llm_history and llm_history[-1].get("role") == "assistant" and llm_history[-1].get("content") == final_text_from_llm_before_media_extraction): 
                 llm_history.append({"role": "assistant", "content": final_text_from_llm_before_media_extraction}) 

        # --- Clean up Status Messages ---
        # Delete the "using tools" status message if it was sent
        if tool_status_msg and chat_id: # tool_status_msg is a TelegramMessage object
            try: await tool_status_msg.delete()
            except Exception as e_del: logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): Could not delete tool status msg ID {tool_status_msg.message_id}: {e_del}")

        # --- START OF MEDIA URL DETECTION (BEFORE TEXT SENDING) ---
        image_urls_to_send: List[str] = []
        audio_urls_to_send: List[str] = []
        text_part_after_media_extraction = final_text_from_llm_before_media_extraction # Start with the full AI response

        # Define regex patterns
        image_url_pattern = re.compile(r"(https://files\.shapes\.inc/[\w.-]+\.(?:png|jpg|jpeg|gif|webp))\b", re.IGNORECASE)
        audio_url_pattern = re.compile(r"(https://files\.shapes\.inc/[\w.-]+\.(?:mp3|ogg|wav|m4a|flac))\b", re.IGNORECASE)

        all_media_matches_in_response: List[Dict[str, Any]] = []
        for pattern, media_type_label in [(image_url_pattern, "image"), (audio_url_pattern, "audio")]:
            for match_obj in pattern.finditer(text_part_after_media_extraction):
                all_media_matches_in_response.append({
                    "start": match_obj.start(), "end": match_obj.end(),
                    "url": match_obj.group(0), "type": media_type_label
                })
        
        all_media_matches_in_response.sort(key=lambda m: m["start"]) 
        
        plain_text_segments_after_extraction: List[str] = []
        last_char_index_processed = 0
        for media_item_match_info in all_media_matches_in_response:
            if media_item_match_info["start"] > last_char_index_processed:
                plain_text_segments_after_extraction.append(text_part_after_media_extraction[last_char_index_processed:media_item_match_info["start"]])
            if media_item_match_info["type"] == "image": image_urls_to_send.append(media_item_match_info["url"])
            elif media_item_match_info["type"] == "audio": audio_urls_to_send.append(media_item_match_info["url"])
            last_char_index_processed = media_item_match_info["end"]
        if last_char_index_processed < len(text_part_after_media_extraction):
            plain_text_segments_after_extraction.append(text_part_after_media_extraction[last_char_index_processed:])
            
        text_part_after_media_extraction = "".join(plain_text_segments_after_extraction)
        
        # <<< FIX #2: Clean up empty markdown link artifacts >>>
        text_part_after_media_extraction = re.sub(r'\[([^\]]*)\]\(\s*\)', r'\1', text_part_after_media_extraction)
        
        text_part_after_media_extraction = re.sub(r'(\r\n|\r|\n){2,}', '\n', text_part_after_media_extraction).strip()

        if image_urls_to_send: logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Extracted image URLs: {image_urls_to_send}")
        if audio_urls_to_send: logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Extracted audio URLs: {audio_urls_to_send}")
        # --- END OF MEDIA URL DETECTION ---

        # --- INDICATOR SWITCH ---
        # If the AI response contains media, cancel the "typing" indicator
        # and start the appropriate "uploading" indicator.
        if image_urls_to_send or audio_urls_to_send:
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass # This is expected

            action_for_media = ChatAction.UPLOAD_PHOTO if image_urls_to_send else ChatAction.UPLOAD_VOICE
            logger.info(f"Switching to media indicator: {action_for_media}")

            # Send the action ONCE explicitly *before* starting the loop.
            # This guarantees the indicator appears, defeating the race condition.
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=action_for_media, message_thread_id=effective_send_thread_id)
            except Exception as e:
                logger.warning(f"Could not send initial media chat action '{action_for_media}': {e}")
            
            # Now, create the new background task to KEEP the indicator alive while we send.
            typing_task = asyncio.create_task(
                _keep_typing_loop(context, chat_id, effective_send_thread_id, action=action_for_media)
            )
        # --- END: INDICATOR SWITCH ---

        # Ensure final text is not empty or just whitespace IF IT'S THE ONLY THING
        if not text_part_after_media_extraction.strip() and not (image_urls_to_send or audio_urls_to_send):
            logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): Final AI text was empty or whitespace (and no media extracted). Defaulting to error message.")
            text_part_after_media_extraction = "I'm sorry, I couldn't generate a valid response for that. Please try again."
            # Update history if last message was empty assistant content and we are overriding it
            if llm_history and llm_history[-1].get("role") == "assistant" and not llm_history[-1].get("content","").strip(): 
                llm_history[-1]["content"] = text_part_after_media_extraction
        
        # Prepare for sending the remaining text part, if any
        if text_part_after_media_extraction.strip():
            logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Text part for user (before media sending): >>>{text_part_after_media_extraction[:200].replace(chr(10), '/N')}...<<<")
            escaped_text_for_splitting = telegram_markdown_v2_escape(text_part_after_media_extraction)
            logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): FULLY ESCAPED text part for user: >>>{escaped_text_for_splitting[:200].replace(chr(10), '/N')}...<<<")
        else:
            escaped_text_for_splitting = "" # No text to send
        
        if escaped_text_for_splitting.strip():
            max_mdv2_len = 4096 
            send_thread_id = current_message_thread_id # Use the determined thread ID for sending
            
            # Wrapper for sending a single message chunk with fallback logic
            async def attempt_send_one_message_wrapper(
                bot_obj: Bot, chat_id_val: int, text_content: str,
                preferred_thread_id_val: Optional[int], try_markdown: bool
            ):
                parse_mode_to_try = ParseMode.MARKDOWN_V2 if try_markdown else None
                try:
                    await send_message_to_chat_or_general(bot_obj, chat_id_val, text_content, preferred_thread_id=preferred_thread_id_val, parse_mode=parse_mode_to_try)
                    return True 
                except telegram.error.BadRequest as e_send:
                    logger.error(f"Chat {chat_id_val} (attempted thread {preferred_thread_id_val}, then general): Send failed with {'MDv2' if try_markdown else 'Plain'}. Error: {e_send}. Text snippet: '{text_content[:100]}'")
                    if try_markdown: return False 
                    else: raise e_send 
                except Exception as e_other_send_error:
                    logger.error(f"Chat {chat_id_val}: Unexpected error during attempt_send_one_message_wrapper: {e_other_send_error}", exc_info=True)
                    raise e_other_send_error

            if len(escaped_text_for_splitting) <= max_mdv2_len: 
                if not await attempt_send_one_message_wrapper(context.bot, chat_id, escaped_text_for_splitting, send_thread_id, try_markdown=True):
                    logger.info(f"Chat {chat_id} (thread {send_thread_id}): MDv2 failed for single message, falling back to plain text: {text_part_after_media_extraction[:100]}")
                    try:
                        await attempt_send_one_message_wrapper(context.bot, chat_id, text_part_after_media_extraction, send_thread_id, try_markdown=False)
                    except Exception as e_final_plain_send:
                        logger.error(f"Chat {chat_id} (thread {send_thread_id}): Final plain text send also failed: {e_final_plain_send}")
                        await send_message_to_chat_or_general(context.bot, chat_id, "Error formatting and sending my response. (S_ULT_FAIL)", preferred_thread_id=send_thread_id)
            else: 
                logger.info(f"Chat {chat_id} (thread {send_thread_id}): Escaped text too long ({len(escaped_text_for_splitting)}), using intelligent splitting.")
                message_parts_to_send = split_message_with_markdown_balancing(escaped_text_for_splitting, max_mdv2_len, logger)
                if not message_parts_to_send and escaped_text_for_splitting.strip(): 
                    logger.warning(f"Chat {chat_id} (thread {send_thread_id}): Splitter returned no parts for non-empty text. Sending truncated.")
                    safe_truncate_len = 4096 - 150 
                    trunc_txt = escaped_text_for_splitting[:safe_truncate_len] + "\n\\[MESSAGE_TRUNCATED_SPLIT_FAIL\\]"
                    message_parts_to_send.append(trunc_txt[:max_mdv2_len])
                for i, part_chunk_md_escaped in enumerate(message_parts_to_send): 
                    current_chunk_to_send = part_chunk_md_escaped.strip()
                    if not current_chunk_to_send:
                        temp_strip_styles = part_chunk_md_escaped
                        for delim_s in _BALANCING_MARKDOWN_DELIMITERS: temp_strip_styles = temp_strip_styles.replace(delim_s, "")
                        if not temp_strip_styles.strip(): 
                            logger.info(f"Chat {chat_id} (thread {send_thread_id}): Skipping empty or style-only part {i+1} from splitter."); continue
                    if len(current_chunk_to_send) > max_mdv2_len: 
                        safe_truncate_len = 4096 - 150 
                        logger.warning(f"Chat {chat_id} (thread {send_thread_id}): Split part {i+1} still too long ({len(current_chunk_to_send)}). Truncating.")
                        current_chunk_to_send = current_chunk_to_send[:safe_truncate_len] + "\n\\[MESSAGE_PART_TRUNCATED\\]"
                        current_chunk_to_send = current_chunk_to_send[:max_mdv2_len] 
                    logger.info(f"Chat {chat_id} (thread {send_thread_id}): Sending MDv2 part {i+1}/{len(message_parts_to_send)} (len: {len(current_chunk_to_send)}): '{current_chunk_to_send[:200].replace(chr(10),'/N')}...'")
                    if not await attempt_send_one_message_wrapper(context.bot, chat_id, current_chunk_to_send, send_thread_id, try_markdown=True):
                        logger.info(f"Chat {chat_id} (thread {send_thread_id}): MDv2 failed for split part {i+1}, falling back to plain text for this part.")
                        try:
                            await attempt_send_one_message_wrapper(context.bot, chat_id, current_chunk_to_send, send_thread_id, try_markdown=False)
                        except Exception as e_final_plain_chunk_send:
                            logger.error(f"Chat {chat_id} (thread {send_thread_id}): Final plain text send for chunk {i+1} also failed: {e_final_plain_chunk_send}")
                            await send_message_to_chat_or_general(context.bot, chat_id, f"[Problem sending part {i+1} of my response. (C_ULT_FAIL)]", preferred_thread_id=send_thread_id)
                    if i < len(message_parts_to_send) - 1: await asyncio.sleep(0.75) 

        # --- START OF MEDIA SENDING (MOVED TO AFTER TEXT SENDING) ---
        # Send Images if any were extracted
        if image_urls_to_send:
            for img_url in image_urls_to_send:
                logger.info(f"Chat {chat_id} (thread {effective_send_thread_id}): Sending image from URL: {img_url}")
                try:
                    await send_photo_to_chat_or_general(context.bot, chat_id, img_url, preferred_thread_id=effective_send_thread_id)
                    await asyncio.sleep(0.5)
                except Exception as e_send_img:
                    logger.error(f"Chat {chat_id} (thread {effective_send_thread_id}): Failed to send image {img_url}. Error: {e_send_img}", exc_info=True)

        # Send Audio files if any were extracted
        if audio_urls_to_send:
            for audio_url in audio_urls_to_send:
                logger.info(f"Chat {chat_id} (thread {effective_send_thread_id}): Sending audio from URL: {audio_url}")
                try:
                    await send_audio_to_chat_or_general(context.bot, chat_id, audio_url, preferred_thread_id=effective_send_thread_id)
                    await asyncio.sleep(0.5)
                except Exception as e_send_audio:
                    logger.error(f"Chat {chat_id} (thread {effective_send_thread_id}): Failed to send audio {audio_url}. Error: {e_send_audio}", exc_info=True)
        # --- END OF MEDIA SENDING ---

    # --- Specific Exception Handling ---
    # Use imported InternalServerError and APITimeoutError directly from openai
    except InternalServerError as e_openai_ise: 
        logger_chat_thread_info = f"Chat {chat_id} (thread {current_message_thread_id})"
        # Handle specific 504 Gateway Timeout from the API
        if hasattr(e_openai_ise, 'response') and e_openai_ise.response and e_openai_ise.response.status_code == 504:
            logger.error(f"{logger_chat_thread_info}: Received 504 Gateway Timeout from Shapes API.", exc_info=True) 
            final_text_to_send_to_user = "The AI is taking a bit too long to generate a response and timed out. You could try asking for something more concise, or try again in a moment."
        else: # Handle other internal server errors
            logger.error(f"{logger_chat_thread_info}: OpenAI InternalServerError: {e_openai_ise}", exc_info=True)
            final_text_to_send_to_user = "Sorry, the AI service encountered an internal error. Please try again later."
        
        # Append error message to history if different from last assistant message
        if not (llm_history and llm_history[-1].get("role") == "assistant" and llm_history[-1].get("content") == final_text_to_send_to_user):
                llm_history.append({"role": "assistant", "content": final_text_to_send_to_user})
        # Notify user
        if chat_id: 
            try:
                await send_message_to_chat_or_general(context.bot, chat_id, final_text_to_send_to_user, preferred_thread_id=current_message_thread_id)
            except Exception as e_send_err: logger.error(f"Error sending ISE message: {e_send_err}")

    except APITimeoutError as e_openai_timeout:
        # Handle client-side timeout waiting for the API
        logger.error(f"Chat {chat_id} (thread {current_message_thread_id}): OpenAI APITimeoutError. Client-side timeout for Shapes API.", exc_info=True)
        final_text_to_send_to_user = "The AI is taking too long to respond and the request timed out on my side. Please try asking for something shorter or try again later."
        # Append error message to history if different
        if not (llm_history and llm_history[-1].get("role") == "assistant" and llm_history[-1].get("content") == final_text_to_send_to_user):
             llm_history.append({"role": "assistant", "content": final_text_to_send_to_user})
        # Notify user
        if chat_id: 
            try:
                await send_message_to_chat_or_general(context.bot, chat_id, final_text_to_send_to_user, preferred_thread_id=current_message_thread_id)
            except Exception as e_send_err: logger.error(f"Error sending Timeout message: {e_send_err}")

    # --- Telegram Specific Error Handling ---
    except telegram.error.BadRequest as e_outer_tg_badreq: 
        # This catch block is for BadRequest errors that might occur *outside* the robust sending logic 
        # (e.g., during initial chat action send, or if the sending logic itself had an issue causing a re-raise).
        # It can also catch errors if the final plain text fallback send fails within the logic above.
        logger.error(f"Outer Telegram BadRequest for chat {chat_id} (thread {current_message_thread_id}): {e_outer_tg_badreq}. Raw AI response: '{final_text_from_llm_before_media_extraction[:500]}'", exc_info=True)
        try: 
            plain_fb_full_outer = final_text_from_llm_before_media_extraction # Use the original AI response
            # Re-attempt sending the *entire* original message as plain text directly to general chat as a last resort.
            if len(plain_fb_full_outer) <= 4096: # Telegram's absolute max length
                await context.bot.send_message(chat_id, text=plain_fb_full_outer, message_thread_id=None) # Try general explicitly
            else: 
                # If even plain text is too long, split it for general chat
                logger.warning(f"Chat {chat_id} (thread {current_message_thread_id}): Outer plain text fallback also too long ({len(plain_fb_full_outer)}), splitting for general.")
                num_chunks_outer = (len(plain_fb_full_outer) + 4096 -1) // 4096
                for i, chunk in enumerate([plain_fb_full_outer[j:j+4096] for j in range(0, len(plain_fb_full_outer), 4096)]):
                    hdr = f"[Fallback Part {i+1}/{num_chunks_outer}]\n" if num_chunks_outer > 1 else ""
                    await context.bot.send_message(chat_id, text=hdr + chunk, message_thread_id=None) # Try general explicitly
                    if i < num_chunks_outer -1 : await asyncio.sleep(0.5) # Brief pause between chunks
            logger.info(f"Chat {chat_id} (thread {current_message_thread_id}): Sent entire message as plain text to general (outer error fallback).")
        except Exception as e_fb_outer_send: 
            # If sending the outer plain text fallback *also* fails
            logger.error(f"Chat {chat_id} (thread {current_message_thread_id}): Outer plain text fallback send also failed: {e_fb_outer_send}")
            # Send a very generic error message to general chat
            try:
                await context.bot.send_message(chat_id, "A general error occurred while formatting my response. (OBRF)", message_thread_id=None)
            except Exception as e_final_send_err : logger.error(f"Even final OBRF message failed: {e_final_send_err}")

    # --- Network/Communication Error Handling ---
    except (httpx.NetworkError, httpx.TimeoutException, httpx.ConnectError, telegram.error.NetworkError, telegram.error.TimedOut) as e_net_comm:
        # Catch common network errors from both httpx (API calls) and telegram (Bot communication)
        logger.error(f"Network-related error for chat {chat_id} (thread {current_message_thread_id}): {e_net_comm}", exc_info=False) # exc_info=False as traceback might be less useful here
        # Notify the user about network issues if possible
        if chat_id: 
            try:
                await send_message_to_chat_or_general(context.bot, chat_id, "⚠️ I'm having some network issues. Please try again in a little while.", preferred_thread_id=current_message_thread_id)
            except Exception as e_send_net_err_msg: # If sending the notification itself fails
                 logger.error(f"Chat {chat_id} (thread {current_message_thread_id}): Failed to send network error notification: {e_send_net_err_msg}")

    # --- General Catch-All Exception Handling ---
    except Exception as e_main_handler: 
        # Catch any other unexpected error within the main handler
        logger.error(f"General, unhandled error in process_message_entrypoint for chat {chat_id} (thread {current_message_thread_id}): {e_main_handler}", exc_info=True)
        # Notify the user about the generic error if possible
        if chat_id: 
            try:
                await send_message_to_chat_or_general(context.bot, chat_id, "😵‍💫 Oops! Something went wrong. I've noted it. Please try again. (MGEN)", preferred_thread_id=current_message_thread_id)
            except Exception as e_send_gen_err_msg: # If sending the notification itself fails
                logger.error(f"Chat {chat_id} (thread {current_message_thread_id}): Failed to send general error notification: {e_send_gen_err_msg}")
    finally:
        # Ensure typing indicator is always cancelled
        if typing_task and not typing_task.done():
            typing_task.cancel()
            logger.debug(f"Chat {chat_id} (thread {current_message_thread_id}): Typing task cancelled in finally block.")
# --- END OF Main Message Handler ---

# --- ERROR HANDLER ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Log the error with traceback
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = "".join(tb_list)
    
    # Format update object for logging
    update_str = "Update data not available or not an Update instance."
    effective_chat_id_for_error = "N/A" # Default
    message_thread_id_for_error: Optional[int] = None

    if isinstance(update, Update):
        try: update_str = json.dumps(update.to_dict(), indent=2, ensure_ascii=False, default=str)
        except Exception: update_str = str(update) # Fallback if to_dict fails
        if update.effective_chat:
            effective_chat_id_for_error = str(update.effective_chat.id)
        # Get thread ID from effective_message if available (covers commands, etc.)
        if update.effective_message and update.effective_message.message_thread_id is not None: 
            message_thread_id_for_error = update.effective_message.message_thread_id
    elif update: # If update is not None but not an Update instance
        update_str = str(update)

    # Format context data (truncated)
    chat_data_str = str(context.chat_data)[:500] if context.chat_data else "N/A"
    user_data_str = str(context.user_data)[:500] if context.user_data else "N/A"
    
    # Prepare error message for admin (plain text)
    thread_info_for_error_log = f"(thread {message_thread_id_for_error})" if message_thread_id_for_error is not None else ""
    send_message_plain = (
        f"Bot Exception in chat {effective_chat_id_for_error} {thread_info_for_error_log}:\n"
        f"Update: {update_str[:1500]}...\n\n"
        f"Chat Data: {chat_data_str}\nUser Data: {user_data_str}\n\n"
        f"Traceback (last 1500 chars):\n{tb_string[-1500:]}"
    )

    # Determine admin chat ID for notification
    chat_id_to_notify_admin: Optional[int] = None
    if ALLOWED_USERS: # Check if ALLOWED_USERS is populated
        try: chat_id_to_notify_admin = int(ALLOWED_USERS[0]) # Attempt to get the first admin user ID
        except (ValueError, IndexError, TypeError): logger.error("No valid admin user ID for error notification from ALLOWED_USERS[0].")

    # --- Notify User (Optional) ---
    user_notified_by_handler = False
    if isinstance(update, Update) and update.effective_chat:
        # Avoid sending generic error if it's a network/API timeout, as those are handled in process_message_entrypoint
        # Check against imported exceptions
        if not isinstance(context.error, (InternalServerError, APITimeoutError, telegram.error.NetworkError, httpx.NetworkError, httpx.TimeoutException)): # Added httpx.TimeoutException
            try:
                # Send a user-friendly HTML message
                user_error_message = f"<b>Bot Error:</b> <pre>{html.escape(str(context.error))}</pre>\n<i>An unexpected error occurred. The admin has been notified if configured.</i>"
                if len(user_error_message) <= 4096: # Telegram message length limit
                    # Use send_message_to_chat_or_general for user notification too, respecting thread
                    await send_message_to_chat_or_general(
                        context.bot, 
                        update.effective_chat.id, 
                        user_error_message, 
                        preferred_thread_id=message_thread_id_for_error, # Pass thread ID
                        parse_mode=ParseMode.HTML
                    )
                    user_notified_by_handler = True
            except Exception as e_send_user_err:
                logger.error(f"Failed to send user-friendly error to chat {update.effective_chat.id} (thread {message_thread_id_for_error}): {e_send_user_err}")

    # --- Notify Admin ---
    if chat_id_to_notify_admin:
        # Avoid duplicate notification if admin is the one who experienced the error and was already notified
        if user_notified_by_handler and update.effective_chat and update.effective_chat.id == chat_id_to_notify_admin:
            logger.info(f"Admin was the user in chat {chat_id_to_notify_admin} (thread {message_thread_id_for_error}) and already notified about the error. Skipping redundant admin report.")
        else:
            max_len = 4096 # Telegram message length limit
            try:
                # Prefer HTML for admin if error is short and not potentially full of HTML itself
                # Use imported InternalServerError
                is_potentially_html_error = isinstance(context.error, (InternalServerError, httpx.HTTPStatusError)) # These might contain HTML in response
                
                thread_info_html = f"(thread {message_thread_id_for_error})" if message_thread_id_for_error is not None else ""
                # Admin messages always go to the admin's direct chat, so no thread_id needed for the send_message call below
                
                # Try sending a short HTML version first if suitable
                if not is_potentially_html_error and len(send_message_plain) < max_len - 200 : # If plain text is short enough for HTML wrapper
                    short_html_err = f"<b>Bot Error in chat {effective_chat_id_for_error} {thread_info_html}:</b>\n<pre>{html.escape(str(context.error))}</pre>\n<i>(Full details in server logs. Update/TB follows if space.)</i>"
                    if len(short_html_err) <=max_len : # Check if HTML version is within limits
                         await context.bot.send_message(chat_id=chat_id_to_notify_admin, text=short_html_err, parse_mode=ParseMode.HTML)
                    else: # HTML version too long, revert to plain
                        is_potentially_html_error = True # Force plain text path 

                # Send as plain text if it's long or potentially contains HTML
                if is_potentially_html_error or len(send_message_plain) >= max_len -200 : 
                    if len(send_message_plain) <= max_len:
                        await context.bot.send_message(chat_id=chat_id_to_notify_admin, text=send_message_plain)
                    else: # Split long plain text message for admin
                        num_err_chunks = (len(send_message_plain) + max_len -1) // max_len
                        for i_err, chunk in enumerate([send_message_plain[j:j+max_len] for j in range(0, len(send_message_plain), max_len)]):
                            # Add context header to each chunk
                            hdr = f"[BOT ERR Pt {i_err+1}/{num_err_chunks} Chat {effective_chat_id_for_error} {thread_info_for_error_log}]\n" if num_err_chunks > 1 else f"[BOT ERR Chat {effective_chat_id_for_error} {thread_info_for_error_log}]\n"
                            # Ensure chunk with header doesn't exceed max_len
                            await context.bot.send_message(chat_id=chat_id_to_notify_admin, text=(hdr + chunk)[:max_len]) 
                            if i_err < num_err_chunks -1 : await asyncio.sleep(0.5) # Brief pause between chunks
            except Exception as e_send_err: # Fallback if sending detailed error fails
                logger.error(f"Failed sending detailed error report to admin {chat_id_to_notify_admin}: {e_send_err}")
                # Send a minimal plain text error to admin
                try: await context.bot.send_message(chat_id=chat_id_to_notify_admin, text=f"Bot Error in chat {effective_chat_id_for_error} {thread_info_for_error_log}: {str(context.error)[:1000]}\n(Details in server logs. Report sending failed.)")
                except Exception as e_final_fb: logger.error(f"Final admin error report fallback failed: {e_final_fb}")
    # Log if no notification was sent anywhere
    elif not user_notified_by_handler: 
        logger.error(f"No chat ID found to send error message via Telegram (admin not set, or user already got specific error from main handler). Error details for chat {effective_chat_id_for_error} {thread_info_for_error_log} logged to server.")
# --- END OF ERROR HANDLER ---

async def post_initialization(application: Application) -> None:
    """Actions to perform after the bot is initialized, like setting commands."""
    bot_commands = [
        BotCommand("start", "Display the welcome message."),
        BotCommand("help", "Show this help message."),
        BotCommand("newchat", "Clear conversation history for this topic/chat."),
        BotCommand("activate", "(Groups/Topics) Respond to all messages here."),
        BotCommand("deactivate", "(Groups/Topics) Stop responding to all messages here."),
    ]
    if BING_IMAGE_CREATOR_AVAILABLE and BING_AUTH_COOKIE: # Only show /imagine if fully configured
        bot_commands.append(BotCommand("imagine", "Generate images from a prompt (Bing)."))
    
    # setbingcookie is an admin command, only show if BING is available and there are admins
    if BING_IMAGE_CREATOR_AVAILABLE and ALLOWED_USERS: 
         bot_commands.append(BotCommand("setbingcookie", "(Admin) Update Bing auth cookie."))
    
    try:
        await application.bot.set_my_commands(bot_commands)
        logger.info(f"Successfully set bot commands: {[cmd.command for cmd in bot_commands]}")
    except Exception as e:
        logger.error(f"Failed to set bot commands: {e}")


if __name__ == "__main__":
    # Record bot startup time (UTC)
    BOT_STARTUP_TIMESTAMP = datetime.now(dt_timezone.utc)
    logger.info(f"Bot starting up at: {BOT_STARTUP_TIMESTAMP}")
    if IGNORE_OLD_MESSAGES_ON_STARTUP:
        logger.info("Bot will ignore messages received before this startup time.")

    # Increase timeouts for network requests, especially for sending files
    # Default is 5s, which is too short for uploads. We'll set a longer read/write timeout.
    app_builder = ApplicationBuilder().token(TELEGRAM_TOKEN).connect_timeout(10.0).read_timeout(60.0).write_timeout(60.0) # Increased timeouts
    
    app_builder.post_init(post_initialization) # Register post_init hook
    app = app_builder.build()

    # Command Handlers
    # The CommandHandler itself, by default, correctly handles /command@botname scenarios.
    # The internal checks added to each command handler are an additional layer to ensure
    # the bot doesn't respond to commands explicitly aimed at *another* bot if somehow
    # the CommandHandler still triggered (e.g., if the command string was just "command"
    # but the text contained "@otherbot" later, though this is unlikely for command parsing).
    # The main benefit of the internal check is slightly more explicit logging.
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("newchat", new_chat_command))
    app.add_handler(CommandHandler("activate", activate_command))
    app.add_handler(CommandHandler("deactivate", deactivate_command))
    # Only add imagine/setbingcookie if library is available
    if BING_IMAGE_CREATOR_AVAILABLE: 
        # Only add /imagine if cookie is also set initially
        if BING_AUTH_COOKIE: 
            app.add_handler(CommandHandler("imagine", imagine_command))
        # Admin can set cookie even if not initially present
        app.add_handler(CommandHandler("setbingcookie", set_bing_cookie_command)) 

    # --- Handler for Forum Topic Updates (Created/Edited) ---
    # This handler is specifically for populating the topic_names_cache.
    app.add_handler(MessageHandler(
        filters.StatusUpdate.FORUM_TOPIC_CREATED | filters.StatusUpdate.FORUM_TOPIC_EDITED,
        handle_forum_topic_updates
    ))
    # Message Handler for text, photos, voice, and replies (but not commands)
    app.add_handler(MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.VOICE | filters.REPLY) & (~filters.COMMAND) & (~filters.StatusUpdate.ALL),
        process_message_entrypoint
    ))

    # Error Handler
    app.error_handler = error_handler

    logger.info("Bot is starting to poll for updates...")
    try:
        # Start polling
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    # Specific catch for initial network errors during startup
    except telegram.error.NetworkError as ne: 
        logger.critical(f"CRITICAL: Initial NetworkError during polling setup: {ne}. Check network/token.", exc_info=True)
    # Catch any other critical errors during startup/polling loop
    except Exception as main_e: 
        logger.critical(f"CRITICAL: Unhandled exception at main polling level: {main_e}", exc_info=True)