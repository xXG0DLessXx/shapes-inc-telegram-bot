#!/usr/bin/env python3

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
from datetime import datetime, timedelta
import pytz
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote as url_unquote


# Telegram imports
from telegram import Update, InputMediaPhoto, Message as TelegramMessage, User as TelegramUser, Chat, Voice
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
)
from telegram.constants import ChatAction, ParseMode
import telegram.error

# OpenAI imports
from openai import AsyncOpenAI
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
except Exception as e:
    logger.error(f"Failed to init Shapes client: {e}"); raise

chat_histories: dict[int, list[ChatCompletionMessageParam]] = {}
MAX_HISTORY_LENGTH = 10
group_raw_message_log: Dict[int, List[Dict[str, str]]] = {}
MAX_RAW_LOG_PER_GROUP = 50
# --- END OF Global Config & Setup ---

# --- TOOL IMPLEMENTATIONS ---
async def create_poll_tool(
    question: str,
    options: List[str],
    is_anonymous: bool = True,
    allows_multiple_answers: bool = False,
    # Parameters to be passed by the main handler:
    telegram_bot_context: Optional[ContextTypes.DEFAULT_TYPE] = None,
    current_chat_id: Optional[int] = None
) -> str:
    logger.info(f"TOOL: create_poll_tool for chat_id {current_chat_id} with question='{question}', options={options}")

    if not telegram_bot_context or not current_chat_id:
        err_msg = "Telegram context or chat ID not provided to create_poll_tool."
        logger.error(err_msg)
        return json.dumps({"error": err_msg, "details": "This tool requires internal context to function."})

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

        await telegram_bot_context.bot.send_poll(
            chat_id=current_chat_id,
            question=question,
            options=unique_options,
            is_anonymous=is_anonymous,
            allows_multiple_answers=allows_multiple_answers
        )
        logger.info(f"Poll sent to chat {current_chat_id}: '{question}'")
        return json.dumps({
            "status": "poll_created_successfully",
            "question": question,
            "options_sent": unique_options,
            "chat_id": current_chat_id
        })
    except telegram.error.BadRequest as e:
        logger.error(f"Telegram BadRequest creating poll in chat {current_chat_id}: {e}", exc_info=True)
        return json.dumps({"error": "Failed to create poll.", "details": f"Telegram API error: {e.message}"})
    except Exception as e:
        logger.error(f"Unexpected error in create_poll_tool for chat {current_chat_id}: {e}", exc_info=True)
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

async def web_search(query: str, site: str = '', region: str = '', date_filter: str = '') -> str:
    logger.info(f"TOOL: web_search for query='{query}', site='{site}', region='{region}', date_filter='{date_filter}'")
    params = {
        'q': query,
        'b': site,
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
# --- END OF TOOL IMPLEMENTATIONS ---

AVAILABLE_TOOLS_PYTHON_FUNCTIONS = {
    "calculator": calculator_tool,
    "get_weather": get_weather_tool,
    "web_search": web_search,
    "create_poll_in_chat": create_poll_tool,
}
TOOL_DEFINITIONS_FOR_API: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates a mathematical expression. Supports +, -, *, /, ** (exponentiation), and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2+2', '(5*8-3)/2', '2^10')."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather data for a location with flexible timeframes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g., London, UK or a specific address."
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ['current', 'hourly', 'daily'],
                        "default": 'current',
                        "description": "Time resolution of the forecast: 'current' for current conditions, 'hourly' for hour-by-hour, 'daily' for day summaries."
                    },
                    "hours_ahead": {
                        "type": "number",
                        "description": "Optional. For 'hourly' timeframe, specifies the number of hours from now for which to get a single forecast point (e.g., 0 for current hour, 1 for next hour). Max 167."
                    },
                    "forecast_days": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 14,
                        "default": 1,
                        "description": "Number of days to forecast (for daily or hourly if hours_ahead is not specified). E.g., 1 for today, 7 for a week."
                    },
                    "unit": {
                        "type": "string",
                        "enum": ['celsius', 'fahrenheit'],
                        "default": 'celsius',
                        "description": "Temperature unit."
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information on a given topic. Fetches only the first page of search results from DuckDuckGo. Use when you require information you are unsure or unaware of.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g., 'What is the capital of France?'"
                    },
                    "site": {
                        "type": "string",
                        "description": "Optional. Limit search to a specific website (e.g., 'wikipedia.org') or use a DuckDuckGo bang (e.g., '!w' for Wikipedia). This is passed to DuckDuckGo's 'b' (bang) parameter. Leave empty for general search."
                    },
                    "region": {
                        "type": "string",
                        "description": "Optional. Limit search to results from a specific region/language (e.g., 'us-en' for US English, 'de-de' for Germany German). This is a DuckDuckGo region code passed to 'kl' parameter. Leave empty for global results."
                    },
                    "date_filter": {
                        "type": "string",
                        "description": "Optional. Filter search results by date: 'd' (past day), 'w' (past week), 'm' (past month), 'y' (past year). Passed to DuckDuckGo's 'df' parameter. Leave empty for no date filter.",
                        "enum": ['', 'd', 'w', 'm', 'y']
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_poll_in_chat",
            "description": "Creates a new poll in the current Telegram chat. This is useful for asking questions with multiple choice answers to the group or user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question for the poll. Max 300 characters."
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 10,
                        "description": "A list of 2 to 10 answer options for the poll. Each option max 100 characters."
                    },
                    "is_anonymous": {
                        "type": "boolean",
                        "default": True,
                        "description": "Optional. If true, the poll is anonymous. Defaults to true."
                    },
                    "allows_multiple_answers": {
                        "type": "boolean",
                        "default": False,
                        "description": "Optional. If true, users can select multiple answers. Defaults to false."
                    }
                },
                "required": ["question", "options"]
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
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, you are not authorized to use this command or interact with me.")

def get_llm_chat_history(chat_id: int) -> list[ChatCompletionMessageParam]:
    if chat_id not in chat_histories: chat_histories[chat_id] = []
    if len(chat_histories[chat_id]) > MAX_HISTORY_LENGTH * 7: 
        logger.info(f"Trimming LLM chat history for {chat_id} from {len(chat_histories[chat_id])} to {MAX_HISTORY_LENGTH * 5}")
        chat_histories[chat_id] = chat_histories[chat_id][-(MAX_HISTORY_LENGTH * 5):]
    return chat_histories[chat_id]

def add_to_raw_group_log(chat_id: int, sender_name: str, text: str):
    if chat_id not in group_raw_message_log:
        group_raw_message_log[chat_id] = []
    log_entry = {"sender_name": sender_name, "text": text if text else "[empty message content]"}
    group_raw_message_log[chat_id].append(log_entry)
    if len(group_raw_message_log[chat_id]) > MAX_RAW_LOG_PER_GROUP:
        group_raw_message_log[chat_id] = group_raw_message_log[chat_id][-MAX_RAW_LOG_PER_GROUP:]
        logger.debug(f"Trimmed raw message log for group {chat_id} to {len(group_raw_message_log[chat_id])} messages.")

def format_freewill_context_from_raw_log(
    chat_id: int,
    num_messages_to_include: int,
    bot_name: str
    ) -> str:
    if chat_id not in group_raw_message_log or not group_raw_message_log[chat_id] or num_messages_to_include <= 0:
        return "[Recent conversation context is minimal or unavailable.]\n"
    log = group_raw_message_log[chat_id]
    start_index = max(0, len(log) - num_messages_to_include)
    context_messages_to_format = log[start_index:]
    if not context_messages_to_format:
         return "[No prior messages in raw log to form context.]\n"

    formatted_context_parts = ["[Recent conversation excerpt:]"]
    triggering_user_name = "Unknown User"
    triggering_message_text = "[message content not available]"

    for i, msg_data in enumerate(context_messages_to_format):
        sender = msg_data.get("sender_name", "Unknown User")
        text = msg_data.get("text", "[message content not available]")
        max_len_per_msg_in_context = 250
        if len(text) > max_len_per_msg_in_context:
            text = text[:max_len_per_msg_in_context].strip() + "..."
        formatted_context_parts.append(f"- '{sender}' said: \"{text}\"")
        if i == len(context_messages_to_format) - 1:
            triggering_user_name = sender
            triggering_message_text = msg_data.get("text", "[message content not available]")
            if len(triggering_message_text) > 500:
                triggering_message_text = triggering_message_text[:500].strip() + "..."

    formatted_context_parts.append(
        f"\n[You are '{bot_name}', chatting on Telegram. Based on the excerpt above, where '{triggering_user_name}' "
        f"just said: \"{triggering_message_text}\", "
        "make a relevant and in character interjection or comment. Be concise and natural.]"
    )
    return "\n".join(formatted_context_parts) + "\n\n"

async def _keep_typing_loop(context: ContextTypes.DEFAULT_TYPE, chat_id: int, action: str = ChatAction.TYPING, interval: float = 4.5):
    while True:
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=action)
            await asyncio.sleep(interval)
        except asyncio.CancelledError: break
        except Exception as e:
            logger.warning(f"Error sending {action} action in loop for chat {chat_id}: {e}")
            await asyncio.sleep(interval) # Continue loop even on error, but log it
# --- END OF UTILITY FUNCTIONS ---

# --- COMMAND HANDLERS ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.effective_chat: return
    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        await handle_permission_denied(update, context); return
    start_message = (
        f"Hello! I am {SHAPESINC_SHAPE_USERNAME}, chatting here on Telegram! "
        "I can chat, use tools, and even understand images and voice messages.\n\n"
        "Type /help to see a list of commands."
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=start_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.effective_chat: return
    help_text_parts = [
        "Here are the available commands:",
        "/start - Display the welcome message.",
        "/help - Show this help message.",
        "/newchat - Clear the current conversation history and start fresh."
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
        help_text_parts.append(f"\nGroup Free Will is enabled! I might respond randomly about {GROUP_FREE_WILL_PROBABILITY:.1%} of the time, considering the last ~{GROUP_FREE_WILL_CONTEXT_MESSAGES} messages.")

    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        help_text_parts.append("\n\nNote: Your access to interact with me is currently restricted.")

    escaped_help_text = telegram_markdown_v2_escape("\n".join(help_text_parts))
    await context.bot.send_message(chat_id=update.effective_chat.id, text=escaped_help_text, parse_mode=ParseMode.MARKDOWN_V2)

async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.effective_chat or not update.message: return
    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        await handle_permission_denied(update, context); return
    chat_id = update.effective_chat.id
    cleared_any = False
    if chat_id in chat_histories and chat_histories[chat_id]:
        chat_histories[chat_id] = []
        logger.info(f"LLM Conversation history cleared for chat ID: {chat_id}")
        cleared_any = True
    if chat_id in group_raw_message_log: # Also clear raw log if it exists for the chat
        group_raw_message_log[chat_id] = []
        logger.info(f"Raw group message log cleared for chat ID: {chat_id}")
        cleared_any = True
    
    if cleared_any:
        await update.message.reply_text("✨ Conversation history cleared! Let's start a new topic.")
    else:
        await update.message.reply_text("There's no conversation history to clear yet.")

async def imagine_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not update.effective_chat or not update.message: return
    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        await handle_permission_denied(update, context); return
    if not (BING_IMAGE_CREATOR_AVAILABLE and ImageGen and BING_AUTH_COOKIE):
        await update.message.reply_text("The /imagine command is currently unavailable or not configured. Please contact an admin."); return
    if not context.args: await update.message.reply_text("Please provide a prompt for the image. Usage: /imagine <your image prompt>"); return

    prompt = " ".join(context.args)
    chat_id=update.effective_chat.id
    typing_task: Optional[asyncio.Task] = None
    status_msg: Optional[TelegramMessage] = None
    temp_dir = f"temp_bing_images_{chat_id}_{random.randint(1000,9999)}" # Unique temp dir

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
        typing_task = asyncio.create_task(_keep_typing_loop(context, chat_id, action=ChatAction.UPLOAD_PHOTO, interval=5.0))
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
            await context.bot.send_media_group(chat_id=chat_id, media=media_photos)
            if status_msg: await status_msg.delete() # Clean up status message
        else:
            err_msg_no_proc = "Sorry, no images could be processed or sent from Bing."
            if status_msg: await status_msg.edit_text(err_msg_no_proc)
            else: await update.message.reply_text(err_msg_no_proc)

    except Exception as e:
        logger.error(f"Error during /imagine command for prompt '{prompt}': {e}", exc_info=True)
        err_text = "An error occurred while generating images with Bing. Please try again later."
        try:
            if status_msg: await status_msg.edit_text(err_text)
            else: await context.bot.send_message(chat_id, err_text) # Fallback if status_msg failed to send
        except Exception: await context.bot.send_message(chat_id, err_text) # Ultimate fallback
    finally:
        if typing_task and not typing_task.done(): typing_task.cancel()
        if os.path.exists(temp_dir): # Cleanup temp directory
            try: shutil.rmtree(temp_dir)
            except Exception as e_clean: logger.error(f"Error cleaning up temporary directory {temp_dir}: {e_clean}")

async def set_bing_cookie_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not update.message: return
    user_id_str = str(update.effective_user.id)
    if not (ALLOWED_USERS and user_id_str in ALLOWED_USERS and BING_IMAGE_CREATOR_AVAILABLE): # Check admin and feature availability
        await update.message.reply_text("This command is restricted to authorized administrators or is currently unavailable."); return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("Usage: /setbingcookie <new_cookie_value>"); return

    new_cookie = context.args[0]
    global BING_AUTH_COOKIE
    BING_AUTH_COOKIE = new_cookie
    logger.info(f"BING_AUTH_COOKIE updated by admin: {user_id_str}")
    await update.message.reply_text("Bing authentication cookie has been updated for the /imagine command.")
# --- END OF COMMAND HANDLERS ---

# --- Main Message Handler ---
async def process_message_entrypoint(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat or not update.effective_user:
        logger.debug("Update is missing essential message, chat, or user. Ignoring.")
        return

    current_user = update.effective_user
    chat_id = update.effective_chat.id
    chat_type = update.effective_chat.type
    
    user_message_text_original = update.message.text or ""
    user_message_caption_original = update.message.caption or ""
    
    current_message_content_for_raw_log = user_message_text_original or \
                                        user_message_caption_original or \
                                        ("[Image]" if update.message.photo else \
                                         ("[Voice Message]" if update.message.voice else "[Unsupported Message Type]"))

    if chat_type in [Chat.GROUP, Chat.SUPERGROUP]:
        add_to_raw_group_log(chat_id, get_display_name(current_user), current_message_content_for_raw_log)

    if not is_user_or_chat_allowed(current_user.id, chat_id):
        logger.warning(f"Ignoring message from non-allowed user ID {current_user.id} in chat ID {chat_id}")
        return

    llm_history = get_llm_chat_history(chat_id)
    user_content_parts_for_llm: List[Dict[str, Any]] = []
    
    speaker_context_prefix = ""
    reply_context_prefix = ""
    should_process_message = False
    is_direct_reply_to_bot = False
    is_mention_to_bot = False
    is_free_will_triggered = False
    
    bot_username_at = f"@{context.bot.username}"
    text_for_trigger_check = user_message_text_original or user_message_caption_original

    if chat_type in [Chat.GROUP, Chat.SUPERGROUP]:
        if update.message.reply_to_message and \
           update.message.reply_to_message.from_user and \
           update.message.reply_to_message.from_user.id == context.bot.id:
            is_direct_reply_to_bot = True; should_process_message = True
            logger.info(f"Chat {chat_id}: Message is a direct reply to the bot.")

        if not should_process_message and bot_username_at in text_for_trigger_check:
            is_mention_to_bot = True; should_process_message = True
            logger.info(f"Chat {chat_id}: Bot was mentioned.")

        if not should_process_message and GROUP_FREE_WILL_ENABLED and GROUP_FREE_WILL_PROBABILITY > 0:
            if random.random() < GROUP_FREE_WILL_PROBABILITY:
                is_free_will_triggered = True; should_process_message = True
                logger.info(f"Chat {chat_id}: Free will triggered! (Prob: {GROUP_FREE_WILL_PROBABILITY:.2%}, Context: {GROUP_FREE_WILL_CONTEXT_MESSAGES} msgs)")
            else:
                logger.debug(f"Chat {chat_id}: Free will not triggered this time.")
        
        if not should_process_message:
            logger.debug(f"Message in group {chat_id} not for bot. Ignoring.")
            return
            
    elif chat_type == Chat.PRIVATE:
        should_process_message = True
    else:
        logger.debug(f"Message in unhandled chat_type '{chat_type}'. Ignoring.")
        return

    current_speaker_display_name = get_display_name(current_user)

    if is_free_will_triggered:
        free_will_prompt = format_freewill_context_from_raw_log(
            chat_id,
            GROUP_FREE_WILL_CONTEXT_MESSAGES,
            context.bot.username or SHAPESINC_SHAPE_USERNAME
        )
        if free_will_prompt.strip():
             user_content_parts_for_llm.append({"type": "text", "text": free_will_prompt})
    else: # Direct interaction (DM, reply, mention) or voice/image message
        if chat_type in [Chat.GROUP, Chat.SUPERGROUP]:
            speaker_context_prefix = f"[User '{current_speaker_display_name}' (ID: {current_user.id}) on Telegram says:]\n"

        replied_msg: Optional[TelegramMessage] = None 
        if update.message.reply_to_message:
            replied_msg = update.message.reply_to_message
            original_author_of_replied_msg = replied_msg.from_user
            generate_this_reply_context = False
            if original_author_of_replied_msg and original_author_of_replied_msg.id == context.bot.id:
                generate_this_reply_context = True
            elif is_mention_to_bot and update.message.reply_to_message: # Also add context if mentioning bot while replying to someone else
                generate_this_reply_context = True
            
            if generate_this_reply_context:
                original_author_display_name = get_display_name(original_author_of_replied_msg)
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
        
        actual_user_text = ""
        if user_message_text_original: actual_user_text = user_message_text_original
        elif user_message_caption_original: actual_user_text = user_message_caption_original
        
        if is_mention_to_bot and bot_username_at in actual_user_text:
            cleaned_actual_text = re.sub(r'\s*' + re.escape(bot_username_at) + r'\s*', ' ', actual_user_text).strip()
            if not cleaned_actual_text and (user_message_text_original or user_message_caption_original):
                actual_user_text = "(You were addressed directly)"
            else:
                actual_user_text = cleaned_actual_text

        full_text_for_llm = ""
        if speaker_context_prefix: full_text_for_llm += speaker_context_prefix
        if reply_context_prefix: full_text_for_llm += reply_context_prefix
        
        if actual_user_text:
            full_text_for_llm += actual_user_text
        elif (speaker_context_prefix or reply_context_prefix) and not (update.message.photo or update.message.voice):
             # Only add this if there's no other content like photo/voice and some context prefix exists
            if update.message.reply_to_message and not (user_message_text_original or user_message_caption_original):
                 full_text_for_llm += "(User replied without new text/caption)"
            # else: # This case is less likely to be meaningful without media
            #      full_text_for_llm += "(User sent a message without new text/caption)"
        
        if full_text_for_llm.strip():
            user_content_parts_for_llm.append({"type": "text", "text": full_text_for_llm.strip()})

        has_image_from_current_message = False
        has_voice_from_current_message = False

        if update.message.photo:
            has_image_from_current_message = True
            photo_file = await update.message.photo[-1].get_file()
            file_bytes = await photo_file.download_as_bytearray()
            base64_image = base64.b64encode(file_bytes).decode('utf-8')
            mime_type = mimetypes.guess_type(photo_file.file_path or "img.jpg")[0] or 'image/jpeg'
            user_content_parts_for_llm.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
            logger.info(f"Chat {chat_id}: Added image_url from current message to LLM content.")

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
                    logger.info(f"Chat {chat_id}: Added audio_url from current message to LLM content: {voice_file.file_path}")
                else:
                    logger.warning(f"Chat {chat_id}: Could not get file_path for current voice message.")
            except Exception as e_voice:
                logger.error(f"Chat {chat_id}: Error processing current voice message: {e_voice}", exc_info=True)
        
        if replied_msg and not is_free_will_triggered and (is_mention_to_bot or is_direct_reply_to_bot):
            if replied_msg.photo and not has_image_from_current_message:
                try:
                    photo_file_replied = await replied_msg.photo[-1].get_file()
                    file_bytes_replied = await photo_file_replied.download_as_bytearray()
                    base64_image_replied = base64.b64encode(file_bytes_replied).decode('utf-8')
                    mime_type_replied = mimetypes.guess_type(photo_file_replied.file_path or "img.jpg")[0] or 'image/jpeg'
                    if not any(p.get("type") == "image_url" for p in user_content_parts_for_llm):
                        user_content_parts_for_llm.append({"type": "image_url", "image_url": {"url": f"data:{mime_type_replied};base64,{base64_image_replied}"}})
                        logger.info(f"Chat {chat_id}: Added image_url from replied message to LLM content.")
                        if not any(p.get("type") == "text" and p.get("text","").strip() for p in user_content_parts_for_llm):
                            placeholder_text = (speaker_context_prefix or "") + (reply_context_prefix or "") + "(Note: Image sent in the replied message)"
                            user_content_parts_for_llm.insert(0, {"type": "text", "text": placeholder_text.strip()})
                except Exception as e_img_replied:
                    logger.error(f"Chat {chat_id}: Error processing replied image: {e_img_replied}", exc_info=True)

            if replied_msg.voice and not has_voice_from_current_message:
                try:
                    voice_replied: Voice = replied_msg.voice
                    voice_file_replied = await voice_replied.get_file()
                    if voice_file_replied.file_path:
                        if not any(p.get("type") == "audio_url" for p in user_content_parts_for_llm):
                            user_content_parts_for_llm.append({
                                "type": "audio_url",
                                "audio_url": {"url": voice_file_replied.file_path}
                            })
                            logger.info(f"Chat {chat_id}: Added audio_url from replied message to LLM content: {voice_file_replied.file_path}")
                            if not any(p.get("type") == "text" and p.get("text","").strip() for p in user_content_parts_for_llm):
                                placeholder_text = (speaker_context_prefix or "") + (reply_context_prefix or "") + "(Note: Audio sent in the replied message)"
                                user_content_parts_for_llm.insert(0, {"type": "text", "text": placeholder_text.strip()})
                    else:
                        logger.warning(f"Chat {chat_id}: Could not get file_path for replied voice message.")
                except Exception as e_voice_replied:
                    logger.error(f"Chat {chat_id}: Error processing replied voice message: {e_voice_replied}", exc_info=True)
            
        final_has_any_media = any(p.get("type") in ["image_url", "audio_url"] for p in user_content_parts_for_llm)
        final_has_any_text = any(p.get("type") == "text" and p.get("text","").strip() for p in user_content_parts_for_llm)

        if final_has_any_media and not final_has_any_text:
            placeholder_base_text = "(Note: Media present in message)" 
            is_primarily_audio = any(p.get("type") == "audio_url" for p in user_content_parts_for_llm) and \
                                 not any(p.get("type") == "image_url" for p in user_content_parts_for_llm)
            is_primarily_image = any(p.get("type") == "image_url" for p in user_content_parts_for_llm) and \
                                 not any(p.get("type") == "audio_url" for p in user_content_parts_for_llm)

            if is_primarily_audio: placeholder_base_text = "(Note: An audio file is sent in this message)"
            elif is_primarily_image: placeholder_base_text = "(Note: An image is sent in this message)"
            
            final_placeholder_text = (speaker_context_prefix or "") + (reply_context_prefix or "") + placeholder_base_text
            user_content_parts_for_llm.insert(0, {"type": "text", "text": final_placeholder_text.strip()})


    if not user_content_parts_for_llm:
        logger.warning(f"Chat {chat_id}: No content parts generated for LLM. Trigger: reply_bot={is_direct_reply_to_bot}, mention={is_mention_to_bot}, freewill={is_free_will_triggered}")
        if should_process_message and (not is_free_will_triggered):
             await update.message.reply_text("I'm not sure how to respond to that.")
        return

    final_llm_content: Union[str, List[Dict[str, Any]]]
    if len(user_content_parts_for_llm) == 1 and user_content_parts_for_llm[0].get("type") == "text":
        final_llm_content = user_content_parts_for_llm[0]["text"]
        if not final_llm_content.strip():
            logger.warning(f"Chat {chat_id}: Final LLM content string is empty. Not sending.")
            if should_process_message and (not is_free_will_triggered):
                await update.message.reply_text("I didn't get any text to process.")
            return
    else:
        final_llm_content = [p for p in user_content_parts_for_llm if not (p.get("type") == "text" and not p.get("text","").strip())]
        if not final_llm_content:
            logger.warning(f"Chat {chat_id}: Final LLM multi-part content is empty after filtering. Not sending.")
            if should_process_message and (not is_free_will_triggered):
                 await update.message.reply_text("I didn't get any content to process.")
            return
        
        # If only media remains after filtering, re-add a placeholder text if one wasn't added before
        # This check is more nuanced now with multiple media types.
        # The previous logic to add placeholder if (has_image or has_voice) and no text should cover this.
        # However, a final check if only media part(s) exist and no text part:
        if all(p.get("type") in ["image_url", "audio_url"] for p in final_llm_content) and \
           not any(p.get("type") == "text" for p in final_llm_content):
            placeholder_text_for_lone_media = "Regarding the attached media:"
            if any(p.get("type") == "audio_url" for p in final_llm_content) and \
               not any(p.get("type") == "image_url" for p in final_llm_content):
                placeholder_text_for_lone_media = "Please transcribe and respond to the attached audio message:"
            final_placeholder_text_lone = (speaker_context_prefix or "") + (reply_context_prefix or "") + placeholder_text_for_lone_media
            final_llm_content.insert(0, {"type": "text", "text": final_placeholder_text_lone.strip()})

    llm_history.append({"role": "user", "content": final_llm_content})
    
    log_content_summary = ""
    if isinstance(final_llm_content, str):
        log_content_summary = f"Content (string): '{final_llm_content[:150].replace(chr(10), '/N')}...'"
    elif isinstance(final_llm_content, list):
        # Improved logging for multi-part including audio
        part_summaries = []
        for p_idx, p_content in enumerate(final_llm_content):
            p_type = p_content.get("type", "unknown")
            p_summary_text = ""
            if p_type == "text":
                p_summary_text = p_content.get("text", "")[:70]
            elif p_type == "image_url":
                p_summary_text = "[Base64 Image Data]" # Or log part of URL if not base64
            elif p_type == "audio_url":
                p_summary_text = p_content.get("audio_url", {}).get("url", "")[:70] # Log part of URL
            else:
                p_summary_text = str(p_content)[:70]
            part_summaries.append(f"{p_type.capitalize()}[{p_idx}]: {p_summary_text.replace(chr(10), '/N')}...")
        log_content_summary = f"Content (multi-part): {part_summaries}"
    logger.info(f"Chat {chat_id}: Appended user message to LLM history. {log_content_summary}. Processed due to: reply_to_bot={is_direct_reply_to_bot}, mention={is_mention_to_bot}, free_will={is_free_will_triggered}, DM={chat_type==Chat.PRIVATE}")

    typing_task: Optional[asyncio.Task] = None
    initial_action = ChatAction.TYPING
    # Check for image_url or audio_url type for initial action NOT NEEDED BECAUSE THE AI REPLIES WITH TEXT!
    #
    #if isinstance(final_llm_content, list):
    #    if any(isinstance(part, dict) and part.get("type") == "image_url" for part in final_llm_content):
    #        initial_action = ChatAction.UPLOAD_PHOTO
    #        logger.debug(f"Chat {chat_id}: Setting initial chat action to UPLOAD_PHOTO due to image content.")
    #    elif any(isinstance(part, dict) and part.get("type") == "audio_url" for part in final_llm_content):
    #        initial_action = ChatAction.UPLOAD_VOICE # Using UPLOAD_VOICE
    #        logger.debug(f"Chat {chat_id}: Setting initial chat action to UPLOAD_VOICE due to audio content.")

    await context.bot.send_chat_action(chat_id, initial_action)
    typing_task = asyncio.create_task(_keep_typing_loop(context, chat_id, action=initial_action))

    MAX_TOOL_ITERATIONS, current_iteration = 5, 0
    tool_status_msg: Optional[TelegramMessage] = None
    final_text_to_send_to_user = "I encountered an issue and couldn't generate a response. Please try again."
    escaped_text_for_splitting = ""

    try:
        ai_msg_obj: Optional[ChatCompletionMessage] = None
        while current_iteration < MAX_TOOL_ITERATIONS:
            current_iteration += 1
            messages_for_this_api_call = list(llm_history)
            api_params: Dict[str, Any] = {
                "model": f"shapesinc/{SHAPESINC_SHAPE_USERNAME}",
                "messages": messages_for_this_api_call
            }
            if ENABLE_TOOL_USE and TOOL_DEFINITIONS_FOR_API and not is_free_will_triggered:
                last_message_in_llm_history = llm_history[-1] if llm_history else None
                if last_message_in_llm_history and last_message_in_llm_history.get("role") == "tool":
                    api_params.update({"tools": TOOL_DEFINITIONS_FOR_API, "tool_choice": "none"})
                else:
                    api_params.update({"tools": TOOL_DEFINITIONS_FOR_API, "tool_choice": "auto"})
            elif is_free_will_triggered:
                api_params.pop("tools", None)
                api_params.pop("tool_choice", None)
            
            # Logging for API call content (more robust for different types)
            try: 
                logged_msgs_sample_parts = []
                num_messages_to_log_sample = min(3, len(api_params["messages"]))
                for msg_param_idx, msg_param_any_type in enumerate(api_params["messages"][-num_messages_to_log_sample:]): 
                    if not isinstance(msg_param_any_type, dict):
                        logged_msgs_sample_parts.append(f"Msg {len(api_params['messages']) - num_messages_to_log_sample + msg_param_idx}: Non-dict item - {str(msg_param_any_type)[:100]}") 
                        continue
                    msg_dict_for_log = dict(msg_param_any_type) 
                    if "content" in msg_dict_for_log:
                        content_val = msg_dict_for_log["content"]
                        if isinstance(content_val, str):
                            msg_dict_for_log["content"] = content_val[:100] + ('...' if len(content_val) > 100 else '')
                        elif isinstance(content_val, list): 
                            summarized_parts = []
                            for part_content_idx, part_content in enumerate(content_val):
                                part_type_str = part_content.get("type", "unknown_part_type")
                                summary_text_val = ""
                                if part_type_str == "text":
                                    summary_text_val = part_content.get('text','')[:70].replace(chr(10),'/N')
                                elif part_type_str == "image_url":
                                    summary_text_val = "[Base64 Image]" # Or part_content.get('image_url',{}).get('url','')[:70]
                                elif part_type_str == "audio_url":
                                    summary_text_val = part_content.get('audio_url',{}).get('url','')[:70].replace(chr(10),'/N')
                                else: 
                                    summary_text_val = str(part_content)[:70].replace(chr(10),'/N')
                                summarized_parts.append(f"{part_type_str.capitalize()}[{part_content_idx}]: {summary_text_val}...")
                            msg_dict_for_log["content"] = summarized_parts
                    if "tool_calls" in msg_dict_for_log and isinstance(msg_dict_for_log.get("tool_calls"), list):
                        tool_calls_list = msg_dict_for_log["tool_calls"]
                        num_tc = len(tool_calls_list)
                        tc_names_sample = [
                            tc.get('function', {}).get('name', '?') 
                            for tc in tool_calls_list[:2] if isinstance(tc, dict)
                        ]
                        msg_dict_for_log["tool_calls"] = f"<{num_tc} tool_calls: {tc_names_sample}...>"
                    logged_msgs_sample_parts.append(f"Msg {len(api_params['messages']) - num_messages_to_log_sample + msg_param_idx}: {msg_dict_for_log}") 
                messages_log_str = "\n".join(logged_msgs_sample_parts)
            except Exception as log_e_inner: 
                raw_sample_str = [str(m)[:150] for m in api_params["messages"][-2:]] 
                messages_log_str = f"Error in detailed message logging: {log_e_inner}. Raw sample: {raw_sample_str}"

            custom_headers_for_api = {
                "X-User-Id": str(current_user.id),
                "X-Channel-Id": str(chat_id)
            }

            logger.info(
                f"API Call (iter {current_iteration}/{MAX_TOOL_ITERATIONS}, FreeWill={is_free_will_triggered}) to {api_params['model']} for chat {chat_id}. "
                f"Tool choice: {api_params.get('tool_choice', 'N/A')}. LLM History len: {len(llm_history)}. "
                f"Custom Headers: {custom_headers_for_api}. \n"
                f"API Messages (sample):\n{messages_log_str}"
            )

            response_from_ai = await aclient_shape.chat.completions.create(
                model=api_params["model"],
                messages=api_params["messages"], 
                tools=api_params.get("tools"), 
                tool_choice=api_params.get("tool_choice"), 
                extra_headers=custom_headers_for_api
            )
            ai_msg_obj = response_from_ai.choices[0].message
            llm_history.append(ai_msg_obj.model_dump(exclude_none=True))
            logger.debug(f"Chat {chat_id}: Appended assistant's response to LLM history. Last item: {str(llm_history[-1])[:250].replace(chr(10), '/N')}...")

            if ai_msg_obj.tool_calls:
                if not ENABLE_TOOL_USE: 
                    logger.warning(f"Chat {chat_id}: AI attempted tool use, but ENABLE_TOOL_USE is false. Tool calls: {ai_msg_obj.tool_calls}")
                    final_text_to_send_to_user = "I tried to use a special tool, but it's currently disabled. Please ask in a different way."
                    llm_history[-1] = {"role": "assistant", "content": final_text_to_send_to_user} 
                    break
                if is_free_will_triggered:
                    logger.warning(f"Chat {chat_id}: AI attempted tool use during free will, but tools are disabled for free will. Ignoring tool call.")
                    final_text_to_send_to_user = ai_msg_obj.content or "I had a thought but it involved a tool I can't use for spontaneous comments. Never mind!"
                    llm_history[-1] = {"role": "assistant", "content": final_text_to_send_to_user}
                    break

                if not tool_status_msg and chat_id:
                    tool_names_str = ", ".join(sorted(list(set(tc.function.name for tc in ai_msg_obj.tool_calls if tc.function and tc.function.name))))
                    status_text = f"🛠️ Activating tools: {tool_names_str}..."
                    try: tool_status_msg = await context.bot.send_message(chat_id, text=status_text)
                    except Exception as e_send_status: logger.warning(f"Chat {chat_id}: Failed to send tool status message: {e_send_status}")

                tool_results_for_history: list[ChatCompletionMessageParam] = []
                for tool_call in ai_msg_obj.tool_calls:
                    func_name, tool_call_id, args_str = tool_call.function.name, tool_call.id, tool_call.function.arguments
                    logger.info(f"Chat {chat_id}: AI requests tool: '{func_name}' (ID: {tool_call_id}) with raw args: {args_str}")
                    tool_content_result = f"Error: Tool '{func_name}' execution failed or tool is unknown."
                    if func_name in AVAILABLE_TOOLS_PYTHON_FUNCTIONS:
                        try:
                            py_func = AVAILABLE_TOOLS_PYTHON_FUNCTIONS[func_name]
                            parsed_args: Dict[str, Any] = {}
                            if args_str and args_str.strip(): 
                                parsed_args = json.loads(args_str)
                            else: 
                                logger.warning(f"Chat {chat_id}: Tool '{func_name}' called with empty/null arguments string. Raw: '{args_str}'. Proceeding with empty dict if function allows.")
                            if not isinstance(parsed_args, dict): 
                                raise TypeError(f"Parsed arguments for tool '{func_name}' are not a dictionary. Got {type(parsed_args)} from '{args_str}'")
                            
                            kwargs_for_tool = parsed_args.copy()
                            if func_name == "create_poll_in_chat": 
                                kwargs_for_tool["telegram_bot_context"] = context
                                kwargs_for_tool["current_chat_id"] = chat_id
                            # Add similar blocks if other tools need context in the future
                            
                            output = await py_func(**kwargs_for_tool) if asyncio.iscoroutinefunction(py_func) else await asyncio.to_thread(py_func, **kwargs_for_tool)
                            tool_content_result = str(output)
                            logger.info(f"Chat {chat_id}: Tool '{func_name}' executed. Output snippet: {tool_content_result[:200].replace(chr(10), ' ')}")
                        except (json.JSONDecodeError, TypeError, ValueError) as e_parse_args:
                            err_msg = f"Error parsing arguments or calling tool '{func_name}': {e_parse_args}. Raw args: '{args_str}'"
                            logger.error(f"Chat {chat_id}: {err_msg}", exc_info=True); tool_content_result = err_msg
                        except Exception as e_tool_exec: 
                            err_msg = f"Unexpected error executing tool '{func_name}': {e_tool_exec}"
                            logger.error(f"Chat {chat_id}: {err_msg}", exc_info=True); tool_content_result = err_msg
                    else: 
                        logger.warning(f"Chat {chat_id}: AI requested unknown tool: '{func_name}'")
                        tool_content_result = f"Error: Tool '{func_name}' is not available."
                    
                    tool_results_for_history.append({"tool_call_id": tool_call_id, "role": "tool", "name": func_name, "content": tool_content_result}) 
                
                llm_history.extend(tool_results_for_history)
                logger.debug(f"Chat {chat_id}: Extended LLM history with {len(tool_results_for_history)} tool results.")

            elif ai_msg_obj.content is not None: 
                final_text_to_send_to_user = str(ai_msg_obj.content)
                logger.info(f"Chat {chat_id}: AI final text response (iter {current_iteration}): '{final_text_to_send_to_user[:120].replace(chr(10), ' ')}...'")
                break 
            else: 
                logger.warning(f"Chat {chat_id}: AI response (iter {current_iteration}) had no tool_calls and content was None. Response: {ai_msg_obj.model_dump_json(indent=2)}")
                final_text_to_send_to_user = "AI provided an empty or unusual response. Please try rephrasing."
                llm_history[-1] = {"role": "assistant", "content": final_text_to_send_to_user} 
                break
        
        if current_iteration >= MAX_TOOL_ITERATIONS and not (ai_msg_obj and ai_msg_obj.content is not None):
            logger.warning(f"Chat {chat_id}: Max tool iterations ({MAX_TOOL_ITERATIONS}) reached without final content from AI.")
            final_text_to_send_to_user = "I tried using my tools multiple times but couldn't get a final answer. Could you try rephrasing your request or ask in a different way?"
            if not (llm_history and llm_history[-1].get("role") == "assistant" and llm_history[-1].get("content") == final_text_to_send_to_user): 
                 llm_history.append({"role": "assistant", "content": final_text_to_send_to_user}) 

        if tool_status_msg and chat_id:
            try: await tool_status_msg.delete()
            except Exception as e_del: logger.warning(f"Chat {chat_id}: Could not delete tool status msg ID {tool_status_msg.message_id}: {e_del}")

        if not final_text_to_send_to_user.strip():
            logger.warning(f"Chat {chat_id}: Final AI text to send was empty or whitespace. Defaulting to error message.")
            final_text_to_send_to_user = "I'm sorry, I couldn't generate a valid response for that. Please try again."
            if llm_history and llm_history[-1].get("role") == "assistant" and not llm_history[-1].get("content","").strip(): 
                llm_history[-1]["content"] = final_text_to_send_to_user

        logger.info(f"Chat {chat_id}: RAW AI response for user: >>>{final_text_to_send_to_user[:200].replace(chr(10), '/N')}...<<<")
        escaped_text_for_splitting = telegram_markdown_v2_escape(final_text_to_send_to_user)
        logger.info(f"Chat {chat_id}: FULLY ESCAPED text for user: >>>{escaped_text_for_splitting[:200].replace(chr(10), '/N')}...<<<")

        if typing_task and not typing_task.done(): typing_task.cancel(); typing_task = None
        max_mdv2_len, safe_truncate_len = 4096, 4096 - 150 

        if len(escaped_text_for_splitting) <= max_mdv2_len:
            try:
                await context.bot.send_message(chat_id, text=escaped_text_for_splitting, parse_mode=ParseMode.MARKDOWN_V2)
            except telegram.error.BadRequest as e_single_send: 
                logger.error(f"Chat {chat_id}: MDv2 send failed (len {len(escaped_text_for_splitting)}): {e_single_send}. Esc: '{escaped_text_for_splitting[:100].replace(chr(10), '/N')}...' Attempting plain text.")
                try:
                    plain_fb_full = final_text_to_send_to_user 
                    if len(plain_fb_full) <= max_mdv2_len:
                        await context.bot.send_message(chat_id, text=plain_fb_full)
                    else: 
                        logger.warning(f"Chat {chat_id}: Plain text fallback also too long ({len(plain_fb_full)}), splitting.")
                        num_chunks_plain = (len(plain_fb_full) + max_mdv2_len -1) // max_mdv2_len
                        for i, chunk in enumerate([plain_fb_full[j:j+max_mdv2_len] for j in range(0, len(plain_fb_full), max_mdv2_len)]):
                            hdr = f"[Fallback Part {i+1}/{ num_chunks_plain }]\n" if num_chunks_plain > 1 else ""
                            await context.bot.send_message(chat_id, text=hdr + chunk)
                            if i < num_chunks_plain -1: await asyncio.sleep(0.5)
                    logger.info(f"Chat {chat_id}: Sent as plain text fallback after MDv2 error.")
                except Exception as e_fb_send:
                    logger.error(f"Chat {chat_id}: Plain text fallback send also failed: {e_fb_send}");
                    await context.bot.send_message(chat_id, "Error formatting my response. (SNGF)")
        else: 
            logger.info(f"Chat {chat_id}: Escaped text too long ({len(escaped_text_for_splitting)}), using intelligent splitting.")
            message_parts_to_send = split_message_with_markdown_balancing(escaped_text_for_splitting, max_mdv2_len, logger)

            if not message_parts_to_send and escaped_text_for_splitting.strip(): 
                 logger.warning(f"Chat {chat_id}: Splitter returned no parts for non-empty text. Sending truncated. Text: {escaped_text_for_splitting[:100].replace(chr(10), '/N')}")
                 trunc_txt = escaped_text_for_splitting[:safe_truncate_len] + "\n\\[MESSAGE_TRUNCATED_SPLIT_FAIL\\]"
                 message_parts_to_send.append(trunc_txt[:max_mdv2_len]) 

            for i, part_chunk in enumerate(message_parts_to_send):
                current_chunk_to_send = part_chunk.strip()
                if not current_chunk_to_send: 
                    temp_strip_styles = part_chunk
                    for delim_s in _BALANCING_MARKDOWN_DELIMITERS: temp_strip_styles = temp_strip_styles.replace(delim_s, "")
                    if not temp_strip_styles.strip(): 
                        logger.info(f"Chat {chat_id}: Skipping empty or style-only part {i+1} from splitter."); continue
                    current_chunk_to_send = part_chunk 

                if len(current_chunk_to_send) > max_mdv2_len: 
                    logger.warning(f"Chat {chat_id}: Split part {i+1} still too long ({len(current_chunk_to_send)}). Truncating.")
                    current_chunk_to_send = current_chunk_to_send[:safe_truncate_len] + "\n\\[MESSAGE_PART_TRUNCATED\\]"
                    current_chunk_to_send = current_chunk_to_send[:max_mdv2_len] 

                logger.info(f"Chat {chat_id}: Sending MDv2 part {i+1}/{len(message_parts_to_send)} (len: {len(current_chunk_to_send)}): '{current_chunk_to_send[:70].replace(chr(10),'/N')}...'")
                try:
                    await context.bot.send_message(chat_id, text=current_chunk_to_send, parse_mode=ParseMode.MARKDOWN_V2)
                    if i < len(message_parts_to_send) - 1: await asyncio.sleep(0.75) 
                except telegram.error.BadRequest as e_split_part_send:
                    logger.error(f"Chat {chat_id}: MDv2 send failed for SPLIT part {i+1} (len {len(current_chunk_to_send)}): {e_split_part_send}. Chunk: '{current_chunk_to_send[:100].replace(chr(10),'/N')}...' Attempting plain text for this part.")
                    fb_hdr = f"[Part {i+1}/{len(message_parts_to_send)} (escaped content, shown plain due to formatting error)]:\n"
                    plain_chunk_fb = fb_hdr + current_chunk_to_send 

                    if len(plain_chunk_fb) > max_mdv2_len:
                        avail_len_for_content = max_mdv2_len - len(fb_hdr) - len("\n[...TRUNCATED_PLAIN_PART]")
                        plain_chunk_fb = fb_hdr + current_chunk_to_send[:max(0, avail_len_for_content)] + "\n[...TRUNCATED_PLAIN_PART]"
                    try:
                        await context.bot.send_message(chat_id, text=plain_chunk_fb[:max_mdv2_len])
                    except Exception as e_fb_split_send:
                        logger.error(f"Chat {chat_id}: Plain fallback for split part {i+1} also failed: {e_fb_split_send}")
                        await context.bot.send_message(chat_id, f"[Problem sending part {i+1} of my response. (FPF2)]")
                except Exception as e_gen_split_send: 
                    logger.error(f"Chat {chat_id}: General error sending split part {i+1}: {e_gen_split_send}", exc_info=True)
                    await context.bot.send_message(chat_id, f"[A problem occurred sending part {i+1} of my response. (SPF)]")

    except openai.InternalServerError as e_openai_ise: 
        if hasattr(e_openai_ise, 'response') and e_openai_ise.response and e_openai_ise.response.status_code == 504:
            logger.error(f"Chat {chat_id}: Received 504 Gateway Timeout from Shapes API.", exc_info=True) 
            final_text_to_send_to_user = "The AI is taking a bit too long to generate a response and timed out. You could try asking for something more concise, or try again in a moment."
            if not (llm_history and llm_history[-1].get("role") == "assistant" and llm_history[-1].get("content") == final_text_to_send_to_user):
                 llm_history.append({"role": "assistant", "content": final_text_to_send_to_user})
            if chat_id: await context.bot.send_message(chat_id, final_text_to_send_to_user)
        else: 
            logger.error(f"Chat {chat_id}: OpenAI InternalServerError: {e_openai_ise}", exc_info=True)
            # Add a generic message to history and send to user
            final_text_to_send_to_user = "Sorry, the AI service encountered an internal error. Please try again later."
            if not (llm_history and llm_history[-1].get("role") == "assistant" and llm_history[-1].get("content") == final_text_to_send_to_user):
                 llm_history.append({"role": "assistant", "content": final_text_to_send_to_user})
            if chat_id: await context.bot.send_message(chat_id, final_text_to_send_to_user)
            # We don't re-raise here as we've handled it by informing the user.
    except openai.APITimeoutError as e_openai_timeout:
        logger.error(f"Chat {chat_id}: OpenAI APITimeoutError. Client-side timeout for Shapes API.", exc_info=True)
        final_text_to_send_to_user = "The AI is taking too long to respond and the request timed out on my side. Please try asking for something shorter or try again later."
        if not (llm_history and llm_history[-1].get("role") == "assistant" and llm_history[-1].get("content") == final_text_to_send_to_user):
             llm_history.append({"role": "assistant", "content": final_text_to_send_to_user})
        if chat_id: await context.bot.send_message(chat_id, final_text_to_send_to_user) 
    except telegram.error.BadRequest as e_outer_tg_badreq: # This might catch errors from send_message if text is malformed beyond MDV2
        logger.error(f"Outer Telegram BadRequest for chat {chat_id}: {e_outer_tg_badreq}. Raw AI response: '{final_text_to_send_to_user[:100]}'", exc_info=True)
        try: 
            plain_fb_full_outer = final_text_to_send_to_user # Attempt to send the original AI response as plain text
            if len(plain_fb_full_outer) <= max_mdv2_len: # Telegram's absolute max length
                await context.bot.send_message(chat_id, text=plain_fb_full_outer)
            else: # If even plain text is too long, split it
                logger.warning(f"Chat {chat_id}: Outer plain text fallback also too long ({len(plain_fb_full_outer)}), splitting.")
                num_chunks_outer = (len(plain_fb_full_outer) + max_mdv2_len -1) // max_mdv2_len
                for i, chunk in enumerate([plain_fb_full_outer[j:j+max_mdv2_len] for j in range(0, len(plain_fb_full_outer), max_mdv2_len)]):
                    hdr = f"[Fallback Part {i+1}/{num_chunks_outer}]\n" if num_chunks_outer > 1 else ""
                    await context.bot.send_message(chat_id, text=hdr + chunk)
                    if i < num_chunks_outer -1 : await asyncio.sleep(0.5)
            logger.info(f"Chat {chat_id}: Sent entire message as plain text (outer error fallback).")
        except Exception as e_fb_outer_send: # If sending plain text also fails
            logger.error(f"Chat {chat_id}: Outer plain text fallback send also failed: {e_fb_outer_send}")
            await context.bot.send_message(chat_id, "A general error occurred while formatting my response. (OBRF)")
    except (httpx.NetworkError, httpx.TimeoutException, httpx.ConnectError, telegram.error.NetworkError, telegram.error.TimedOut) as e_net_comm:
        logger.error(f"Network-related error for chat {chat_id}: {e_net_comm}", exc_info=False) # exc_info=False as traceback might be less useful
        if chat_id: # Ensure chat_id is available
            try:
                await context.bot.send_message(chat_id, "⚠️ I'm having some network issues. Please try again in a little while.")
            except Exception as e_send_net_err_msg: # If sending the notification itself fails
                 logger.error(f"Chat {chat_id}: Failed to send network error notification: {e_send_net_err_msg}")
    except Exception as e_main_handler: # Catch-all for truly unexpected errors in this handler
        logger.error(f"General, unhandled error in process_message_entrypoint for chat {chat_id}: {e_main_handler}", exc_info=True)
        if chat_id: # Ensure chat_id is available
            try:
                await context.bot.send_message(chat_id, "😵‍💫 Oops! Something went wrong. I've noted it. Please try again. (MGEN)")
            except Exception as e_send_gen_err_msg: # If sending the notification itself fails
                logger.error(f"Chat {chat_id}: Failed to send general error notification: {e_send_gen_err_msg}")
    finally:
        if typing_task and not typing_task.done():
            typing_task.cancel()
            logger.debug(f"Chat {chat_id}: Typing task cancelled in finally block.")
# --- END OF Main Message Handler ---

# --- ERROR HANDLER ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = "".join(tb_list)
    update_str = "Update data not available or not an Update instance."
    effective_chat_id_for_error = "N/A" # Default
    if isinstance(update, Update):
        try: update_str = json.dumps(update.to_dict(), indent=2, ensure_ascii=False, default=str)
        except Exception: update_str = str(update) # Fallback if to_dict fails
        if update.effective_chat:
            effective_chat_id_for_error = str(update.effective_chat.id)
    elif update: # If update is not None but not an Update instance
        update_str = str(update)

    chat_data_str = str(context.chat_data)[:500] if context.chat_data else "N/A"
    user_data_str = str(context.user_data)[:500] if context.user_data else "N/A"

    send_message_plain = (
        f"Bot Exception in chat {effective_chat_id_for_error}:\n"
        f"Update: {update_str[:1500]}...\n\n"
        f"Chat Data: {chat_data_str}\nUser Data: {user_data_str}\n\n"
        f"Traceback (last 1500 chars):\n{tb_string[-1500:]}"
    )

    chat_id_to_notify_admin: Optional[int] = None
    if ALLOWED_USERS: # Check if ALLOWED_USERS is populated
        try: chat_id_to_notify_admin = int(ALLOWED_USERS[0]) # Attempt to get the first admin user ID
        except (ValueError, IndexError, TypeError): logger.error("No valid admin user ID for error notification from ALLOWED_USERS[0].")

    # Attempt to notify the user in the chat where the error occurred, if appropriate
    user_notified_by_handler = False
    if isinstance(update, Update) and update.effective_chat:
        # Avoid sending generic error if it's a network/API timeout, as those are handled in process_message_entrypoint
        if not isinstance(context.error, (openai.InternalServerError, openai.APITimeoutError, telegram.error.NetworkError, httpx.NetworkError)):
            try:
                user_error_message = f"<b>Bot Error:</b> <pre>{html.escape(str(context.error))}</pre>\n<i>An unexpected error occurred. The admin has been notified if configured.</i>"
                if len(user_error_message) <= 4096: # Telegram message length limit
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=user_error_message, parse_mode=ParseMode.HTML)
                    user_notified_by_handler = True
            except Exception as e_send_user_err:
                logger.error(f"Failed to send user-friendly error to chat {update.effective_chat.id}: {e_send_user_err}")

    # Send detailed error to admin if configured
    if chat_id_to_notify_admin:
        # Avoid duplicate notification if admin is the one who experienced the error and was already notified
        if user_notified_by_handler and update.effective_chat and update.effective_chat.id == chat_id_to_notify_admin:
            logger.info(f"Admin was the user in chat {chat_id_to_notify_admin} and already notified about the error. Skipping redundant admin report.")
        else:
            max_len = 4096 # Telegram message length limit
            try:
                # Prefer HTML for admin if error is short and not potentially full of HTML itself
                is_potentially_html_error = isinstance(context.error, (openai.InternalServerError, httpx.HTTPStatusError)) # These might contain HTML in response
                
                if not is_potentially_html_error and len(send_message_plain) < max_len - 200 : # If plain text is short enough for HTML wrapper
                    short_html_err = f"<b>Bot Error in chat {effective_chat_id_for_error}:</b>\n<pre>{html.escape(str(context.error))}</pre>\n<i>(Full details in server logs. Update/TB follows if space.)</i>"
                    if len(short_html_err) <=max_len : # Check if HTML version is within limits
                         await context.bot.send_message(chat_id=chat_id_to_notify_admin, text=short_html_err, parse_mode=ParseMode.HTML)
                    else: # HTML version too long, revert to plain
                        is_potentially_html_error = True # Force plain text path

                if is_potentially_html_error or len(send_message_plain) >= max_len -200 : # Send as plain text if it's long or potentially HTML
                    if len(send_message_plain) <= max_len:
                        await context.bot.send_message(chat_id=chat_id_to_notify_admin, text=send_message_plain)
                    else: # Split long plain text message for admin
                        num_err_chunks = (len(send_message_plain) + max_len -1) // max_len
                        for i_err, chunk in enumerate([send_message_plain[j:j+max_len] for j in range(0, len(send_message_plain), max_len)]):
                            hdr = f"[BOT ERR Pt {i_err+1}/{num_err_chunks}]\n" if num_err_chunks > 1 else "[BOT ERR]\n"
                            await context.bot.send_message(chat_id=chat_id_to_notify_admin, text=(hdr + chunk)[:max_len]) # Ensure chunk with header doesn't exceed max_len
                            if i_err < num_err_chunks -1 : await asyncio.sleep(0.5) # Brief pause between chunks
            except Exception as e_send_err: # Fallback if sending detailed error fails
                logger.error(f"Failed sending detailed error report to admin {chat_id_to_notify_admin}: {e_send_err}")
                try: await context.bot.send_message(chat_id=chat_id_to_notify_admin, text=f"Bot Error: {str(context.error)[:1000]}\n(Details in server logs. Report sending failed.)")
                except Exception as e_final_fb: logger.error(f"Final admin error report fallback failed: {e_final_fb}")
    elif not user_notified_by_handler: # If no admin to notify and user wasn't notified by specific error message
        logger.error("No chat ID found to send error message via Telegram (admin not set, or user already got specific error from main handler). Error details logged to server.")
# --- END OF ERROR HANDLER ---

if __name__ == "__main__":
    app_builder = ApplicationBuilder().token(TELEGRAM_TOKEN)
    app = app_builder.build()

    # Command Handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("newchat", new_chat_command))
    if BING_IMAGE_CREATOR_AVAILABLE: # Only add imagine/setbingcookie if library is available
        if BING_AUTH_COOKIE: # Only add /imagine if cookie is also set
            app.add_handler(CommandHandler("imagine", imagine_command))
        app.add_handler(CommandHandler("setbingcookie", set_bing_cookie_command)) # Admin can set cookie even if not initially present

    # Message Handler for text, photos, voice, and replies (not commands)
    app.add_handler(MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.VOICE | filters.REPLY) & (~filters.COMMAND),
        process_message_entrypoint
    ))

    # Error Handler
    app.error_handler = error_handler

    logger.info("Bot is starting to poll for updates...")
    try:
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    except telegram.error.NetworkError as ne: # Specific catch for initial network errors during startup
        logger.critical(f"CRITICAL: Initial NetworkError during polling setup: {ne}. Check network/token.", exc_info=True)
    except Exception as main_e: # Catch any other critical errors during startup/polling loop
        logger.critical(f"CRITICAL: Unhandled exception at main polling level: {main_e}", exc_info=True)