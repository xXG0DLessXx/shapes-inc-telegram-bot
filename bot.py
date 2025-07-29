import logging
import os
import json
import random
import shutil
import base64
import mimetypes
import asyncio
import re  # For sanitization and new markdown, and calculator tool
# telegramify_markdown for advanced message formatting
import telegramify_markdown
from telegramify_markdown.type import ContentTypes
from telegramify_markdown.interpreters import (
    TextInterpreter, FileInterpreter, MermaidInterpreter, InterpreterChain
)
import traceback  # For detailed error handler
import html  # For detailed error handler
import httpx  # For specific exception types
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Set, Union, Any

# New imports for additional tools
import math
from datetime import (
    datetime,
    timedelta,
    timezone as dt_timezone,
)  # Added timezone for startup timestamp
import pytz
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote as url_unquote

# New imports for document handling
import docx  # Add this line if missing
from odf import text, teletype  # Add this line if missing
from odf.opendocument import load as odf_load  # Add this line if missing
from io import BytesIO  # Add this line if missing

# Telegram imports
from telegram import (
    Update,
    InputMediaPhoto,
    Message as TelegramMessage,
    User as TelegramUser,
    Chat,
    Voice,
    Bot,
    BotCommand,
    ChatPermissions,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
    ContextTypes,
    Application,
)
from telegram.helpers import escape_markdown
from telegram.constants import ChatAction, ParseMode
import telegram.error

# OpenAI imports
from openai import (
    AsyncOpenAI,
    InternalServerError,
    APITimeoutError,
)  # Added InternalServerError, APITimeoutError
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_message import ChatCompletionMessage

import sqlite3
# --- DatabaseManager class ---
class DatabaseManager:
    """Handles all database operations for the bot."""
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Creates the necessary database tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_tokens (
                user_id INTEGER PRIMARY KEY,
                auth_token TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_histories (
                chat_id INTEGER NOT NULL,
                thread_id INTEGER,
                history_json TEXT NOT NULL,
                last_updated TIMESTAMP NOT NULL,
                PRIMARY KEY (chat_id, thread_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activated_topics (
                chat_id INTEGER NOT NULL,
                thread_id INTEGER,
                PRIMARY KEY (chat_id, thread_id)
            )
        """)
        self.conn.commit()
        logger.info("Database setup complete. All tables are ready.")

    # --- Token Management ---
    def save_user_token(self, user_id: int, token: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO user_tokens (user_id, auth_token) VALUES (?, ?)",
            (user_id, token)
        )
        self.conn.commit()

    def load_all_user_tokens(self) -> Dict[int, str]:
        cursor = self.conn.execute("SELECT user_id, auth_token FROM user_tokens")
        return {row['user_id']: row['auth_token'] for row in cursor.fetchall()}

    # --- History Management ---
    def save_history(self, chat_id: int, thread_id: Optional[int], history: list):
        history_json = json.dumps(history)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO conversation_histories (chat_id, thread_id, history_json, last_updated)
            VALUES (?, ?, ?, ?)
            """,
            (chat_id, thread_id, history_json, datetime.now())
        )
        self.conn.commit()

    def get_history(self, chat_id: int, thread_id: Optional[int]) -> list:
        cursor = self.conn.execute(
            "SELECT history_json FROM conversation_histories WHERE chat_id = ? AND thread_id IS ?",
            (chat_id, thread_id)
        )
        row = cursor.fetchone()
        return json.loads(row['history_json']) if row else []

    def delete_history(self, chat_id: int, thread_id: Optional[int]):
        self.conn.execute(
            "DELETE FROM conversation_histories WHERE chat_id = ? AND thread_id IS ?",
            (chat_id, thread_id)
        )
        self.conn.commit()
        
    def load_all_histories(self) -> dict:
        cursor = self.conn.execute("SELECT chat_id, thread_id, history_json FROM conversation_histories")
        histories = {}
        for row in cursor.fetchall():
            key = (row['chat_id'], row['thread_id'])
            histories[key] = json.loads(row['history_json'])
        return histories
        
    # --- Activation Management ---
    def activate_topic(self, chat_id: int, thread_id: Optional[int]):
        self.conn.execute(
            "INSERT OR IGNORE INTO activated_topics (chat_id, thread_id) VALUES (?, ?)",
            (chat_id, thread_id)
        )
        self.conn.commit()
        
    def deactivate_topic(self, chat_id: int, thread_id: Optional[int]):
        self.conn.execute(
            "DELETE FROM activated_topics WHERE chat_id = ? AND thread_id IS ?",
            (chat_id, thread_id)
        )
        self.conn.commit()

    def load_all_activated_topics(self) -> Set[Tuple[int, Optional[int]]]:
        cursor = self.conn.execute("SELECT chat_id, thread_id FROM activated_topics")
        return {(row['chat_id'], row['thread_id']) for row in cursor.fetchall()}

    def close(self):
        """Closes the database connection."""
        self.conn.close()
# --- End of DatabaseManager class ---

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
SHAPESINC_APP_ID = os.getenv("SHAPESINC_APP_ID")
SHAPESINC_SHAPE_USERNAME = os.getenv("SHAPESINC_SHAPE_USERNAME")
ALLOWED_USERS_STR = os.getenv("ALLOWED_USERS", "")
ALLOWED_CHATS_STR = os.getenv("ALLOWED_CHATS", "")
BOT_OWNERS_STR = os.getenv("BOT_OWNERS", "")
NOTIFY_OWNER_ON_ERROR = os.getenv("NOTIFY_OWNER_ON_ERROR", "true").lower() == "true"

BING_AUTH_COOKIE = (
    os.getenv("BING_AUTH_COOKIE") if BING_IMAGE_CREATOR_AVAILABLE else None
)
ENABLE_TOOL_USE = os.getenv("ENABLE_TOOL_USE", "false").lower() == "true"
SHAPES_API_BASE_URL = os.getenv("SHAPES_API_BASE_URL", "https://api.shapes.inc/v1/")
SHAPES_AUTH_BASE_URL = os.getenv("SHAPES_AUTH_BASE_URL", "https://api.shapes.inc/auth")

SEPARATE_TOPIC_HISTORIES = (
    os.getenv("SEPARATE_TOPIC_HISTORIES", "false").lower() == "true"
)

GROUP_FREE_WILL_ENABLED = (
    os.getenv("GROUP_FREE_WILL_ENABLED", "false").lower() == "true"
)
GROUP_FREE_WILL_PROBABILITY_STR = os.getenv("GROUP_FREE_WILL_PROBABILITY", "0.0")
GROUP_FREE_WILL_CONTEXT_MESSAGES_STR = os.getenv(
    "GROUP_FREE_WILL_CONTEXT_MESSAGES", "3"
)

IGNORE_OLD_MESSAGES_ON_STARTUP = (
    os.getenv("IGNORE_OLD_MESSAGES_ON_STARTUP", "false").lower() == "true"
)
BOT_STARTUP_TIMESTAMP: Optional[datetime] = None  # Will be set in main

if not SHAPESINC_APP_ID:
    logger.warning("SHAPESINC_APP_ID not set in .env. Auth command will not work correctly.")

# This dictionary is now just a cache, loaded from the DB at startup
user_auth_tokens: Dict[int, str] = {}

# States for ConversationHandler
AWAITING_CODE = 1

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
    if not (0 <= GROUP_FREE_WILL_CONTEXT_MESSAGES <= 20):  # Max 20 for sanity
        raise ValueError("GROUP_FREE_WILL_CONTEXT_MESSAGES must be between 0 and 20.")
except ValueError as e:
    logging.warning(f"Invalid GROUP_FREE_WILL_CONTEXT_MESSAGES: {e}. Defaulting to 3.")
    GROUP_FREE_WILL_CONTEXT_MESSAGES = 3

SHAPES_API_CLIENT_TIMEOUT = httpx.Timeout(90.0, connect=5.0, read=85.0)
HTTP_CLIENT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

if not TELEGRAM_TOKEN:
    raise ValueError("BOT_TOKEN not set in environment.")
if not SHAPESINC_API_KEY:
    raise ValueError("SHAPESINC_API_KEY not set in environment.")
if not SHAPESINC_SHAPE_USERNAME:
    raise ValueError("SHAPESINC_SHAPE_USERNAME not set in environment.")

ALLOWED_USERS = [user.strip() for user in ALLOWED_USERS_STR.split(",") if user.strip()]
ALLOWED_CHATS = [chat.strip() for chat in ALLOWED_CHATS_STR.split(",") if chat.strip()]
BOT_OWNERS = [owner.strip() for owner in BOT_OWNERS_STR.split(",") if owner.strip()]

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

try:
    aclient_shape = AsyncOpenAI(
        api_key=SHAPESINC_API_KEY,
        base_url=SHAPES_API_BASE_URL,
        timeout=SHAPES_API_CLIENT_TIMEOUT,
    )
    logger.info(
        f"Shapes client init: {SHAPESINC_SHAPE_USERNAME} at {SHAPES_API_BASE_URL} with timeout {SHAPES_API_CLIENT_TIMEOUT}"
    )
    logger.info(f"Tool use: {'ENABLED' if ENABLE_TOOL_USE else 'DISABLED'}.")
    if GROUP_FREE_WILL_ENABLED:
        logger.info(
            f"Group Free Will: ENABLED with probability {GROUP_FREE_WILL_PROBABILITY:.2%} and context of {GROUP_FREE_WILL_CONTEXT_MESSAGES} messages."
        )
    else:
        logger.info("Group Free Will: DISABLED.")

    logger.info(
        f"Separate Topic Histories: {'ENABLED' if SEPARATE_TOPIC_HISTORIES else 'DISABLED'}"
    )
    logger.info(
        f"Ignore old messages on startup: {'ENABLED' if IGNORE_OLD_MESSAGES_ON_STARTUP else 'DISABLED'}"
    )
except Exception as e:
    logger.error(f"Failed to init Shapes client: {e}")
    raise

# Stores conversation history per chat_id and thread_id tuple
chat_histories: dict[tuple[int, Optional[int]], list[ChatCompletionMessageParam]] = {}
# Stores asyncio.Lock to serialize processing within a single conversational context.
# The key is either a chat_id (int) or a (chat_id, thread_id) tuple.
processing_locks: Dict[Union[int, Tuple[int, Optional[int]]], asyncio.Lock] = {}
MAX_TEXT_FILE_SIZE = 500 * 1024  # 500 KB limit for text files
MAX_HISTORY_LENGTH = 10
# Stores raw messages (sender/text) per group chat_id and thread_id for free will context
group_raw_message_log: Dict[int, Dict[Optional[int], List[Dict[str, str]]]] = {}
MAX_RAW_LOG_PER_THREAD = 50  # Limit raw log size per topic/thread
# Stores chat_id/thread_id tuples where the bot should always be active
activated_chats_topics: Set[Tuple[int, Optional[int]]] = set()
# --- Topic Name Cache ---
topic_names_cache: Dict[Tuple[int, int], str] = {}  # (chat_id, thread_id) -> topic_name
MAX_TOPIC_ENTRIES_PER_CHAT_IN_CACHE = 200
chat_topic_cache_keys_order: Dict[int, List[Tuple[int, int]]] = {}
# --- END OF Global Config & Setup ---

# --- HELPER FUNCTION for sending messages with thread fallback ---
async def send_message_to_chat_or_general(
    bot_instance: Bot,
    chat_id: int,
    text: str,
    preferred_thread_id: Optional[int],
    parse_mode: Optional[str] = None,
    **kwargs,  # For other send_message params like reply_markup, disable_web_page_preview
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
            **kwargs,
        )
    except telegram.error.BadRequest as e:
        if (
            "message thread not found" in e.message.lower()
            and preferred_thread_id is not None
        ):
            logger.warning(
                f"Chat {chat_id}: Preferred thread {preferred_thread_id} not found for message. "
                f"Attempting to send to general chat instead. Error: {e}"
            )
            try:
                # Fallback: send to general chat (message_thread_id=None)
                return await bot_instance.send_message(
                    chat_id=chat_id,
                    text=text,
                    message_thread_id=None,  # Explicitly None for general chat
                    parse_mode=parse_mode,
                    **kwargs,
                )
            except telegram.error.BadRequest as e2:
                # If sending to general also fails with BadRequest (could be for other reasons now)
                logger.error(
                    f"Chat {chat_id}: Sending to general chat also failed with BadRequest. Error: {e2}"
                )
                raise e2  # Re-raise the second error
            except Exception as e_general_unexpected:
                logger.error(
                    f"Chat {chat_id}: Unexpected error sending to general chat. Error: {e_general_unexpected}"
                )
                raise e_general_unexpected  # Re-raise
        else:
            # Different BadRequest error, or preferred_thread_id was already None
            raise e  # Re-raise original error
    except Exception as e_other_send:
        # Catch any other non-BadRequest errors during send
        logger.error(
            f"Chat {chat_id} (thread {preferred_thread_id}): Unexpected error during send_message. Error: {e_other_send}"
        )
        raise e_other_send
# --- END OF HELPER FUNCTION ---

# Add helper functions for media with fallback logic
async def send_photo_to_chat_or_general(
    bot_instance: Bot,
    chat_id: int,
    photo: Union[str, bytes],
    preferred_thread_id: Optional[int],
    **kwargs,
) -> Optional[TelegramMessage]:
    try:
        return await bot_instance.send_photo(
            chat_id=chat_id,
            photo=photo,
            message_thread_id=preferred_thread_id,
            **kwargs,
        )
    except telegram.error.BadRequest as e:
        if (
            "message thread not found" in e.message.lower()
            and preferred_thread_id is not None
        ):
            logger.warning(
                f"Chat {chat_id}: Preferred thread {preferred_thread_id} not found for photo. "
                f"Attempting to send to general chat instead. Error: {e}"
            )
            try:
                # Fallback: send to general chat (message_thread_id=None)
                return await bot_instance.send_photo(
                    chat_id=chat_id, photo=photo, message_thread_id=None, **kwargs
                )
            except Exception as e2:
                logger.error(
                    f"Chat {chat_id}: Sending photo to general chat also failed. Error: {e2}"
                )
                raise e2
        else:
            raise e
    except Exception as e_other:
        logger.error(
            f"Chat {chat_id} (thread {preferred_thread_id}): Unexpected error during send_photo. Error: {e_other}"
        )
        raise e_other

async def send_audio_to_chat_or_general(
    bot_instance: Bot,
    chat_id: int,
    audio: Union[str, bytes],
    preferred_thread_id: Optional[int],
    **kwargs,
) -> Optional[TelegramMessage]:
    try:
        return await bot_instance.send_audio(
            chat_id=chat_id,
            audio=audio,
            message_thread_id=preferred_thread_id,
            **kwargs,
        )
    except telegram.error.BadRequest as e:
        if (
            "message thread not found" in e.message.lower()
            and preferred_thread_id is not None
        ):
            logger.warning(
                f"Chat {chat_id}: Preferred thread {preferred_thread_id} not found for audio. "
                f"Attempting to send to general chat instead. Error: {e}"
            )
            try:
                # Fallback: send to general chat (message_thread_id=None)
                return await bot_instance.send_audio(
                    chat_id=chat_id, audio=audio, message_thread_id=None, **kwargs
                )
            except Exception as e2:
                logger.error(
                    f"Chat {chat_id}: Sending audio to general chat also failed. Error: {e2}"
                )
                raise e2
        else:
            raise e
    except Exception as e_other:
        logger.error(
            f"Chat {chat_id} (thread {preferred_thread_id}): Unexpected error during send_audio. Error: {e_other}"
        )
        raise e_other

async def send_document_to_chat_or_general(
    bot_instance: Bot,
    chat_id: int,
    document: Union[str, bytes, Tuple[str, bytes]],
    preferred_thread_id: Optional[int],
    **kwargs,
) -> Optional[TelegramMessage]:
    try:
        return await bot_instance.send_document(
            chat_id=chat_id,
            document=document,
            message_thread_id=preferred_thread_id,
            **kwargs,
        )
    except telegram.error.BadRequest as e:
        if (
            "message thread not found" in e.message.lower()
            and preferred_thread_id is not None
        ):
            logger.warning(
                f"Chat {chat_id}: Preferred thread {preferred_thread_id} not found for document. "
                f"Attempting to send to general chat instead. Error: {e}"
            )
            try:
                # Fallback: send to general chat (message_thread_id=None)
                return await bot_instance.send_document(
                    chat_id=chat_id, document=document, message_thread_id=None, **kwargs
                )
            except Exception as e2:
                logger.error(
                    f"Chat {chat_id}: Sending document to general chat also failed. Error: {e2}"
                )
                raise e2
        else:
            raise e
    except Exception as e_other:
        logger.error(
            f"Chat {chat_id} (thread {preferred_thread_id}): Unexpected error during send_document. Error: {e_other}"
        )
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
    current_message_thread_id: Optional[int] = None,  # Added thread ID
) -> str:
    logger.info(
        f"TOOL: create_poll_tool for chat_id {current_chat_id} (thread_id: {current_message_thread_id}) with question='{question}', options={options}"
    )

    if not telegram_bot_context or not current_chat_id:
        err_msg = "Telegram context or chat ID not provided to create_poll_tool."
        logger.error(err_msg)
        return json.dumps(
            {
                "error": err_msg,
                "details": "This tool requires internal context to function.",
            }
        )

    # Validation
    if not question or not isinstance(question, str) or len(question.strip()) == 0:
        return json.dumps({"error": "Poll question cannot be empty."})
    if (
        not options
        or not isinstance(options, list)
        or len(options) < 2
        or len(options) > 10
    ):
        return json.dumps({"error": "Poll must have between 2 and 10 options."})
    if not all(isinstance(opt, str) and opt.strip() for opt in options):
        return json.dumps({"error": "All poll options must be non-empty strings."})

    try:
        # Ensure all options are unique, Telegram might enforce this
        unique_options = list(
            dict.fromkeys(options)
        )  # Preserves order while making unique
        if len(unique_options) < len(options):
            logger.warning(
                f"Poll options for '{question}' had duplicates, using unique set: {unique_options}"
            )

        # Telegram API limits for question (1-300 chars) and options (1-100 chars)
        if len(question) > 300:
            return json.dumps(
                {"error": "Poll question is too long (max 300 characters)."}
            )
        for opt_idx, opt_val in enumerate(unique_options):
            if len(opt_val) > 100:
                return json.dumps(
                    {
                        "error": f"Poll option #{opt_idx+1} ('{opt_val[:20]}...') is too long (max 100 characters)."
                    }
                )

        # Attempt to send poll to the specific thread
        await telegram_bot_context.bot.send_poll(
            chat_id=current_chat_id,
            question=question,
            options=unique_options,
            is_anonymous=is_anonymous,
            allows_multiple_answers=allows_multiple_answers,
            message_thread_id=current_message_thread_id,  # Pass the thread ID
        )
        logger.info(
            f"Poll sent to chat {current_chat_id} (thread {current_message_thread_id}): '{question}'"
        )
        return json.dumps(
            {
                "status": "poll_created_successfully",
                "question": question,
                "options_sent": unique_options,
                "chat_id": current_chat_id,
                "message_thread_id": current_message_thread_id,
            }
        )
    except telegram.error.BadRequest as e:
        # Check if this BadRequest is due to thread not found for polls
        if (
            "message thread not found" in e.message.lower()
            and current_message_thread_id is not None
        ):
            logger.warning(
                f"Poll for chat {current_chat_id}, thread {current_message_thread_id} failed (thread not found). Retrying in general chat."
            )
            try:
                # Fallback: Send poll to general chat
                await telegram_bot_context.bot.send_poll(
                    chat_id=current_chat_id,
                    question=question,
                    options=unique_options,
                    is_anonymous=is_anonymous,
                    allows_multiple_answers=allows_multiple_answers,
                    message_thread_id=None,  # Fallback to general
                )
                logger.info(
                    f"Poll sent to chat {current_chat_id} (GENERAL after thread fail): '{question}'"
                )
                return json.dumps(
                    {
                        "status": "poll_created_successfully_in_general_after_thread_fail",
                        "question": question,
                        "options_sent": unique_options,
                        "chat_id": current_chat_id,
                        "message_thread_id": None,
                    }
                )
            except Exception as e2:
                logger.error(
                    f"Error creating poll in general chat (fallback) for {current_chat_id}: {e2}",
                    exc_info=True,
                )
                return json.dumps(
                    {
                        "error": "Failed to create poll in thread and general.",
                        "details": f"Telegram API error: {e2.message if hasattr(e2, 'message') else str(e2)}",
                    }
                )
        else:
            # Different BadRequest error
            logger.error(
                f"Telegram BadRequest creating poll in chat {current_chat_id} (thread {current_message_thread_id}): {e}",
                exc_info=True,
            )
            return json.dumps(
                {
                    "error": "Failed to create poll.",
                    "details": f"Telegram API error: {e.message}",
                }
            )
    except Exception as e:
        logger.error(
            f"Unexpected error in create_poll_tool for chat {current_chat_id} (thread {current_message_thread_id}): {e}",
            exc_info=True,
        )
        return json.dumps(
            {
                "error": "An unexpected error occurred while trying to create the poll.",
                "details": str(e),
            }
        )


async def calculator_tool(expression: str) -> str:
    logger.info(f"TOOL: calculator_tool with expression: '{expression}'")
    processed_expression = expression.replace("^", "**")
    if re.search(r"[a-zA-Z]", processed_expression):
        logger.warning(
            f"Calculator: Forbidden characters (letters) in expression: '{processed_expression}'"
        )
        return json.dumps({"error": "Invalid expression: letters are not allowed."})
    temp_expr = processed_expression
    temp_expr = re.sub(
        r"[0-9\.\s]", "", temp_expr
    )  # Keep dots for floats, remove numbers and whitespace
    temp_expr = (
        temp_expr.replace("**", "")
        .replace("*", "")
        .replace("/", "")
        .replace("+", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
    )
    if temp_expr:
        logger.warning(
            f"Calculator: Invalid characters remain: '{temp_expr}' from '{processed_expression}'"
        )
        return json.dumps(
            {"error": "Invalid expression: contains disallowed characters."}
        )
    try:
        result = eval(processed_expression, {"__builtins__": {}}, {})
        if not isinstance(result, (int, float)):
            return json.dumps(
                {"error": "Expression did not evaluate to a numerical result."}
            )
        if isinstance(result, float):
            result_str = f"{result:.10f}".rstrip("0").rstrip(".")
            if result_str == "-0":
                result_str = "0"
        else:
            result_str = str(result)
        logger.info(
            f"Calculator: Expression '{processed_expression}' evaluated to: {result_str}"
        )
        return json.dumps({"result": result_str})
    except ZeroDivisionError:
        return json.dumps({"error": "Error: Division by zero."})
    except SyntaxError:
        return json.dumps({"error": "Error: Invalid mathematical expression syntax."})
    except TypeError:
        return json.dumps({"error": "Error: Type error in expression."})
    except OverflowError:
        return json.dumps({"error": "Error: Calculation result is too large."})
    except Exception as e:
        logger.error(f"Calculator: Unexpected error: {e}", exc_info=True)
        return json.dumps(
            {"error": f"An unexpected error occurred: {type(e).__name__}."}
        )


_WEATHER_CODES = {
    0: "Clear",
    1: "Mostly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Freezing Fog",
    51: "Light Drizzle",
    53: "Drizzle",
    55: "Heavy Drizzle",
    56: "Freezing Drizzle",
    57: "Heavy Freezing Drizzle",
    61: "Light Rain",
    63: "Rain",
    65: "Heavy Rain",
    66: "Freezing Rain",
    67: "Heavy Freezing Rain",
    71: "Light Snow",
    73: "Snow",
    75: "Heavy Snow",
    77: "Snow Grains",
    80: "Light Showers",
    81: "Showers",
    82: "Violent Showers",
    85: "Light Snow Showers",
    86: "Heavy Snow Showers",
    95: "Thunderstorm",
    96: "Thunderstorm with Hail",
    99: "Severe Thunderstorm",
}


def _weather_code_to_text(code: int) -> str:
    return _WEATHER_CODES.get(code, "Unknown")


def _format_current_weather(
    data: Dict[str, Any], unit: str, location_name: str
) -> Dict[str, Any]:
    u = unit[0].upper()
    return {
        "location": location_name,
        "current": {
            "temperature": f"{data.get('temperature_2m')}°{u}",
            "feels_like": f"{data.get('apparent_temperature')}°{u}",
            "humidity": f"{data.get('relative_humidity_2m')}%",
            "precipitation": f"{data.get('precipitation')} mm",
            "wind": f"{data.get('wind_speed_10m')} km/h",  # Open-Meteo default is km/h for wind_speed_10m
            "condition": _weather_code_to_text(data.get("weather_code", -1)),
        },
    }


def _get_specific_hour_forecast(
    hourly_data: Dict[str, List[Any]],
    hours_ahead_target: int,
    timezone_str: str,
    unit_char: str,
) -> Optional[Dict[str, Any]]:
    try:
        target_timezone = pytz.timezone(timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        logger.error(
            f"Weather: Unknown timezone provided by Open-Meteo: {timezone_str}"
        )
        return {
            "error": f"Internal error: Invalid timezone '{timezone_str}' from weather service."
        }

    if hours_ahead_target < 0 or hours_ahead_target >= len(hourly_data.get("time", [])):
        logger.warning(
            f"Weather: hours_ahead_target {hours_ahead_target} is out of range for available forecast hours ({len(hourly_data.get('time', []))})."
        )
        return {
            "error": f"Requested hour {hours_ahead_target} is beyond forecast range."
        }

    idx = hours_ahead_target
    try:
        timestamp_str = hourly_data["time"][idx]
        dt_obj = datetime.fromisoformat(timestamp_str)
        if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
            dt_obj = target_timezone.localize(dt_obj)
        else:
            dt_obj = dt_obj.astimezone(target_timezone)

        return {
            "time": timestamp_str,
            "local_time": dt_obj.strftime("%a, %H:%M %Z"),
            "temperature": f"{hourly_data['temperature_2m'][idx]}°{unit_char}",
            "feels_like": f"{hourly_data['apparent_temperature'][idx]}°{unit_char}",
            "precipitation_chance": f"{hourly_data['precipitation_probability'][idx]}%",
            "condition": _weather_code_to_text(hourly_data["weather_code"][idx]),
        }
    except (IndexError, KeyError, ValueError) as e:
        logger.error(
            f"Weather: Error processing specific hour forecast at index {idx}: {e}"
        )
        return {"error": "Could not retrieve forecast for the specific hour."}


def _format_hourly_weather(
    data: Dict[str, Any],
    unit: str,
    location_name: str,
    specific_hour: Optional[int],
    timezone_str: str,
    forecast_days: int,
) -> Dict[str, Any]:
    u = unit[0].upper()
    if specific_hour is not None:
        forecast_data = _get_specific_hour_forecast(
            data, specific_hour, timezone_str, u
        )
        if forecast_data and "error" in forecast_data:
            return {"location": location_name, "timeframe": "hourly", **forecast_data}
        return {
            "location": location_name,
            "timeframe": "hourly",
            "forecast_at_specific_hour": forecast_data,
        }

    forecasts = []
    num_entries_to_show = 24 * forecast_days
    try:
        target_timezone = pytz.timezone(timezone_str)
    except pytz.exceptions.UnknownTimeZoneError:
        logger.error(f"Weather: Unknown timezone for hourly formatting: {timezone_str}")
        return {
            "error": f"Internal error: Invalid timezone '{timezone_str}' for hourly display."
        }

    for i, timestamp_str in enumerate(data.get("time", [])):
        if i >= num_entries_to_show:
            break
        try:
            dt_obj = datetime.fromisoformat(timestamp_str)
            if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
                dt_obj = target_timezone.localize(dt_obj)
            else:
                dt_obj = dt_obj.astimezone(target_timezone)

            forecasts.append(
                {
                    "time_utc_iso": timestamp_str,
                    "local_time": dt_obj.strftime("%a, %H:%M %Z"),
                    "temperature": f"{data['temperature_2m'][i]}°{u}",
                    "feels_like": f"{data['apparent_temperature'][i]}°{u}",
                    "precipitation_chance": f"{data['precipitation_probability'][i]}%",
                    "condition": _weather_code_to_text(data["weather_code"][i]),
                }
            )
        except (IndexError, KeyError, ValueError) as e_item:
            logger.warning(
                f"Weather: Skipping hourly item at index {i} due to error: {e_item}"
            )
            continue
    return {"location": location_name, "timeframe": "hourly", "forecasts": forecasts}


def _format_daily_weather(
    data: Dict[str, Any], unit: str, location_name: str
) -> Dict[str, Any]:
    u = unit[0].upper()
    forecasts = []
    for i, date_str in enumerate(data.get("time", [])):
        try:
            dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
            forecasts.append(
                {
                    "date": dt_obj.strftime("%Y-%m-%d (%a)"),
                    "high": f"{data['temperature_2m_max'][i]}°{u}",
                    "low": f"{data['temperature_2m_min'][i]}°{u}",
                    "feels_like_max": f"{data.get('apparent_temperature_max', ['N/A'])[i]}°{u}",
                    "precipitation_chance": f"{data['precipitation_probability_max'][i]}%",
                    "condition": _weather_code_to_text(data["weather_code"][i]),
                }
            )
        except (IndexError, KeyError, ValueError) as e_item:
            logger.warning(
                f"Weather: Skipping daily item at index {i} due to error: {e_item}"
            )
            continue
    return {"location": location_name, "timeframe": "daily", "forecasts": forecasts}


async def get_weather_tool(
    location: str,
    timeframe: str = "current",
    hours_ahead: Optional[int] = None,
    forecast_days: int = 1,
    unit: str = "celsius",
) -> str:
    logger.info(
        f"TOOL: get_weather_tool for {location}, timeframe: {timeframe}, hours: {hours_ahead}, days: {forecast_days}, unit: {unit}"
    )
    async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
        try:
            geo_params = {"q": location, "format": "json", "limit": 1}
            geo_headers = {"User-Agent": "TelegramAIBot/1.0"}
            geo_response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params=geo_params,
                headers=geo_headers,
            )
            geo_response.raise_for_status()
            geo_data = geo_response.json()

            if not geo_data:
                return json.dumps(
                    {
                        "error": "Location not found",
                        "details": f"Could not geocode: {location}",
                    }
                )

            lat = geo_data[0]["lat"]
            lon = geo_data[0]["lon"]
            display_name = geo_data[0]["display_name"]

            max_api_forecast_days = 16
            effective_forecast_days = forecast_days
            if timeframe == "hourly" and hours_ahead is not None:
                days_needed_for_hourly = math.ceil((hours_ahead + 1) / 24)
                effective_forecast_days = max(days_needed_for_hourly, forecast_days)

            effective_forecast_days = min(
                effective_forecast_days, max_api_forecast_days
            )
            effective_forecast_days = max(1, effective_forecast_days)

            weather_params: Dict[str, Any] = {
                "latitude": lat,
                "longitude": lon,
                "temperature_unit": unit,
                "timezone": "auto",
                "forecast_days": effective_forecast_days,
            }

            if timeframe == "current":
                weather_params[
                    "current"
                ] = "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
            elif timeframe == "hourly":
                weather_params[
                    "hourly"
                ] = "temperature_2m,precipitation_probability,weather_code,apparent_temperature,wind_speed_10m"
            elif timeframe == "daily":
                weather_params[
                    "daily"
                ] = "temperature_2m_max,temperature_2m_min,weather_code,apparent_temperature_max,precipitation_probability_max"

            weather_response = await client.get(
                "https://api.open-meteo.com/v1/forecast", params=weather_params
            )
            weather_response.raise_for_status()
            weather_data = weather_response.json()

            result_data = {}
            timezone_returned = weather_data.get("timezone", "UTC")

            if timeframe == "current" and "current" in weather_data:
                result_data = _format_current_weather(
                    weather_data["current"], unit, display_name
                )
            elif timeframe == "hourly" and "hourly" in weather_data:
                result_data = _format_hourly_weather(
                    weather_data["hourly"],
                    unit,
                    display_name,
                    hours_ahead,
                    timezone_returned,
                    effective_forecast_days,
                )
            elif timeframe == "daily" and "daily" in weather_data:
                result_data = _format_daily_weather(
                    weather_data["daily"], unit, display_name
                )
            else:
                return json.dumps(
                    {
                        "error": f"No data for timeframe '{timeframe}' or data missing in response."
                    }
                )

            result_data["coordinates"] = {
                "latitude": float(lat),
                "longitude": float(lon),
            }
            return json.dumps(result_data)
        except httpx.HTTPStatusError as e:
            err_detail = f"HTTP error {e.response.status_code} calling {e.request.url}."
            try:
                api_err = e.response.json()
                if "reason" in api_err:
                    err_detail += f" API Reason: {api_err['reason']}"
            except ValueError:
                pass
            logger.error(f"Weather Tool Error: {err_detail}", exc_info=True)
            return json.dumps(
                {"error": "Weather service request failed.", "details": err_detail}
            )
        except (httpx.RequestError, json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Weather Tool Error: {e}", exc_info=True)
            return json.dumps(
                {
                    "error": "Weather service unavailable or data processing error.",
                    "details": str(e),
                }
            )
        except Exception as e:
            logger.error(f"Unexpected Weather Tool Error: {e}", exc_info=True)
            return json.dumps(
                {
                    "error": "An unexpected error occurred in weather tool.",
                    "details": str(e),
                }
            )


async def web_search_tool(
    query: str, site: str = "", region: str = "", date_filter: str = ""
) -> str:
    logger.info(
        f"TOOL: web_search for query='{query}', site='{site}', region='{region}', date_filter='{date_filter}'"
    )

    # Construct the search query string
    if site:
        query = f"{query} site:{site}"

    params = {"q": query, "kl": region, "df": date_filter, "kp": "-2"}  # No safe search

    try:
        async with httpx.AsyncClient(
            timeout=HTTP_CLIENT_TIMEOUT, follow_redirects=True
        ) as client:
            response = await client.get("https://duckduckgo.com/html/", params=params)
            response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results_list = []
        processed_results_count = 0
        for element_a in soup.select(".result__title a"):
            if processed_results_count >= 5:
                break  # Limit to 5 results

            title = element_a.text.strip()
            raw_url = element_a.get("href")
            if not raw_url:
                logger.warning(
                    f"Web Search: Found a result title ('{title[:50]}...') without an href. Skipping."
                )
                continue

            # Decode DDG's 'uddg' parameter if present
            parsed_url = urlparse(raw_url)
            uddg_list = parse_qs(parsed_url.query).get("uddg")
            if uddg_list and uddg_list[0]:
                clean_url = url_unquote(uddg_list[0])
            else:
                clean_url = raw_url  # Fallback if 'uddg' is not found
                logger.warning(
                    f"Web Search: Could not find 'uddg' param for URL {raw_url} (title: '{title[:50]}...'). Using raw href."
                )

            snippet_text = "Snippet not available."
            # Try to find the snippet more robustly
            result_item_container = element_a.find_parent(
                class_=re.compile(r"\bresult\b")
            )
            if result_item_container:
                snippet_tag = result_item_container.select_one(".result__snippet")
                if snippet_tag:
                    snippet_text = snippet_tag.text.strip()
                else:  # Fallback if typical snippet class not found directly under result container
                    header_candidate = element_a.find_parent(class_="result__title")
                    if header_candidate:
                        header_candidate = (
                            header_candidate.parent
                        )  # up to result__header
                    if header_candidate:
                        snippet_tag_fallback = header_candidate.find_next_sibling(
                            class_="result__snippet"
                        )
                        if snippet_tag_fallback:
                            snippet_text = snippet_tag_fallback.text.strip()
            else:
                logger.warning(
                    f"Web Search: Could not determine result item container for title '{title[:50]}...'. Snippet may be missing."
                )

            results_list.append(
                {
                    "searchResult": f"Web Search Result #{processed_results_count + 1}",
                    "title": title,
                    "searchResultSourceUrl": clean_url,
                    "snippet": snippet_text,
                }
            )
            processed_results_count += 1

        if not results_list:
            return json.dumps({"error": "No results found"})

        # Format as a list of dictionaries as per common tool output patterns
        final_payload_list = [f'Results for search query "{query}":']
        final_payload_list.extend(results_list)  # Add the list of result dicts
        return json.dumps({"results": final_payload_list})
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Web Search: HTTP error {e.response.status_code} for '{query}'. Resp: {e.response.text[:200]}",
            exc_info=False,
        )
        return json.dumps(
            {"error": f"Unable to perform web search: HTTP {e.response.status_code}"}
        )
    except httpx.RequestError as e:
        logger.error(f"Web Search: Request error for '{query}': {e}", exc_info=False)
        return json.dumps(
            {
                "error": f"Unable to perform web search: Request Error - {type(e).__name__}"
            }
        )
    except Exception as e:
        logger.error(
            f"Web Search: Unexpected error during web search for '{query}': {e}",
            exc_info=True,
        )
        return json.dumps({"error": "An unexpected error occurred during web search."})


async def get_game_deals_tool(
    deal_id: Optional[int] = None,
    fetch_worth: bool = False,
    platform: Optional[str] = None,
    type: Optional[str] = None,
    sort_by: Optional[str] = None,
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
        if platform:
            params["platform"] = platform
        if type:
            params["type"] = type
    else:
        api_url = f"{base_url}/giveaways"
        if platform:
            params["platform"] = platform
        if type:
            params["type"] = type
        if sort_by:
            params["sort-by"] = sort_by

    try:
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
            response = await client.get(api_url, params=params)

            # GamerPower API returns 404 for a non-existent ID, which is a valid "not found" case
            if response.status_code == 404:
                return json.dumps(
                    {"result": "No giveaway found with the specified ID."}
                )

            response.raise_for_status()
            data = response.json()

        if not data:
            return json.dumps(
                {"result": "No data returned from the API for the specified criteria."}
            )

        # The 'worth' endpoint returns a single object, not a list
        if fetch_worth:
            return json.dumps({"worth_estimation": data})

        # A single giveaway ID returns a single object
        if deal_id:
            return json.dumps({"giveaway_details": data})

        # Otherwise, we have a list of giveaways
        results_limit = 10  # Limit to a reasonable number for chat
        formatted_results = []
        for item in data[:results_limit]:
            formatted_results.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "platforms": item.get("platforms"),
                    "end_date": item.get("end_date"),
                    "url": item.get("open_giveaway_url"),
                }
            )
        return json.dumps({"giveaways": formatted_results})

    except httpx.HTTPStatusError as e:
        error_details = f"API request failed with status code {e.response.status_code}."
        logger.error(f"GamerPower API Error: {error_details}", exc_info=True)
        return json.dumps(
            {
                "error": "Failed to fetch data from the GamerPower API.",
                "details": error_details,
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_game_deals_tool: {e}", exc_info=True)
        return json.dumps({"error": "An unexpected error occurred."})


async def generate_anime_image_tool(
    prompt: str,
    negative_prompt: str = "",
    orientation: str = "portrait",
    # Parameters to be passed by the main handler:
    telegram_bot_context: Optional[ContextTypes.DEFAULT_TYPE] = None,
    current_chat_id: Optional[int] = None,
    current_message_thread_id: Optional[int] = None,
) -> str:
    """Generates an anime-style image from a prompt using the Perchance API and sends it to the chat."""
    logger.info(
        f"TOOL: generate_anime_image_tool for chat_id {current_chat_id} with prompt='{prompt}' and negative prompt='{negative_prompt}', orientation='{orientation}'"
    )

    if not telegram_bot_context or not current_chat_id:
        err_msg = (
            "Telegram context or chat ID not provided to generate_anime_image_tool."
        )
        logger.error(err_msg)
        return json.dumps(
            {
                "status": "failed",
                "error": err_msg,
                "details": "This tool requires internal context to function.",
            }
        )

    API_BASE_URL = "https://image-generation.perchance.org/api"
    # This key may expire. If it does, you'll need to regenerate it.
    USER_KEY = os.getenv("PERCHANCE_USER_KEY")
    if not USER_KEY:
        return json.dumps(
            {
                "status": "failed",
                "error": "PERCHANCE_USER_KEY is not set in the environment.",
            }
        )

    CHANNEL = "ai-anime-generator"
    SUB_CHANNEL = "public"
    GUIDANCE_SCALE = 7.5

    # --- Map orientation to resolution ---
    orientation_map = {
        "portrait": "512x768",
        "landscape": "768x512",
        "square": "512x512",
    }
    # Get the resolution from the map, defaulting to portrait if the key is invalid
    resolution = orientation_map.get(orientation, "512x768")
    if orientation not in orientation_map:
        logger.warning(
            f"Invalid orientation '{orientation}' provided. Defaulting to portrait."
        )

    prompt_suffix = ""
    #prompt_suffix = ", masterpiece, best quality, world-class masterpiece, 4k, highres, absurdres, extreme detail, anime-inspired artwork, aesthetically pleasing anime art with impeccable attention to detail and beautiful composition"
    full_prompt = f"{prompt}{prompt_suffix}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    }

    IMAGE_GEN_TIMEOUT = httpx.Timeout(90.0, connect=10.0)

    async with httpx.AsyncClient(timeout=IMAGE_GEN_TIMEOUT) as client:
        try:
            generation_params = {
                "prompt": full_prompt,
                "seed": -1,
                "resolution": resolution,  # Use the dynamic resolution
                "guidanceScale": GUIDANCE_SCALE,
                "negativePrompt": negative_prompt,
                "channel": CHANNEL,
                "subChannel": SUB_CHANNEL,
                "userKey": USER_KEY,
            }
            generate_response = await client.get(
                f"{API_BASE_URL}/generate", params=generation_params, headers=headers
            )
            generate_response.raise_for_status()

            generate_data = generate_response.json()

            if generate_data.get("status") != "success":
                # The API call succeeded (e.g., status 200 OK), but the API's own logic failed.
                # Capture the full response and return it to the AI.
                error_detail = generate_data.get("message", "Unknown API error")
                logger.error(
                    f"Perchance API generate failed: {error_detail}. Full response: {generate_data}"
                )
                return json.dumps(
                    {
                        "status": "failed",
                        "error": "Image generation API returned a failure status.",
                        "details": error_detail,
                        "api_response": generate_data, # Provide the full response body
                    }
                )

            if not generate_data.get("imageId"):
                # Handle cases where status might be success but imageId is missing
                error_detail = "API response was 'success' but did not include an imageId."
                logger.error(
                    f"Perchance API logic error: {error_detail}. Full response: {generate_data}"
                )
                return json.dumps(
                    {
                        "status": "failed",
                        "error": "Image generation response was incomplete.",
                        "details": error_detail,
                        "api_response": generate_data,
                    }
                )

            image_id = generate_data["imageId"]
            logger.info(f"Perchance API generated imageId: {image_id}")

            download_params = {"imageId": image_id}
            download_response = await client.get(
                f"{API_BASE_URL}/downloadTemporaryImage",
                params=download_params,
                headers=headers,
            )
            download_response.raise_for_status()
            image_bytes = download_response.content

            await send_photo_to_chat_or_general(
                bot_instance=telegram_bot_context.bot,
                chat_id=current_chat_id,
                photo=image_bytes,
                preferred_thread_id=current_message_thread_id,
                # caption=f'🎨 Vision: "{prompt}"\nNegative: "{negative_prompt}"\n({orientation})',
            )

            logger.info(
                f"Successfully generated and sent anime image to chat {current_chat_id}"
            )
            return json.dumps(
                {
                    "status": "success",
                    "detail": f"An image containing/representing '{prompt}' was successfully sent to the chat.",
                }
            )

        except httpx.TimeoutException:
            error_details = "The image generation service took too long to respond (more than 90 seconds)."
            logger.error(f"Perchance API Timeout: {error_details}", exc_info=True)
            return json.dumps(
                {
                    "status": "failed",
                    "error": "Image generation timed out.",
                    "details": error_details,
                }
            )
        except httpx.HTTPStatusError as e:
            error_details = (
                f"API request failed with HTTP status code {e.response.status_code}."
            )
            full_response_text = e.response.text
            logger.error(
                f"Perchance API HTTP Error: {error_details} Response: {full_response_text}",
                exc_info=True,
            )
            return json.dumps(
                {
                    "status": "failed",
                    "error": "Failed to communicate with the image generation API.",
                    "details": error_details,
                    "api_response_text": full_response_text,
                }
            )
        except json.JSONDecodeError as e:
            error_details = "Failed to parse the response from the image generation API. It may be down."
            logger.error(f"Perchance API JSON Error: {error_details}", exc_info=True)
            return json.dumps(
                {
                    "status": "failed",
                    "error": "Invalid response from API.",
                    "details": error_details,
                }
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in generate_anime_image_tool: {e}", exc_info=True
            )
            return json.dumps(
                {
                    "status": "failed",
                    "error": f"An unexpected error occurred: {type(e).__name__}",
                }
            )

# --- ADD THIS HELPER FUNCTION FOR PARSING DURATION ---
def _parse_duration(duration_str: str) -> timedelta:
    """Parses a simple duration string (e.g., '30m', '2h', '1d') into a timedelta."""
    duration_str = duration_str.lower().strip()
    match = re.match(r"^(\d+)([mhd])$", duration_str)
    if not match:
        raise ValueError(
            "Invalid duration format. Use 'm' for minutes, 'h' for hours, 'd' for days. E.g., '30m', '2h'."
        )

    value, unit = int(match.group(1)), match.group(2)
    if unit == "m":
        return timedelta(minutes=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    return timedelta()  # Should not be reached


# --- THE MODERATION TOOL IMPLEMENTATION ---
async def restrict_user_in_chat_tool(
    user_id: int,
    duration: str = "1h",
    reason: Optional[str] = None,
    # Parameters to be passed by the main handler:
    telegram_bot_context: Optional[ContextTypes.DEFAULT_TYPE] = None,
    current_chat_id: Optional[int] = None,
) -> str:
    """Restricts a user in the current chat."""
    if not telegram_bot_context or not current_chat_id:
        return json.dumps({"error": "Internal error: Context or Chat ID not provided."})

    bot = telegram_bot_context.bot
    logger.info(
        f"TOOL: restrict_user_in_chat for user {user_id} in chat {current_chat_id} for {duration}. Reason: {reason}"
    )

    # --- SAFETY CHECK 1: Bot cannot restrict itself ---
    if user_id == bot.id:
        return json.dumps(
            {"status": "failed", "error": "Action failed: I cannot restrict myself."}
        )

    try:
        # --- SAFETY CHECK 2: Get bot's status to ensure it's an admin ---
        bot_member = await bot.get_chat_member(current_chat_id, bot.id)
        if not isinstance(
            bot_member, (telegram.ChatMemberAdministrator, telegram.ChatMemberOwner)
        ):
            return json.dumps(
                {
                    "status": "failed",
                    "error": "Action failed: I am not an administrator in this chat.",
                }
            )
        if not bot_member.can_restrict_members:
            return json.dumps(
                {
                    "status": "failed",
                    "error": "Action failed: I am an admin, but I don't have permission to restrict members.",
                }
            )

        # --- SAFETY CHECK 3: Get target user's status to ensure they are not an admin ---
        target_member = await bot.get_chat_member(current_chat_id, user_id)
        if target_member.status in [target_member.ADMINISTRATOR, target_member.CREATOR]:
            return json.dumps(
                {
                    "status": "failed",
                    "error": f"Action failed: I cannot restrict another administrator or the chat owner ('{target_member.user.full_name}').",
                }
            )

        # --- Parse duration and set restriction end time ---
        try:
            mute_timedelta = _parse_duration(duration)
            # Telegram expects a Unix timestamp for `until_date`
            until_date = datetime.now() + mute_timedelta
            logger.info(
                f"User {user_id} will be restricted until {until_date.isoformat()}"
            )
        except ValueError as e:
            return json.dumps({"status": "failed", "error": str(e)})

        # --- Perform the action ---
        # To "mute", we set can_send_messages to False. All other perms remain default (None = Unchanged).
        await bot.restrict_chat_member(
            chat_id=current_chat_id,
            user_id=user_id,
            permissions=ChatPermissions(can_send_messages=False),
            until_date=until_date,
        )

        success_message = f"User {user_id} ({target_member.user.full_name}) has been successfully muted for {duration}."
        if reason:
            success_message += f" Reason: {reason}"

        return json.dumps({"status": "success", "details": success_message})

    except telegram.error.BadRequest as e:
        logger.error(
            f"Error restricting user {user_id} in chat {current_chat_id}: {e.message}"
        )
        if "user not found" in e.message.lower():
            return json.dumps(
                {
                    "status": "failed",
                    "error": "User with the specified ID was not found in this chat.",
                }
            )
        return json.dumps(
            {"status": "failed", "error": f"Telegram API error: {e.message}"}
        )
    except Exception as e:
        logger.error(
            f"Unexpected error in restrict_user_in_chat_tool: {e}", exc_info=True
        )
        return json.dumps({"error": "An unexpected internal error occurred."})


async def get_user_info_tool(
    user_id: int,
    fetch_profile_photos: bool = False,
    # Parameters to be passed by the main handler:
    telegram_bot_context: Optional[ContextTypes.DEFAULT_TYPE] = None,
    current_chat_id: Optional[int] = None,
) -> str:
    """Gets comprehensive information about a member of the current chat."""
    if not telegram_bot_context or not current_chat_id:
        return json.dumps({"error": "Internal error: Context or Chat ID not provided."})

    bot = telegram_bot_context.bot
    logger.info(
        f"TOOL: Comprehensive get_user_info for user {user_id} in chat {current_chat_id}"
    )

    try:
        # --- Get the ChatMember object ---
        # This is the most important call, as it confirms the user is in the chat
        # and gives us both their User object and their chat-specific status.
        member = await bot.get_chat_member(current_chat_id, user_id)
        user = member.user

        # --- Assemble the result dictionary ---
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
                "language_code": user.language_code or "N/A",
            },
            "chat_specific_info": {
                "status": member.status,
            },
        }

        # --- Add context-specific details based on status ---
        chat_info = result["chat_specific_info"]

        if isinstance(
            member, (telegram.ChatMemberRestricted, telegram.ChatMemberBanned)
        ):
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

        # --- Optionally fetch profile photos ---
        if fetch_profile_photos:
            try:
                profile_photos = await bot.get_user_profile_photos(
                    user_id, limit=3
                )  # Limit to 3 to keep response size sane
                result["user_profile"]["profile_photos"] = {
                    "total_count": profile_photos.total_count,
                    "fetched_photos": [
                        {
                            "file_id": p[
                                -1
                            ].file_id,  # Get the file_id of the largest photo
                            "file_unique_id": p[-1].file_unique_id,
                            "width": p[-1].width,
                            "height": p[-1].height,
                        }
                        for p in profile_photos.photos
                    ],
                }
            except Exception as e:
                logger.warning(
                    f"Could not fetch profile photos for user {user_id}: {e}"
                )
                result["user_profile"]["profile_photos"] = {
                    "error": "Could not fetch profile photos."
                }

        return json.dumps(result, indent=2)

    except telegram.error.BadRequest as e:
        if "user not found" in e.message.lower():
            return json.dumps(
                {
                    "status": "not_found",
                    "error": "User with the specified ID was not found in this chat.",
                }
            )
        return json.dumps(
            {"status": "failed", "error": f"Telegram API error: {e.message}"}
        )
    except Exception as e:
        logger.error(
            f"Unexpected error in comprehensive get_user_info_tool: {e}", exc_info=True
        )
        return json.dumps({"error": "An unexpected internal error occurred."})


ALL_AVAILABLE_TOOLS_PYTHON_FUNCTIONS = {
    "calculator": calculator_tool,
    "get_weather": get_weather_tool,
    "web_search": web_search_tool,
    "create_poll_in_chat": create_poll_tool,
    "get_game_deals": get_game_deals_tool,
    "restrict_user_in_chat": restrict_user_in_chat_tool,
    "get_user_info": get_user_info_tool,
    "generate_anime_image": generate_anime_image_tool,
}

ALL_TOOL_DEFINITIONS_FOR_API: list[ChatCompletionToolParam] = [
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
                        "description": "The mathematical expression to evaluate (e.g., '2+2', '(5*8-3)/2', '2^10').",
                    }
                },
                "required": ["expression"],
            },
        },
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
                        "description": "The city and country, e.g., London, UK or a specific address.",
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["current", "hourly", "daily"],
                        "default": "current",
                        "description": "Time resolution of the forecast: 'current' for current conditions, 'hourly' for hour-by-hour, 'daily' for day summaries.",
                    },
                    "hours_ahead": {
                        "type": "number",
                        "description": "Optional. For 'hourly' timeframe, specifies the number of hours from now for which to get a single forecast point (e.g., 0 for current hour, 1 for next hour). Max 167.",
                    },
                    "forecast_days": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 14,
                        "default": 1,
                        "description": "Number of days to forecast (for daily or hourly if hours_ahead is not specified). E.g., 1 for today, 7 for a week.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                        "description": "Temperature unit.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web/online/on the internet for information on a given topic. Fetches only the first page of search results from DuckDuckGo. Use when you require information you are unsure or unaware of.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g., 'What is the capital of France?'",
                    },
                    "site": {
                        "type": "string",
                        "description": "Optional. Limit search to a specific website (e.g., 'wikipedia.org') or use a DuckDuckGo bang (e.g., '!w' for Wikipedia). This is passed to DuckDuckGo's 'b' (bang) parameter. Leave empty for general search.",
                    },
                    "region": {
                        "type": "string",
                        "description": "Optional. Limit search to results from a specific region/language (e.g., 'us-en' for US English, 'de-de' for Germany German). This is a DuckDuckGo region code passed to 'kl' parameter. Leave empty for global results.",
                    },
                    "date_filter": {
                        "type": "string",
                        "description": "Optional. Filter search results by date: 'd' (past day), 'w' (past week), 'm' (past month), 'y' (past year). Passed to DuckDuckGo's 'df' parameter. Leave empty for no date filter.",
                        "enum": ["", "d", "w", "m", "y"],
                    },
                },
                "required": ["query"],
            },
        },
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
                        "description": "The question for the poll. Max 300 characters.",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 10,
                        "description": "A list of 2 to 10 answer options for the poll. Each option max 100 characters.",
                    },
                    "is_anonymous": {
                        "type": "boolean",
                        "default": True,
                        "description": "Optional. If true, the poll is anonymous. Defaults to true.",
                    },
                    "allows_multiple_answers": {
                        "type": "boolean",
                        "default": False,
                        "description": "Optional. If true, users can select multiple answers. Defaults to false.",
                    },
                },
                "required": ["question", "options"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_game_deals",
            "description": "Fetches information about free game giveaways. Can fetch a list of deals, a single deal by ID, or the total worth of all active deals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "deal_id": {
                        "type": "integer",
                        "description": "The unique ID of a specific giveaway to fetch details for.",
                    },
                    "fetch_worth": {
                        "type": "boolean",
                        "description": "Set to true to get the total number and estimated USD value of all active giveaways instead of a list.",
                    },
                    "platform": {
                        "type": "string",
                        "description": "The platform to filter giveaways for. e.g., 'pc', 'steam', 'epic-games-store'.",
                        "enum": [
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
                            "xbox-360",
                        ],
                    },
                    "type": {
                        "type": "string",
                        "description": "The type of giveaway to filter for.",
                        "enum": ["game", "loot", "beta"],
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "How to sort the results when fetching a list.",
                        "enum": ["date", "value", "popularity"],
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restrict_user_in_chat",
            "description": "Temporarily mutes a user in the telegram chat, preventing them from sending messages. Requires the user's telegram ID. You must be an admin with permission to restrict members. Do not abuse this, keep mute durations low unless the offense truly warrants more.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "The unique integer ID of the user to mute. You can get this from a user's context or the get_user_info tool.",
                    },
                    "duration": {
                        "type": "string",
                        "description": "The duration for the mute. Examples: '30m' for 30 minutes, '2h' for 2 hours, '1d' for 1 day. If not provided, defaults to 1 hour.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional. The reason for the restriction, which will be logged.",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_info",
            "description": "Retrieves comprehensive information about a user, including their global profile, chat-specific status (like admin rights or restrictions), and optionally their profile pictures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "The unique integer ID of the user to look up. You can get this from a user's context.",
                    },
                    "fetch_profile_photos": {
                        "type": "boolean",
                        "default": False,
                        "description": "Set to true to also fetch the user's profile pictures. This is an extra step and may not always be necessary.",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_anime_image",
            "description": "Generates a high-quality, anime-style image based on a descriptive prompt. Use this to create visual representations of scenes, characters, or concepts in an anime art style. Depictions of violence, sex, and other NSFW content is allowed, there is no filter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A detailed description of the desired image. For example: '1girl, 1woman, golden hair, looking at viewer, castle at the end of the world, masterpiece, best quality, world-class masterpiece, 4k, highres, absurdres, extreme detail, anime-inspired artwork, aesthetically pleasing anime art with impeccable attention to detail and beautiful composition'.",
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Optional. A description of things to avoid in the image, e.g., 'lowres, bad anatomy, ((bad hands)), text, (error), ((missing fingers)), extra digit, fewer digits, awkward fingers, cropped, jpeg artifacts, worst quality, low quality, signature, watermark, username, blurry, extra ears, (deformed, disfigured, mutation, extra limbs:1.5)'.",
                    },
                    "orientation": {
                        "type": "string",
                        "description": "The orientation of the image. Defaults to 'portrait' if not specified.",
                        "default": "portrait",
                        "enum": ["portrait", "landscape", "square"],
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]
# --- END OF TOOL DEFINITIONS ---

# --- Granular Tool Control Logic ---
# These will hold the tools that are actually active based on the .env configuration.
ACTIVE_TOOL_DEFINITIONS: list[ChatCompletionToolParam] = []
ACTIVE_TOOLS_PYTHON_FUNCTIONS: Dict[str, Any] = {}

active_tools_env = os.getenv("ACTIVE_TOOLS")

if active_tools_env is None:
    # If the variable is not set at all, enable all tools for backward compatibility.
    logger.info("ACTIVE_TOOLS environment variable not set. Activating all available tools.")
    ACTIVE_TOOL_DEFINITIONS = ALL_TOOL_DEFINITIONS_FOR_API
    ACTIVE_TOOLS_PYTHON_FUNCTIONS = ALL_AVAILABLE_TOOLS_PYTHON_FUNCTIONS
else:
    # If the variable is set (even if empty), parse it.
    # An empty string means NO tools should be active.
    active_tool_names = {name.strip() for name in active_tools_env.split(',') if name.strip()}
    logger.info(f"ACTIVE_TOOLS configured. Activating: {sorted(list(active_tool_names)) if active_tool_names else 'None'}")

    # Filter the API definitions
    for tool_def in ALL_TOOL_DEFINITIONS_FOR_API:
        if tool_def.get("function", {}).get("name") in active_tool_names:
            ACTIVE_TOOL_DEFINITIONS.append(tool_def)

    # Filter the Python functions
    for tool_name, func in ALL_AVAILABLE_TOOLS_PYTHON_FUNCTIONS.items():
        if tool_name in active_tool_names:
            ACTIVE_TOOLS_PYTHON_FUNCTIONS[tool_name] = func

    # Log a warning for any invalid tool names specified in the .env file
    all_possible_tool_names = set(ALL_AVAILABLE_TOOLS_PYTHON_FUNCTIONS.keys())
    invalid_names = active_tool_names - all_possible_tool_names
    if invalid_names:
        logger.warning(
            f"Invalid tool names found in ACTIVE_TOOLS environment variable and will be ignored: {sorted(list(invalid_names))}"
        )
# --- END OF Granular Tool Control Logic ---

# --- UTILITY FUNCTIONS ---
async def send_telegramify_message(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    markdown_content: str,
    preferred_thread_id: Optional[int],
) -> None:
    """
    Processes a markdown string using telegramify_markdown and sends it to the chat.
    Handles text splitting, code blocks (as images/files), and other rich content.
    """
    if not markdown_content or not markdown_content.strip():
        logger.warning("send_telegramify_message was called with empty or whitespace-only content.")
        return

    try:
        # The library is powerful. It can render code blocks as text, files, or images.
        # It can also handle things like Mermaid diagrams.
        interpreter_chain = InterpreterChain([
            TextInterpreter(),       # Renders most things as pure text
            FileInterpreter(),       # Renders code blocks as files (or images if short)
            MermaidInterpreter()     # Renders mermaid diagrams as images
        ])

        # This is the core call. It processes the entire markdown string from the AI.
        boxes = await telegramify_markdown.telegramify(
            content=markdown_content,
            interpreters_use=interpreter_chain,
            normalize_whitespace=True, # Recommended for cleaner output
            latex_escape=True, # Escape LaTeX syntax to prevent rendering issues
        )

        logger.info(f"Chat {chat_id}: telegramify created {len(boxes)} message parts to send.")

        for i, item in enumerate(boxes):
            # We add a retry loop with exponential backoff for flood control, a good practice.
            for attempt in range(4): # 0, 1, 2, 3
                try:
                    if item.content_type == ContentTypes.TEXT:
                        await send_message_to_chat_or_general(
                            context.bot,
                            chat_id,
                            item.content,
                            preferred_thread_id=preferred_thread_id,
                            parse_mode=ParseMode.MARKDOWN_V2,
                            disable_web_page_preview=True
                        )
                    elif item.content_type == ContentTypes.PHOTO:
                        # The library provides file_data as bytes. We pass it directly to the 'photo' param.
                        # The filename must be passed as a separate, named argument.
                        await send_photo_to_chat_or_general(
                            context.bot,
                            chat_id,
                            photo=item.file_data,  # Pass the raw bytes here
                            preferred_thread_id=preferred_thread_id,
                            caption=item.caption,
                            filename=item.file_name,  # Pass the filename as a named argument
                            parse_mode=ParseMode.MARKDOWN_V2
                        )
                    elif item.content_type == ContentTypes.FILE:
                        await send_document_to_chat_or_general(
                            context.bot,
                            chat_id,
                            document=item.file_data,  # Pass the raw bytes here
                            preferred_thread_id=preferred_thread_id,
                            caption=item.caption,
                            filename=item.file_name,  # Pass the filename as a named argument
                            parse_mode=ParseMode.MARKDOWN_V2
                        )
                    
                    break # Success, exit retry loop

                except telegram.error.RetryAfter as e:
                    wait_time = e.retry_after
                    logger.warning(f"Chat {chat_id}: Flood control triggered. Waiting {wait_time}s before retry {attempt + 1}.")
                    if wait_time > 20: # Don't wait forever
                        raise e 
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    logger.error(f"Chat {chat_id}: Failed to send telegramify part {i+1} (type: {item.content_type}). Error: {e}", exc_info=True)
                    # For most errors other than RetryAfter, we probably shouldn't retry.
                    raise e # Re-raise to be caught by the main handler

            # A small, polite delay between sending multiple message parts.
            if i < len(boxes) - 1:
                await asyncio.sleep(0.8)

    except Exception as e:
        logger.error(f"Chat {chat_id}: Major error in send_telegramify_message. Error: {e}", exc_info=True)
        # Fallback to sending a simple error message to the user
        try:
            await send_message_to_chat_or_general(
                context.bot,
                chat_id,
                "I encountered a problem while formatting my response. The admin has been notified.",
                preferred_thread_id=preferred_thread_id,
            )
        except Exception as e_fb:
            logger.error(f"Chat {chat_id}: Failed to send even the fallback error message. Error: {e_fb}")

def get_display_name(user: Optional[TelegramUser]) -> str:
    if not user:
        return "Unknown User"
    name = user.full_name
    if user.username:
        name = f"{name} (@{user.username})" if name and user.username else user.username
    if not name:
        name = f"User_{user.id}"
    return name


def is_user_or_chat_allowed(user_id: int, chat_id: int) -> bool:
    if not ALLOWED_USERS and not ALLOWED_CHATS:
        return True
    if ALLOWED_USERS and str(user_id) in ALLOWED_USERS:
        return True
    if ALLOWED_CHATS and str(chat_id) in ALLOWED_CHATS:
        return True
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
                preferred_thread_id=message_thread_id,
            )
        except Exception as e:
            logger.error(
                f"Failed to send permission denied message to {update.effective_chat.id} (thread {message_thread_id}): {e}"
            )


def get_llm_chat_history(
    chat_id: int, effective_topic_context_id: Optional[int]  # REFACTORED
) -> list[ChatCompletionMessageParam]:
    # Use effective_topic_context_id for the history key
    history_key = (chat_id, effective_topic_context_id)
    if history_key not in chat_histories:
        chat_histories[history_key] = []

    current_thread_history = chat_histories[history_key]
    # Trim history if it gets excessively long
    if len(current_thread_history) > MAX_HISTORY_LENGTH * 7:
        trimmed_length = MAX_HISTORY_LENGTH * 5
        logger.info(
            f"Trimming LLM chat history for key {history_key} from {len(current_thread_history)} to {trimmed_length}"
        )
        chat_histories[history_key] = current_thread_history[-trimmed_length:]
    return chat_histories[history_key]


def add_to_raw_group_log(
    chat_id: int,
    effective_topic_context_id: Optional[int],
    sender_name: str,
    text: str,  # REFACTORED
):
    # Use effective_topic_context_id for the log key
    if chat_id not in group_raw_message_log:
        group_raw_message_log[chat_id] = {}
    if effective_topic_context_id not in group_raw_message_log[chat_id]:
        group_raw_message_log[chat_id][effective_topic_context_id] = []

    log_entry = {
        "sender_name": sender_name,
        "text": text if text else "[empty message content]",
    }
    current_thread_log = group_raw_message_log[chat_id][effective_topic_context_id]
    current_thread_log.append(log_entry)

    if len(current_thread_log) > MAX_RAW_LOG_PER_THREAD:
        group_raw_message_log[chat_id][effective_topic_context_id] = current_thread_log[
            -MAX_RAW_LOG_PER_THREAD:
        ]

    topic_desc = (
        f"topic ID {effective_topic_context_id}"
        if effective_topic_context_id is not None
        else "general chat"
    )
    # Changed to debug from info for less noise
    logger.debug(
        f"Raw message log for group {chat_id} ({topic_desc}) updated. Length: {len(group_raw_message_log[chat_id][effective_topic_context_id])} messages."
    )


def format_freewill_context_from_raw_log(
    chat_id: int,
    effective_topic_context_id: Optional[
        int
    ],  # REFACTORED: This ID determines the specific log to use (e.g., a Telegram message_thread_id)
    num_messages_to_include: int,
    bot_name: str,
    current_chat_title: Optional[str],
    current_topic_name_if_known: Optional[
        str
    ],  # This name should correspond to effective_topic_context_id
    replied_to_message_context: Optional[Dict[str, str]] = None,
) -> str:
    # Build a descriptive location string
    location_description_parts = []
    if current_chat_title:
        location_description_parts.append(f"group '{current_chat_title}'")
    else:  # Fallback if title not passed or None (e.g. private chat or missing group info)
        location_description_parts.append(f"a group chat (ID: {chat_id})")

    if effective_topic_context_id is not None:
        if (
            current_topic_name_if_known
        ):  # If the name for this topic ID is known/resolved
            location_description_parts.append(f"topic '{current_topic_name_if_known}'")
        else:  # Fallback if name not resolved for the effective ID
            location_description_parts.append(f"topic ID {effective_topic_context_id}")
    elif (
        current_chat_title
    ):  # In a group, but not a specific topic (i.e., the "General" area of a group with topics)
        location_description_parts.append("the general chat area")

    # Default if no parts (e.g., private chat without title, or unexpected scenario)
    topic_desc_log = (
        " ".join(location_description_parts)
        if location_description_parts
        else "this chat"
    )

    # Check if log exists for the chat AND the specific effective_topic_context_id
    log_for_this_topic = group_raw_message_log.get(chat_id, {}).get(
        effective_topic_context_id
    )
    if (
        not log_for_this_topic  # Check if the list itself is None or empty
        or num_messages_to_include <= 0
    ):
        return f"[Recent conversation context for {topic_desc_log} is minimal or unavailable.]\n"

    # log_for_this_topic is now guaranteed to be a list (possibly empty if num_messages_to_include was 0, handled next)
    start_index = max(0, len(log_for_this_topic) - num_messages_to_include)
    context_messages_to_format = log_for_this_topic[start_index:]

    if (
        not context_messages_to_format
    ):  # Should only happen if num_messages_to_include was 0 or log was empty initially
        return f"[No prior messages in raw log for {topic_desc_log} to form context.]\n"

    formatted_context_parts = [f"[Recent conversation excerpt from {topic_desc_log}:]"]
    triggering_user_name = "Unknown User"  # Default value
    # Initialize triggering_message_text with the original text of the last message in context.
    # This will be updated if the last message text gets neutralized or truncated.
    triggering_message_text = context_messages_to_format[-1].get(
        "text", "[message content not available]"
    )

    # --- START OF LOGIC TO ADDRESS PREFIX COMMANDS IN FREE WILL CONTEXT ---
    # If a message is part of free will context and contains a prefix command (e.g., !command),
    # neutralize it to prevent the API (e.g., Shapes API) from misinterpreting it if it has built-in commands.
    # We replace the command with a version that's still readable for AI context
    # but shouldn't be directly executable.
    # (Original issue: API might have built-in prefix commands like '!wack', '!dashboard' etc.)
    # (Original issue: These could get triggered if the command was in one of the messages taken into free will context.)
    # (Refinement: Neutralize any command starting with '!' to '!_'.)

    # Helper function to neutralize a command match
    def neutralize_command(match: re.Match) -> str:
        # match.group(0) is the full matched command, e.g., "!wack"
        # We want to return "!_wack"
        return f"!_{match.group(0)[1:]}"  # Prepend "_" after the "!"

    for i, msg_data in enumerate(context_messages_to_format):
        sender = msg_data.get("sender_name", "Unknown User")
        text = msg_data.get("text", "[message content not available]")
        original_text_for_this_message = (
            text  # Keep a copy of the original for this message
        )

        # Apply command neutralization to all occurrences in the text.
        # The regex `(?<!\S)(!\w+)` ensures the '!' is at the beginning of a "word"
        # (i.e., preceded by a non-non-whitespace character, which means whitespace or start of string).
        # `(!\w+)` captures the '!' and the subsequent word characters (the command).
        # Using re.sub with a function allows dynamic replacement for each found command.
        modified_text = re.sub(r"(?<!\S)(!\w+)", neutralize_command, text)

        if (
            modified_text != text
        ):  # Log if any command was actually neutralized in this message
            logger.debug(
                f"Free will context: Neutralized prefix commands in message from '{sender}'. Original: '{text[:100]}...', Modified: '{modified_text[:100]}...'"
            )
            text = modified_text  # Use the modified text
        # --- END OF LOGIC TO ADDRESS PREFIX COMMANDS IN FREE WILL CONTEXT (for this message) ---

        max_len_per_msg_in_context = (
            4096  # Max length for individual messages in context
        )
        if len(text) > max_len_per_msg_in_context:
            text = text[:max_len_per_msg_in_context].strip() + "..."
        formatted_context_parts.append(f"- '{sender}' said: \"{text}\"")

        # Capture the last message details for the final prompt part.
        if i == len(context_messages_to_format) - 1:
            triggering_user_name = sender
            # If the text of the *last message* was modified by neutralization,
            # `triggering_message_text` should reflect this modified (and potentially truncated) version.
            if modified_text != original_text_for_this_message:
                triggering_message_text = (
                    text  # Use the modified, already truncated if necessary, text
                )
            # If the original text was not modified but is too long, truncate it.
            elif len(triggering_message_text) > max_len_per_msg_in_context:
                triggering_message_text = (
                    triggering_message_text[:max_len_per_msg_in_context].strip() + "..."
                )
            # Otherwise, triggering_message_text (initialized from original) is used as is.

    reply_context_addon = ""
    if replied_to_message_context:
        replied_author = replied_to_message_context.get("author", "someone")
        replied_content = replied_to_message_context.get("content", "[their message]")
        reply_context_addon = (
            f" (in reply to '{replied_author}' who said: \"{replied_content}\")"
        )

    formatted_context_parts.append(
        f"\n[You are '{bot_name}', chatting on Telegram in {topic_desc_log}. Based on the excerpt above, where '{triggering_user_name}' "
        f'just said: "{triggering_message_text}"{reply_context_addon}, '
        "make a relevant and in character interjection or comment. Be concise and natural.]"
    )
    return "\n".join(formatted_context_parts) + "\n\n"


async def _keep_typing_loop(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message_thread_id: Optional[int],
    action: str = ChatAction.TYPING,
    interval: float = 4.5,
):
    while True:
        try:
            await context.bot.send_chat_action(
                chat_id=chat_id, action=action, message_thread_id=message_thread_id
            )
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
        except telegram.error.BadRequest as e:
            # Handle case where the thread might have been deleted while typing
            if (
                "message thread not found" in e.message.lower()
                and message_thread_id is not None
            ):
                logger.warning(
                    f"Typing loop: Thread {message_thread_id} not found for chat {chat_id}. Stopping typing for this thread."
                )
                break  # Stop trying for this specific (now invalid) thread
            logger.warning(
                f"Error sending {action} action in loop for chat {chat_id} (thread {message_thread_id}): {e}"
            )
            await asyncio.sleep(
                interval
            )  # Continue loop even on other BadRequest errors, but log it
        except Exception as e:
            logger.warning(
                f"Error sending {action} action in loop for chat {chat_id} (thread {message_thread_id}): {e}"
            )
            await asyncio.sleep(
                interval
            )  # Continue loop even on other errors, but log it


async def _process_media_and_documents(
    message: TelegramMessage,
) -> tuple[bool, bool, str, list[dict[str, Any]]]:
    """
    Processes photos, voice, and documents from a message.
    Returns a tuple of (has_image, has_voice, appended_text_from_files, media_parts_list).
    """
    has_image = False
    has_voice = False
    appended_text = ""
    media_parts = []  # Create a local list
    chat_id = message.chat_id

    # 1. Process standard Photo object
    if message.photo:
        has_image = True
        photo_file = await message.photo[-1].get_file()
        file_bytes = await photo_file.download_as_bytearray()
        base64_image = base64.b64encode(file_bytes).decode("utf-8")
        mime_type = (
            mimetypes.guess_type(photo_file.file_path or "img.jpg")[0] or "image/jpeg"
        )
        media_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            }
        )
        logger.info(f"Chat {chat_id}: Added image_url from photo object.")

    # 1.5 Process Sticker object (as an image)
    if message.sticker:
        # We can only process static stickers (like .webp), not animated (.tgs) or video (.webm) ones.
        if not message.sticker.is_animated and not message.sticker.is_video:
            has_image = True
            sticker_file = await message.sticker.get_file()
            file_bytes = await sticker_file.download_as_bytearray()
            base64_image = base64.b64encode(file_bytes).decode("utf-8")
            # Stickers are often in webp format, which vision models can handle.
            mime_type = (
                mimetypes.guess_type(sticker_file.file_path or "sticker.webp")[0]
                or "image/webp"
            )
            media_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )
            logger.info(f"Chat {chat_id}: Added image_url from static sticker.")
        else:
            # If the sticker is animated, we can't send it to the AI as an image.
            # Instead, we'll just add a text note about it for context.
            appended_text += "\n\n[INFO: An animated sticker was sent. Unfortunately, only static images can be processed, not animations.]"
            logger.info(
                f"Chat {chat_id}: Ignored animated/video sticker for image processing."
            )

    # 2. Process Voice message
    if message.voice:
        has_voice = True
        try:
            voice_file = await message.voice.get_file()
            if voice_file.file_path:
                media_parts.append(
                    {"type": "audio_url", "audio_url": {"url": voice_file.file_path}}
                )
                logger.info(
                    f"Chat {chat_id}: Added audio_url from voice object: {voice_file.file_path}"
                )
            else:
                logger.warning(
                    f"Chat {chat_id}: Could not get file_path for voice message."
                )
        except Exception as e:
            logger.error(
                f"Chat {chat_id}: Error processing voice message: {e}", exc_info=True
            )

    # 3. Process Document (could be an image, audio, or text file)
    if doc := message.document:
        doc_mime = doc.mime_type or "application/octet-stream"
        doc_name = doc.file_name or "file"

        if doc_mime.startswith("image/"):
            has_image = True
            doc_file = await doc.get_file()
            file_bytes = await doc_file.download_as_bytearray()
            base64_image = base64.b64encode(file_bytes).decode("utf-8")
            media_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{doc_mime};base64,{base64_image}"},
                }
            )
            logger.info(f"Chat {chat_id}: Added image_url from document '{doc_name}'.")

        elif doc_mime.startswith("audio/"):
            has_voice = True
            try:
                doc_file = await doc.get_file()
                if doc_file.file_path:
                    media_parts.append(
                        {"type": "audio_url", "audio_url": {"url": doc_file.file_path}}
                    )
                    logger.info(
                        f"Chat {chat_id}: Added audio_url from document '{doc_name}'."
                    )
            except Exception as e:
                logger.error(
                    f"Chat {chat_id}: Error processing audio document '{doc_name}': {e}",
                    exc_info=True,
                )

        else:  # Attempt to process as a text file
            docx_mimes = (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            odt_mimes = ("application/vnd.oasis.opendocument.text",)
            plain_text_mimes = (
                "text/",
                "application/json",
                "application/xml",
                "application/javascript",
            )
            text_extensions = {
                ".txt",
                ".md",
                ".markdown",
                ".json",
                ".xml",
                ".csv",
                ".py",
                ".js",
                ".html",
                ".css",
            }

            is_known_text_mime = (
                any(doc_mime.startswith(m) for m in plain_text_mimes)
                or doc_mime in docx_mimes
                or doc_mime in odt_mimes
            )
            is_known_text_ext = doc_name and any(
                doc_name.lower().endswith(ext) for ext in text_extensions
            )

            if is_known_text_mime or is_known_text_ext:
                if doc.file_size and doc.file_size > MAX_TEXT_FILE_SIZE:
                    logger.warning(
                        f"Chat {chat_id}: Ignoring document '{doc_name}' ({doc.file_size} bytes) due to size limit."
                    )
                    appended_text += f"\n\n[INFO: An attached document '{doc_name}' was ignored because it is too large.]"
                else:
                    file_content = None
                    try:
                        file_bytes = await (
                            await doc.get_file()
                        ).download_as_bytearray()
                        if doc_mime in docx_mimes:
                            file_stream = BytesIO(file_bytes)
                            doc_obj = docx.Document(file_stream)
                            file_content = "\n".join(
                                [p.text for p in doc_obj.paragraphs if p.text.strip()]
                            )
                        elif doc_mime in odt_mimes:
                            doc_obj = odf_load(BytesIO(file_bytes))
                            all_paras = doc_obj.getElementsByType(text.P)
                            file_content = "\n".join(
                                teletype.extractText(p)
                                for p in all_paras
                                if teletype.extractText(p).strip()
                            )
                        else:  # Plain text
                            try:
                                file_content = file_bytes.decode("utf-8")
                            except UnicodeDecodeError:
                                file_content = file_bytes.decode("latin-1")
                    except Exception as e:
                        logger.error(
                            f"Failed to read or parse text file '{doc_name}': {e}",
                            exc_info=True,
                        )

                    if file_content is not None:
                        appended_text += f"\n\n[Content of uploaded file '{doc_name}']:\n---\n{file_content.strip()}\n---"
                        logger.info(
                            f"Chat {chat_id}: Appended content of text document '{doc_name}'."
                        )
                    else:
                        appended_text += f"\n\n[INFO: Could not extract text from the attached document '{doc_name}'.]"

    return has_image, has_voice, appended_text, media_parts
# --- END OF UTILITY FUNCTIONS ---

# --- Status Update Handler for Forum Topics (to populate cache) ---
async def handle_forum_topic_updates(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not update.message or not update.effective_chat:
        logger.debug("Forum topic update lacks message or effective_chat. Skipping.")
        return

    # Ignore old status updates if feature is enabled
    if IGNORE_OLD_MESSAGES_ON_STARTUP and BOT_STARTUP_TIMESTAMP:
        message_date_utc = (
            update.message.date
        )  # Status update messages also have a date
        if message_date_utc.tzinfo is None:
            message_date_utc = message_date_utc.replace(tzinfo=dt_timezone.utc)
        else:
            message_date_utc = message_date_utc.astimezone(dt_timezone.utc)
        if message_date_utc < BOT_STARTUP_TIMESTAMP:
            logger.info(
                f"Ignoring old forum topic update (MsgID: {update.message.message_id}) from before bot startup."
            )
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
            chat_topic_cache_keys_order[chat_id].remove(key)  # Move to end if re-edited
        chat_topic_cache_keys_order[chat_id].append(key)

        # Prune if cache for this chat_id exceeds limit
        if (
            len(chat_topic_cache_keys_order[chat_id])
            > MAX_TOPIC_ENTRIES_PER_CHAT_IN_CACHE
        ):
            key_to_remove = chat_topic_cache_keys_order[chat_id].pop(0)  # Remove oldest
            if key_to_remove in topic_names_cache:
                removed_name = topic_names_cache.pop(key_to_remove)
                logger.info(
                    f"Pruned oldest topic '{removed_name}' (Key: {key_to_remove}) from cache for chat {chat_id} due to size limit."
                )

        if action_taken == "created":
            logger.info(
                f"Topic '{topic_name_to_cache}' (ID: {thread_id}) {action_taken} in chat {chat_id}. Cached."
            )
        elif action_taken == "edited":  # Log edit only if name changed or for debug
            if old_name != topic_name_to_cache:
                logger.info(
                    f"Topic ID {thread_id} in chat {chat_id} renamed from '{old_name}' to '{topic_name_to_cache}'. Cache updated."
                )
            else:
                logger.debug(
                    f"Topic ID {thread_id} in chat {chat_id} edited but name ('{topic_name_to_cache}') unchanged. Cache refreshed."
                )


# --- COMMAND HANDLERS ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat or not update.effective_user:
        return

    # Check if the command is specifically for this bot in group chats
    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(" ", 1)
        command_with_potential_mention = command_parts[0]
        if (
            "@" in command_with_potential_mention
            and f"@{context.bot.username}" not in command_with_potential_mention
        ):
            logger.info(
                f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring."
            )
            return

    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        await handle_permission_denied(update, context)
        return

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
            context.bot,
            update.effective_chat.id,
            start_message,
            preferred_thread_id=message_thread_id,
        )
    except Exception as e:
        logger.error(
            f"Failed to send start message to {update.effective_chat.id} (thread {message_thread_id}): {e}"
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat or not update.effective_user:
        return

    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(" ", 1)
        command_with_potential_mention = command_parts[0]
        if (
            "@" in command_with_potential_mention
            and f"@{context.bot.username}" not in command_with_potential_mention
        ):
            logger.info(
                f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring."
            )
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
        "/deactivate - (Groups/Topics only) Stop me from responding to every message (revert to mentions/replies/free will).",
        "/auth_shapes - Connect your Shapes.inc account for personalized memory and recognition of Shapes.inc username and persona.",
        "/cancel - Stop a multi-step process like authentication.",
    ]
    if BING_IMAGE_CREATOR_AVAILABLE and BING_AUTH_COOKIE:
        help_text_parts.append(
            "/imagine <prompt> - Generate images based on your prompt using Bing."
        )
        if ALLOWED_USERS and str(update.effective_user.id) in ALLOWED_USERS:
            help_text_parts.append(
                "/setbingcookie <cookie_value> - (Admin) Update the Bing authentication cookie."
            )

    help_text_parts.append(
        "\nSimply send me a message, an image (with or without a caption), or a voice message to start chatting!"
    )
    if ENABLE_TOOL_USE and ACTIVE_TOOL_DEFINITIONS:
        help_text_parts.append("\nI can also use tools like:")
        for tool_def in ACTIVE_TOOL_DEFINITIONS:
            if tool_def["type"] == "function":
                func_info = tool_def["function"]
                desc_first_sentence = func_info["description"].split(".")[0] + "."
                help_text_parts.append(
                    f"  - `{func_info['name']}`: {desc_first_sentence}"
                )

    if (
        GROUP_FREE_WILL_ENABLED
        and update.effective_chat
        and update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]
    ):
        # Adjusted comment for thread awareness
        help_text_parts.append(
            f"\nGroup Free Will is enabled! I might respond randomly about {GROUP_FREE_WILL_PROBABILITY:.1%} of the time, considering the last ~{GROUP_FREE_WILL_CONTEXT_MESSAGES} messages in this specific topic/chat."
        )

    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        help_text_parts.append(
            "\n\nNote: Your access to interact with me is currently restricted."
        )

    escaped_help_text = escape_markdown("\n".join(help_text_parts), version=2)
    try:
        # Use helper to send, respecting thread ID
        await send_message_to_chat_or_general(
            context.bot,
            update.effective_chat.id,
            escaped_help_text,
            preferred_thread_id=message_thread_id,
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except Exception as e:
        logger.error(
            f"Failed to send help message to {update.effective_chat.id} (thread {message_thread_id}): {e}"
        )
        # Fallback to plain text if MDv2 fails for reasons other than thread not found (handled by helper)
        try:
            # Use helper for plain text fallback as well
            await send_message_to_chat_or_general(
                context.bot,
                update.effective_chat.id,
                "\n".join(help_text_parts),  # Send unescaped for plain
                preferred_thread_id=message_thread_id,
            )
        except Exception as e2:
            logger.error(
                f"Failed to send plain text help fallback to {update.effective_chat.id} (thread {message_thread_id}): {e2}"
            )


async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat or not update.effective_user:
        return

    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(" ", 1)
        command_with_potential_mention = command_parts[0]
        if (
            "@" in command_with_potential_mention
            and f"@{context.bot.username}" not in command_with_potential_mention
        ):
            logger.info(
                f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring."
            )
            return

    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        await handle_permission_denied(update, context)
        return

    chat_id = update.effective_chat.id

    raw_command_thread_id = update.message.message_thread_id
    effective_topic_id_for_command: Optional[int] = raw_command_thread_id

    if update.effective_chat.is_forum and SEPARATE_TOPIC_HISTORIES:
        # For /newchat, the "context" it clears should match the message's apparent location.
        # If it's a reply to a "classic General" message, clear "General" history.
        if (
            update.message.reply_to_message
            and update.message.reply_to_message.message_thread_id is None
        ):
            effective_topic_id_for_command = None
            logger.debug(
                f"/newchat: Command replied to classic General. Effective Topic ID: None."
            )
        # If the command itself is in what appears to be "General" (even with numeric ID), clear "General".
        elif (
            not update.message.is_topic_message
        ):  # is_topic_message is False for General
            effective_topic_id_for_command = None
            logger.debug(
                f"/newchat: Command in General (is_topic_message=False, raw_thread_id={raw_command_thread_id}). Effective Topic ID: None."
            )
        # Otherwise, it's a specific topic. effective_topic_id_for_command remains raw_command_thread_id.
        else:
            logger.debug(
                f"/newchat: Command in specific topic. Effective Topic ID: {effective_topic_id_for_command}."
            )
    elif not (update.effective_chat.is_forum and SEPARATE_TOPIC_HISTORIES):
        # Not a forum or not separating, all context is 'general'
        effective_topic_id_for_command = None
        logger.debug(
            f"/newchat: Not a forum or not separating. Effective Topic ID: None."
        )

    history_key = (chat_id, effective_topic_id_for_command)
    topic_desc = (
        f"topic ID {effective_topic_id_for_command}"
        if effective_topic_id_for_command is not None
        else "general chat"
    )

    cleared_any = False

    # Clear LLM history for this specific thread/chat
    if history_key in chat_histories and chat_histories[history_key]:
        chat_histories[history_key] = []
        db_manager.delete_history(chat_id, effective_topic_id_for_command)
        logger.info(
            f"LLM Conversation history cleared for chat ID: {chat_id} ({topic_desc})"
        )
        cleared_any = True

    # Also clear raw log if it exists for the chat and thread
    if chat_id in group_raw_message_log:
        if effective_topic_id_for_command in group_raw_message_log.get(chat_id, {}):
            if group_raw_message_log[chat_id][
                effective_topic_id_for_command
            ]:  # Check if the thread log is non-empty
                group_raw_message_log[chat_id][effective_topic_id_for_command] = []
                logger.info(
                    f"Raw group message log cleared for chat ID: {chat_id} ({topic_desc})"
                )
                cleared_any = True

    # Use topic-aware message
    reply_text = (
        "✨ Conversation history for this topic/chat cleared! Let's start a new topic."
        if cleared_any
        else "There's no conversation history for this topic/chat to clear yet."
    )
    # update.message.reply_text automatically handles the thread_id
    await update.message.reply_text(reply_text)


async def imagine_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_chat or not update.effective_user:
        return

    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(" ", 1)
        command_with_potential_mention = command_parts[0]
        if (
            "@" in command_with_potential_mention
            and f"@{context.bot.username}" not in command_with_potential_mention
        ):
            logger.info(
                f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring."
            )
            return

    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        await handle_permission_denied(update, context)
        return

    # reply_text handles thread_id automatically
    if not (BING_IMAGE_CREATOR_AVAILABLE and ImageGen and BING_AUTH_COOKIE):
        await update.message.reply_text(
            "The /imagine command is currently unavailable or not configured. Please contact an admin."
        )
        return
    if not context.args:
        await update.message.reply_text(
            "Please provide a prompt for the image. Usage: /imagine <your image prompt>"
        )
        return

    prompt = " ".join(context.args)
    chat_id = update.effective_chat.id
    message_thread_id: Optional[int] = None
    if update.message.message_thread_id is not None:
        message_thread_id = update.message.message_thread_id

    typing_task: Optional[asyncio.Task] = None
    status_msg: Optional[TelegramMessage] = None
    temp_dir = (
        f"temp_bing_images_{chat_id}_{random.randint(1000,9999)}"  # Unique temp dir
    )

    try:
        # Send initial action respecting thread_id
        await context.bot.send_chat_action(
            chat_id=chat_id,
            action=ChatAction.UPLOAD_PHOTO,
            message_thread_id=message_thread_id,
        )
        typing_task = asyncio.create_task(
            _keep_typing_loop(
                context,
                chat_id,
                message_thread_id,
                action=ChatAction.UPLOAD_PHOTO,
                interval=5.0,
            )
        )
        # Reply handles thread_id automatically
        status_msg = await update.message.reply_text(
            f'🎨 Working on your vision: "{prompt[:50]}..." (using Bing)'
        )

        image_gen = ImageGen(
            auth_cookie=BING_AUTH_COOKIE
        )  # Uses the global BING_AUTH_COOKIE
        image_links = await asyncio.to_thread(image_gen.get_images, prompt)

        if not image_links:
            if status_msg:
                await status_msg.edit_text(
                    "Sorry, Bing couldn't generate images for that prompt, or no images were returned. Try rephrasing!"
                )
            return

        os.makedirs(temp_dir, exist_ok=True)
        # Download images to the temporary directory
        await asyncio.to_thread(
            image_gen.save_images,
            image_links,
            temp_dir,
            download_count=len(image_links),
        )

        media_photos: List[InputMediaPhoto] = []
        image_files = [
            f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))
        ]
        image_files.sort()  # Ensure some order if multiple images

        for filename in image_files:
            if len(media_photos) >= 10:
                break  # Telegram media group limit
            image_path = os.path.join(temp_dir, filename)
            try:
                with open(image_path, "rb") as img_f:
                    img_bytes = img_f.read()
                    media_photos.append(InputMediaPhoto(media=img_bytes))
            except Exception as e_img:
                logger.error(
                    f"Error processing image file {filename} for media group: {e_img}"
                )

        if typing_task:
            typing_task.cancel()
            typing_task = None  # Cancel before sending media

        if media_photos:
            # send_media_group needs explicit thread_id
            await context.bot.send_media_group(
                chat_id=chat_id, media=media_photos, message_thread_id=message_thread_id
            )
            if status_msg:
                await status_msg.delete()  # Clean up status message
        else:
            err_msg_no_proc = "Sorry, no images could be processed or sent from Bing."
            if status_msg:
                await status_msg.edit_text(err_msg_no_proc)
            else:
                await update.message.reply_text(err_msg_no_proc)  # Fallback reply

    except Exception as e:
        logger.error(
            f"Error during /imagine command for prompt '{prompt}': {e}", exc_info=True
        )
        err_text = "An error occurred while generating images with Bing. Please try again later."
        try:
            if status_msg:
                await status_msg.edit_text(err_text)
            # Use helper for direct send fallback
            else:
                await send_message_to_chat_or_general(
                    context.bot,
                    chat_id,
                    err_text,
                    preferred_thread_id=message_thread_id,
                )
        except Exception:  # Ultimate fallback if editing/sending fails
            await send_message_to_chat_or_general(
                context.bot, chat_id, err_text, preferred_thread_id=message_thread_id
            )
    finally:
        if typing_task and not typing_task.done():
            typing_task.cancel()
        if os.path.exists(temp_dir):  # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e_clean:
                logger.error(
                    f"Error cleaning up temporary directory {temp_dir}: {e_clean}"
                )


async def set_bing_cookie_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_user:
        return

    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(" ", 1)
        command_with_potential_mention = command_parts[0]
        if (
            "@" in command_with_potential_mention
            and f"@{context.bot.username}" not in command_with_potential_mention
        ):
            logger.info(
                f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring."
            )
            return

    user_id_str = str(update.effective_user.id)
    # Check admin and feature availability
    if not (
        BOT_OWNERS and user_id_str in BOT_OWNERS and BING_IMAGE_CREATOR_AVAILABLE
    ):
        await update.message.reply_text(
            "This command is restricted to the bot owner or is currently unavailable."
        )
        return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("Usage: /setbingcookie <new_cookie_value>")
        return

    new_cookie = context.args[0]
    global BING_AUTH_COOKIE
    BING_AUTH_COOKIE = new_cookie
    logger.info(f"BING_AUTH_COOKIE updated by admin: {user_id_str}")
    # Reply handles thread ID
    await update.message.reply_text(
        "Bing authentication cookie has been updated for the /imagine command."
    )


async def activate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat or not update.effective_user:
        return

    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(" ", 1)
        command_with_potential_mention = command_parts[0]
        if (
            "@" in command_with_potential_mention
            and f"@{context.bot.username}" not in command_with_potential_mention
        ):
            logger.info(
                f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring."
            )
            return

    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        await handle_permission_denied(update, context)
        return

    if update.effective_chat.type not in [Chat.GROUP, Chat.SUPERGROUP]:
        await update.message.reply_text(
            "The /activate command can only be used in groups or supergroups."
        )
        return
    
    # --- Check if the user is an admin in the group ---
    try:
        member = await context.bot.get_chat_member(update.effective_chat.id, update.effective_user.id)
        if member.status not in ["creator", "administrator"]:
            await update.message.reply_text("🚫 Sorry, only group administrators can use this command.")
            return
    except telegram.error.BadRequest as e:
        logger.error(f"Error checking admin status for user {update.effective_user.id} in chat {update.effective_chat.id}: {e}")
        await update.message.reply_text("Could not verify your admin status. Please try again.")
        return

    chat_id = update.effective_chat.id
    message_thread_id: Optional[
        int
    ] = update.message.message_thread_id  # Can be None for general group chat

    chat_topic_key = (chat_id, message_thread_id)
    topic_desc = (
        f"this topic (ID: {message_thread_id})"
        if message_thread_id is not None
        else "this group's general chat"
    )

    if chat_topic_key in activated_chats_topics:
        reply_text = f"I am already actively listening in {topic_desc}."
    else:
        activated_chats_topics.add(chat_topic_key)
        db_manager.activate_topic(chat_id, message_thread_id)
        reply_text = f"✅ Activated! I will now respond to all messages in {topic_desc}."
        logger.info(
            f"Bot activated for chat {chat_id} ({topic_desc}) by user {update.effective_user.id}"
        )

    await update.message.reply_text(reply_text)


async def deactivate_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not update.message or not update.effective_chat or not update.effective_user:
        return

    if update.effective_chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        command_parts = update.message.text.split(" ", 1)
        command_with_potential_mention = command_parts[0]
        if (
            "@" in command_with_potential_mention
            and f"@{context.bot.username}" not in command_with_potential_mention
        ):
            logger.info(
                f"Command '{command_with_potential_mention}' in chat {update.effective_chat.id} (thread {update.message.message_thread_id}) is for another bot. Ignoring."
            )
            return

    if not is_user_or_chat_allowed(update.effective_user.id, update.effective_chat.id):
        await handle_permission_denied(update, context)
        return

    if update.effective_chat.type not in [Chat.GROUP, Chat.SUPERGROUP]:
        await update.message.reply_text(
            "The /deactivate command can only be used in groups or supergroups."
        )
        return
    
    # --- Check if the user is an admin in the group ---
    try:
        member = await context.bot.get_chat_member(update.effective_chat.id, update.effective_user.id)
        if member.status not in ["creator", "administrator"]:
            await update.message.reply_text("🚫 Sorry, only group administrators can use this command.")
            return
    except telegram.error.BadRequest as e:
        logger.error(f"Error checking admin status for user {update.effective_user.id} in chat {update.effective_chat.id}: {e}")
        await update.message.reply_text("Could not verify your admin status. Please try again.")
        return

    chat_id = update.effective_chat.id
    message_thread_id: Optional[int] = update.message.message_thread_id

    chat_topic_key = (chat_id, message_thread_id)
    topic_desc = (
        f"this topic (ID: {message_thread_id})"
        if message_thread_id is not None
        else "this group's general chat"
    )

    if chat_topic_key in activated_chats_topics:
        activated_chats_topics.remove(chat_topic_key)
        db_manager.deactivate_topic(chat_id, message_thread_id)
        reply_text = f"💤 Deactivated. I will no longer respond to all messages in {topic_desc}. (I'll still respond to mentions, replies, or free will)."
        logger.info(
            f"Bot deactivated for chat {chat_id} ({topic_desc}) by user {update.effective_user.id}"
        )
    else:
        reply_text = (
            f"I was not actively listening to all messages in {topic_desc} anyway."
        )

    await update.message.reply_text(reply_text)


# --- AUTHENTICATION FUNCTIONS USING ConversationHandler ---
async def auth_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the auth conversation. Sends the link and asks for the code."""
    if not update.message or not update.effective_user:
        return ConversationHandler.END

    if not SHAPESINC_APP_ID:
        await update.message.reply_text("The authentication feature is not configured by the bot admin (missing App ID).")
        return ConversationHandler.END # End the conversation

    # This part is the same as your old command
    auth_url = f"https://shapes.inc/authorize?app_id={SHAPESINC_APP_ID}"
    reply_text = (
        "*🔐 AUTHENTICATION PROCESS STARTED*\n"
        "------------------------------------\n"
        "*Step 1: Get Your Code* ➡️\n"
        "Click the link below. A new page will open and give you a one-time code.\n\n"
        f"[🔗 Authorize on Shapes.inc]({auth_url})\n"
        "------------------------------------\n"
        "*Step 2: Send the Code Here* 🔢\n"
        "Once you have the code, *paste it directly into this chat and press send*.\n\n"
        "_I am now waiting only for your code._\n"
        "_Type_ /cancel _to abort._"
    )

    await send_message_to_chat_or_general(
        context.bot,
        update.effective_chat.id,
        escape_markdown(reply_text, version=2),
        preferred_thread_id=update.message.message_thread_id,
        parse_mode=ParseMode.MARKDOWN_V2,
        disable_web_page_preview=True,
    )
    
    # Tell the ConversationHandler that we are now waiting for the user's code
    return AWAITING_CODE


async def auth_receive_code(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives the code, exchanges it for a token, and ends the conversation."""
    if not update.message or not update.effective_user or not update.message.text:
        await update.message.reply_text("Something went wrong. Please try starting with /auth_shapes again.")
        return ConversationHandler.END

    user_id = update.effective_user.id
    one_time_code = update.message.text.strip()
    status_msg = await update.message.reply_text("Verifying your one-time code...")

    try:
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
            response = await client.post(
                f"{SHAPES_AUTH_BASE_URL}/nonce",
                json={"app_id": SHAPESINC_APP_ID, "code": one_time_code},
            )
            response.raise_for_status()

            data = response.json()
            auth_token = data.get("auth_token")

            if not auth_token:
                await status_msg.edit_text("Authentication failed. The API did not return an auth token. Please try again or type /cancel.")
                return AWAITING_CODE # Stay in the same state to allow another try

            # Store the token
            user_auth_tokens[user_id] = auth_token
            db_manager.save_user_token(user_id, auth_token)
            
            logger.info(f"Successfully obtained and stored Shapes.inc auth token for user {user_id}.")
            await status_msg.edit_text("✅ Authentication successful! I will now recognize you by your Shapes.inc account.")

    except httpx.HTTPStatusError as e:
        await status_msg.edit_text(f"Authentication failed (Code: {e.response.status_code}). Please check your code and send it again, or type /cancel.")
        return AWAITING_CODE # Stay in the same state
    except Exception as e:
        logger.error(f"Unexpected error during Shapes.inc authentication for user {user_id}: {e}", exc_info=True)
        await status_msg.edit_text("An unexpected error occurred. Please try again later by starting with /auth_shapes.")

    # The conversation is over
    return ConversationHandler.END


async def auth_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels the authentication process."""
    if update.message:
      await update.message.reply_text("Authentication process canceled.")
    return ConversationHandler.END
# --- END OF COMMAND HANDLERS ---

# --- Main Message Handler ---
async def process_message_entrypoint(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not update.message or not update.effective_chat or not update.effective_user:
        logger.debug("Update missing essential message, chat, or user. Ignoring.")
        return

    if IGNORE_OLD_MESSAGES_ON_STARTUP and BOT_STARTUP_TIMESTAMP:
        message_date_utc = update.message.date.astimezone(dt_timezone.utc)
        if message_date_utc < BOT_STARTUP_TIMESTAMP:
            logger.info(
                f"Ignoring old message (ID: {update.message.message_id}) from before bot startup."
            )
            return

    chat_id = update.effective_chat.id
    current_user = update.effective_user
    chat_type = update.effective_chat.type

    raw_current_message_thread_id: Optional[int] = update.message.message_thread_id

    # --- Determine effective_topic_context_id (for HISTORY, LOGS, LOCKING, API CHANNEL) ---
    # This ID defines the conversational context.
    # None represents the "General" topic in forums or non-forum/non-separated contexts.
    effective_topic_context_id: Optional[int] = raw_current_message_thread_id

    if update.effective_chat.is_forum and SEPARATE_TOPIC_HISTORIES:
        is_current_msg_topic_msg = update.message.is_topic_message # True if in a specific user-created topic
        is_reply_to_classic_general = (
            update.message.reply_to_message
            and update.message.reply_to_message.message_thread_id is None # "Classic General" has no thread_id on replies to it
        )
        if is_reply_to_classic_general:
            # If replying to a message in the "classic General" (where its thread_id is None),
            # the context is General.
            effective_topic_context_id = None
            # logger.debug(f"HISTORY CTX ID (Forum, Separated): Reply to classic General. Effective ID: None. (Raw: {raw_current_message_thread_id})")
        elif not is_current_msg_topic_msg:
            # If the current message itself is in the "General" topic (is_topic_message is False for General,
            # regardless of its raw_current_message_thread_id), the context is General.
            effective_topic_context_id = None
            # logger.debug(f"HISTORY CTX ID (Forum, Separated): Current msg in General. Effective ID: None. (Raw: {raw_current_message_thread_id})")
        # Else, it's a message within a specific user-created topic.
        # effective_topic_context_id remains raw_current_message_thread_id.
        # logger.debug(f"HISTORY CTX ID (Forum, Separated): Current msg in specific topic. Effective ID: {effective_topic_context_id}.")

    elif not (update.effective_chat.is_forum and SEPARATE_TOPIC_HISTORIES):
        # Not a forum or SEPARATE_TOPIC_HISTORIES is False. All context is "general" for this chat.
        effective_topic_context_id = None
        # logger.debug(f"HISTORY CTX ID (Not Forum or Not Separated): Effective ID: None. (Raw: {raw_current_message_thread_id})")

    logger.debug(
        f"Derived HISTORY/CONTEXT ID: {effective_topic_context_id} "
        f"(Raw incoming thread_id: {raw_current_message_thread_id}, is_forum: {update.effective_chat.is_forum}, separate_hist: {SEPARATE_TOPIC_HISTORIES})"
    )

    # --- Determine effective_send_thread_id for SENDING REPLIES ---
    effective_send_thread_id: Optional[int] = raw_current_message_thread_id

    if update.message.reply_to_message:
        effective_send_thread_id = update.message.reply_to_message.message_thread_id
        # logger.debug(f"SEND THREAD ID: Message is a reply. Using replied message's thread_id: {effective_send_thread_id}.")

    if update.effective_chat.is_forum:
        if not update.message.is_topic_message:  # Current message is in General
            # if effective_send_thread_id is not None:
            # logger.debug(f"SEND THREAD ID (Forum General Correction): Current message in General. Overriding send thread to None. (Original: {effective_send_thread_id})")
            effective_send_thread_id = None
        elif (
            update.message.reply_to_message
            and update.message.reply_to_message.message_thread_id is None
        ):  # Replying to classic General
            # if effective_send_thread_id is not None:
            # logger.debug(f"SEND THREAD ID (Forum General Correction): Replying to classic General. Overriding send thread to None. (Original: {effective_send_thread_id})")
            effective_send_thread_id = None

    logger.info(
        f"Final IDs - History/Context ID: {effective_topic_context_id}, Send Reply Thread ID: {effective_send_thread_id}"
    )

    # --- Lock Key based on effective_topic_context_id ---
    lock_key: Union[int, Tuple[int, Optional[int]]]
    if update.effective_chat.is_forum and SEPARATE_TOPIC_HISTORIES:
        lock_key = (chat_id, effective_topic_context_id)
    else:
        lock_key = chat_id

    if lock_key not in processing_locks:
        processing_locks[lock_key] = asyncio.Lock()
    lock = processing_locks[lock_key]

    logger.info(
        f"Message received for context key: {lock_key} (Raw TG thread_id: {raw_current_message_thread_id}). Waiting for lock..."
    )
    async with lock:
        logger.info(f"Lock acquired for context key: {lock_key}. Processing message...")
        try:
            user_message_text_original = update.message.text or ""
            user_message_caption_original = update.message.caption or ""
            replied_msg = update.message.reply_to_message

            log_media_description = ""
            if update.message.photo:
                log_media_description = "[Image]"
            elif update.message.sticker:
                log_media_description = "[Sticker]"
            elif update.message.voice:
                log_media_description = "[Voice Message]"
            elif doc_ := update.message.document:
                doc_name_ = doc_.file_name or "file"
                if doc_.mime_type and doc_.mime_type.startswith("image/"):
                    log_media_description = f"[Image File: {doc_name_}]"
                elif doc_.mime_type and doc_.mime_type.startswith("audio/"):
                    log_media_description = f"[Audio File: {doc_name_}]"
                else:
                    log_media_description = f"[Document: {doc_name_}]"
            elif not (user_message_text_original or user_message_caption_original):
                log_media_description = "[Unsupported or Empty Message Type]"

            current_message_content_for_raw_log = (
                user_message_text_original
                or user_message_caption_original
                or log_media_description
            )
            if chat_type in [Chat.GROUP, Chat.SUPERGROUP]:
                add_to_raw_group_log(
                    chat_id,
                    effective_topic_context_id,
                    get_display_name(current_user),
                    current_message_content_for_raw_log,
                )

            if not is_user_or_chat_allowed(current_user.id, chat_id):
                return

            (
                should_process_message,
                is_direct_reply_to_bot,
                is_mention_to_bot,
                is_free_will_triggered,
                is_activated_chat_topic,
            ) = (False, False, False, False, False)
            bot_username_at = f"@{context.bot.username}"
            text_for_trigger_check = (
                user_message_text_original or user_message_caption_original
            )
            current_chat_topic_key_for_activation = (
                chat_id,
                raw_current_message_thread_id,
            )

            if chat_type == Chat.PRIVATE:
                should_process_message = True
            elif chat_type in [Chat.GROUP, Chat.SUPERGROUP]:
                if current_chat_topic_key_for_activation in activated_chats_topics:
                    should_process_message, is_activated_chat_topic = True, True
                elif (
                    replied_msg
                    and replied_msg.from_user
                    and replied_msg.from_user.id == context.bot.id
                ):
                    should_process_message, is_direct_reply_to_bot = True, True
                elif bot_username_at in text_for_trigger_check:
                    should_process_message, is_mention_to_bot = True, True
                elif (
                    GROUP_FREE_WILL_ENABLED
                    and random.random() < GROUP_FREE_WILL_PROBABILITY
                ):
                    should_process_message, is_free_will_triggered = True, True

            if not should_process_message:
                logger.debug(
                    f"Message in group {chat_id} (context_id {effective_topic_context_id}, raw_thread_id {raw_current_message_thread_id}) not for bot. Ignoring."
                )
                return

            llm_history = get_llm_chat_history(chat_id, effective_topic_context_id)
            user_content_parts_for_llm: List[Dict[str, Any]] = []

            (
                has_image_from_current,
                has_voice_from_current,
                appended_file_content_str,
                current_media_parts,
            ) = await _process_media_and_documents(update.message)
            user_content_parts_for_llm.extend(current_media_parts)

            if replied_msg and not is_free_will_triggered:
                (
                    replied_has_image,
                    replied_has_voice,
                    replied_doc_text,
                    replied_media_parts,
                ) = await _process_media_and_documents(replied_msg)
                if not has_image_from_current and replied_has_image:
                    user_content_parts_for_llm.extend(
                        [p for p in replied_media_parts if p.get("type") == "image_url"]
                    )
                if not has_voice_from_current and replied_has_voice:
                    user_content_parts_for_llm.extend(
                        [p for p in replied_media_parts if p.get("type") == "audio_url"]
                    )
                if replied_doc_text:
                    appended_file_content_str += replied_doc_text

            base_text_for_llm = ""
            current_speaker_display_name = get_display_name(current_user)
            known_topic_name_for_context = (
                topic_names_cache.get((chat_id, effective_topic_context_id))
                if effective_topic_context_id is not None
                else None
            )

            if is_free_will_triggered:
                replied_to_info = None
                if replied_msg:
                    author = get_display_name(replied_msg.from_user)
                    content = replied_msg.text or replied_msg.caption or ""
                    if not content:
                        if replied_msg.photo or (
                            replied_msg.document
                            and replied_msg.document.mime_type
                            and replied_msg.document.mime_type.startswith("image/")
                        ):
                            content = "[Image]"
                        elif replied_msg.voice or (
                            replied_msg.document
                            and replied_msg.document.mime_type
                            and replied_msg.document.mime_type.startswith("audio/")
                        ):
                            content = "[Voice Message]"
                        elif replied_msg.document:
                            content = f"[Document: {replied_msg.document.file_name or 'file'}]"
                        else:
                            content = "[message with no text content]"
                    replied_to_info = {"author": author, "content": content}
                base_text_for_llm = format_freewill_context_from_raw_log(
                    chat_id,
                    effective_topic_context_id,
                    GROUP_FREE_WILL_CONTEXT_MESSAGES,
                    context.bot.username or SHAPESINC_SHAPE_USERNAME,
                    update.effective_chat.title,
                    known_topic_name_for_context,
                    replied_to_message_context=replied_to_info,
                )
            else:
                speaker_context_prefix, reply_context_prefix = "", ""
                location_addon = ""
                if chat_type in [Chat.GROUP, Chat.SUPERGROUP]:
                    parts = (
                        [f"in group '{update.effective_chat.title}'"]
                        if update.effective_chat.title
                        else []
                    )
                    if effective_topic_context_id is not None:
                        parts.append(
                            f"with topic name '{known_topic_name_for_context}'"
                            if known_topic_name_for_context
                            else f"with topic ID '{effective_topic_context_id}'"
                        )
                    elif update.effective_chat.title:
                        parts.append("the general chat area")
                    location_addon = f" {' '.join(parts)}" if parts else ""
                elif chat_type == Chat.PRIVATE:
                    location_addon = " in a private chat"
                speaker_context_prefix = f"[Person '{current_speaker_display_name}' (ID: {current_user.id}) on Telegram{location_addon} says:]\n"

                if replied_msg:
                    generate_reply_context = False
                    service_attrs = [
                        "forum_topic_created",
                        "forum_topic_reopened",
                        "forum_topic_edited",
                        "forum_topic_closed",
                        "general_forum_topic_hidden",
                        "general_forum_topic_unhidden",
                        "write_access_allowed",
                        "group_chat_created",
                        "supergroup_chat_created",
                        "message_auto_delete_timer_changed",
                        "migrate_to_chat_id",
                        "migrate_from_chat_id",
                        "pinned_message",
                        "new_chat_members",
                        "left_chat_member",
                        "new_chat_title",
                        "new_chat_photo",
                        "delete_chat_photo",
                        "video_chat_scheduled",
                        "video_chat_started",
                        "video_chat_ended",
                        "video_chat_participants_invited",
                        "web_app_data",
                    ]
                    is_ignorable_reply_target = any(
                        getattr(replied_msg, attr, False) for attr in service_attrs
                    )
                    if (
                        replied_msg.from_user
                        and replied_msg.from_user.id == context.bot.id
                    ):
                        generate_reply_context = True
                    elif not is_ignorable_reply_target and (
                        is_mention_to_bot or is_activated_chat_topic
                    ):
                        generate_reply_context = True
                    if generate_reply_context:
                        author = get_display_name(replied_msg.from_user)
                        desc = replied_msg.text or replied_msg.caption or ""
                        if not desc:
                            if replied_msg.photo:
                                desc = "[Image]"
                            elif replied_msg.voice:
                                desc = "[Voice Message]"
                            elif replied_msg.document:
                                doc_name = replied_msg.document.file_name or "file"
                                if (
                                    replied_msg.document.mime_type
                                    and replied_msg.document.mime_type.startswith(
                                        "image/"
                                    )
                                ):
                                    desc = f"[Image File: {doc_name}]"
                                else:
                                    desc = f"[Document: {doc_name}]"
                            else:
                                desc = "[non-text/media message]"
                        author_id_desc = (
                            f"'{author}'"
                            if not (
                                replied_msg.from_user
                                and replied_msg.from_user.id == context.bot.id
                            )
                            else "your previous message"
                        )
                        reply_context_prefix = f'[Replying to {author_id_desc} which said: "{desc[:4096]}"]\n'
                    elif is_ignorable_reply_target and (
                        is_mention_to_bot or is_activated_chat_topic
                    ):
                        logger.info(
                            f"Chat {chat_id}: Bot addressed in reply to ignorable service message. Suppressing reply context."
                        )

                actual_user_text = (
                    user_message_text_original or user_message_caption_original
                )
                if is_mention_to_bot and bot_username_at in actual_user_text:
                    actual_user_text = re.sub(
                        r"\s*" + re.escape(bot_username_at) + r"\s*",
                        " ",
                        actual_user_text,
                        count=1,
                        flags=re.IGNORECASE,
                    ).strip()
                    if not actual_user_text and (
                        user_message_text_original or user_message_caption_original
                    ):
                        actual_user_text = "(You were addressed directly)"
                base_text_for_llm = (
                    speaker_context_prefix + reply_context_prefix + actual_user_text
                )
                if (
                    not actual_user_text
                    and not has_image_from_current
                    and not has_voice_from_current
                ):
                    base_text_for_llm += "(This was a reply with no new text/media)"

            full_text_for_llm = (
                base_text_for_llm.strip() + appended_file_content_str
            ).strip()
            if full_text_for_llm:
                user_content_parts_for_llm.insert(
                    0, {"type": "text", "text": full_text_for_llm}
                )
            elif not any(p.get("type") == "text" for p in user_content_parts_for_llm):
                user_content_parts_for_llm.insert(
                    0, {"type": "text", "text": "(Regarding the attached media)"}
                )

            if not user_content_parts_for_llm or not any(
                isinstance(p, dict)
                and p.get("type") == "text"
                and p.get("text", "").strip()
                for p in user_content_parts_for_llm
            ):
                logger.warning(f"Chat {chat_id}: No valid content for LLM. Skipping.")
                if not is_free_will_triggered:
                    await update.message.reply_text(
                        "I'm not sure how to respond to that.",
                        message_thread_id=effective_send_thread_id,
                    )
                return

            final_llm_content: Union[str, List[Dict[str, Any]]]
            if (
                len(user_content_parts_for_llm) == 1
                and user_content_parts_for_llm[0].get("type") == "text"
            ):
                final_llm_content = user_content_parts_for_llm[0]["text"]
            else:
                final_llm_content = [
                    p
                    for p in user_content_parts_for_llm
                    if isinstance(p, dict)
                    and not (p.get("type") == "text" and not p.get("text", "").strip())
                ]

            if not final_llm_content or (
                isinstance(final_llm_content, str) and not final_llm_content.strip()
            ):
                logger.warning(f"Chat {chat_id}: Final LLM content empty. Not sending.")
                if not is_free_will_triggered:
                    await update.message.reply_text(
                        "I didn't get any content to process.",
                        message_thread_id=effective_send_thread_id,
                    )
                return

            llm_history.append({"role": "user", "content": final_llm_content})
            db_manager.save_history(chat_id, effective_topic_context_id, llm_history)

            log_content_summary = ""
            if isinstance(final_llm_content, str):
                log_content_summary = f"Content (string): '{final_llm_content[:150].replace(chr(10), '/N')}...'"
            elif isinstance(final_llm_content, list):
                part_summaries = []
                for p_idx, p_content in enumerate(final_llm_content):
                    p_type = (
                        p_content.get("type", "unknown")
                        if isinstance(p_content, dict)
                        else "unknown_item_type"
                    )
                    p_summary_text = (
                        p_content.get("text", "")[:80]
                        if p_type == "text"
                        else p_content.get("audio_url", {}).get(
                            "url",
                            "[Base64 Image Data]"
                            if p_type == "image_url"
                            else str(p_content),
                        )[:80]
                    )
                    part_summaries.append(
                        f"{p_type.capitalize()}[{p_idx}]: {p_summary_text.replace(chr(10),'/N')}..."
                    )
                log_content_summary = f"Content (multi-part): {part_summaries}"
            logger.info(
                f"Chat {chat_id} (context_id {effective_topic_context_id}, send_thread_id {effective_send_thread_id}, raw_thread_id {raw_current_message_thread_id}): Appended user message. {log_content_summary}. "
                f"Trigger: reply={is_direct_reply_to_bot}, mention={is_mention_to_bot}, free_will={is_free_will_triggered}, DM={chat_type==Chat.PRIVATE}, activated={is_activated_chat_topic}"
            )

            final_text_from_llm_before_media_extraction = ""
            text_part_after_media_extraction = ""
            image_urls_to_send: List[str] = []
            audio_urls_to_send: List[str] = []
            MAX_EMPTY_RETRIES, empty_retry_count, is_response_valid = 3, 0, False

            typing_task: Optional[asyncio.Task] = None
            try:
                typing_task = asyncio.create_task(
                    _keep_typing_loop(
                        context,
                        chat_id,
                        effective_send_thread_id,
                        action=ChatAction.TYPING,
                    )
                )

                while empty_retry_count < MAX_EMPTY_RETRIES:
                    history_before_this_attempt = list(llm_history)
                    text_from_this_iteration = ""
                    ai_msg_obj: Optional[ChatCompletionMessage] = None
                    tool_status_msg: Optional[TelegramMessage] = None
                    MAX_TOOL_ITERATIONS, current_iteration = 5, 0

                    while current_iteration < MAX_TOOL_ITERATIONS:
                        current_iteration += 1

                        #messages_for_this_api_call = list(llm_history)
                        # Instead of sending the whole history, send ONLY the last message (shapes inc api is stateful and manages the history).
                        last_message_turn = [llm_history[-1]] if llm_history else []

                        api_params: Dict[str, Any] = {
                            "model": f"shapesinc/{SHAPESINC_SHAPE_USERNAME}",
                            "messages": last_message_turn,
                        }

                        if (
                            ENABLE_TOOL_USE
                            and ACTIVE_TOOL_DEFINITIONS
                            and not is_free_will_triggered
                        ):
                            last_msg_role = (
                                llm_history[-1].get("role") if llm_history else None
                            )
                            api_params["tools"] = ACTIVE_TOOL_DEFINITIONS
                            api_params["tool_choice"] = (
                                "none" if last_msg_role == "tool" else "auto"
                            )
                        elif is_free_will_triggered:
                            api_params.pop("tools", None)
                            api_params.pop("tool_choice", None)

                        channel_id_for_api: str
                        if update.effective_chat.is_forum and SEPARATE_TOPIC_HISTORIES:
                            if effective_topic_context_id is not None:
                                channel_id_for_api = (
                                    f"{chat_id}_{effective_topic_context_id}"
                                )
                            else:
                                channel_id_for_api = f"{chat_id}_general"
                        else:
                            channel_id_for_api = str(chat_id)


                        client_for_this_request = aclient_shape  # Default to the global, API-key based client
                        custom_headers_for_api = {}
                        # Check if the user has an auth token from the /auth_shapes command
                        if current_user.id in user_auth_tokens:
                            logger.info(f"User {current_user.id} has an auth token. Using user-authenticated client.")
                            client_for_this_request = AsyncOpenAI(
                                api_key="not-needed",
                                base_url=SHAPES_API_BASE_URL,
                                timeout=SHAPES_API_CLIENT_TIMEOUT,
                                default_headers={
                                    "X-App-ID": SHAPESINC_APP_ID,
                                    "X-User-Auth": user_auth_tokens[current_user.id],
                                    "X-Channel-Id": channel_id_for_api,
                                },
                            )
                        else:
                            logger.info(f"User {current_user.id} does not have an auth token. Using API key with custom headers.")
                            # Fallback to the original method if user is not authenticated
                            custom_headers_for_api = {
                                "X-User-Id": str(current_user.id),
                                "X-Channel-Id": channel_id_for_api,
                            }

                        logger.debug(
                            f"API HISTORY SCOPING: X-Channel-Id: {channel_id_for_api} (based on context_id: {effective_topic_context_id})"
                        )

                        logger.info(
                            f"API Call (Attempt {empty_retry_count+1}, Tool Iter {current_iteration}) for chat {chat_id} (context_id {effective_topic_context_id}). Tool choice: {api_params.get('tool_choice', 'N/A')}."
                        )
                        response_from_ai = await client_for_this_request.chat.completions.create(
                            **api_params, extra_headers=custom_headers_for_api
                        )
                        ai_msg_obj = response_from_ai.choices[0].message
                        llm_history.append(ai_msg_obj.model_dump(exclude_none=True))
                        db_manager.save_history(chat_id, effective_topic_context_id, llm_history)
                        logger.debug(
                            f"Chat {chat_id}: Appended assistant response. Last: {str(llm_history[-1])[:150].replace(chr(10),'/N')}..."
                        )

                        if ai_msg_obj.tool_calls:
                            if not ENABLE_TOOL_USE or is_free_will_triggered:
                                text_from_this_iteration = "I tried to use a tool, but it's disabled for this action."
                                logger.warning(
                                    f"Chat {chat_id}: AI tool use attempt when disabled. Calls: {ai_msg_obj.tool_calls}"
                                )
                                llm_history[-1] = {
                                    "role": "assistant",
                                    "content": text_from_this_iteration,
                                }
                                break
                            if not tool_status_msg:
                                tool_names = ", ".join(
                                    sorted(
                                        list(
                                            set(
                                                tc.function.name
                                                for tc in ai_msg_obj.tool_calls
                                                if tc.function and tc.function.name
                                            )
                                        )
                                    )
                                )
                                try:
                                    tool_status_msg = await send_message_to_chat_or_general(
                                        context.bot,
                                        chat_id,
                                        f"🛠️ Using tools: {tool_names}...",
                                        preferred_thread_id=effective_send_thread_id,
                                    )
                                except Exception as e_tsm:
                                    logger.warning(
                                        f"Chat {chat_id}: Failed to send tool status: {e_tsm}"
                                    )
                            tool_results: list[ChatCompletionMessageParam] = []
                            for tool_call in ai_msg_obj.tool_calls:
                                func_name, tool_id, args_str = (
                                    tool_call.function.name,
                                    tool_call.id,
                                    tool_call.function.arguments,
                                )
                                tool_output = f"Error: Tool '{func_name}' failed."
                                if func_name in ACTIVE_TOOLS_PYTHON_FUNCTIONS:
                                    try:
                                        py_func = ACTIVE_TOOLS_PYTHON_FUNCTIONS[
                                            func_name
                                        ]
                                        parsed_args = json.loads(args_str or "{}")
                                        tool_kwargs = parsed_args.copy()
                                        if func_name in [
                                            "create_poll_in_chat",
                                            "restrict_user_in_chat",
                                            "get_user_info",
                                            "generate_anime_image",
                                        ]:
                                            tool_kwargs.update(
                                                {
                                                    "telegram_bot_context": context,
                                                    "current_chat_id": chat_id,
                                                }
                                            )
                                        if func_name in [
                                            "create_poll_in_chat",
                                            "generate_anime_image",
                                        ]:
                                            tool_kwargs[
                                                "current_message_thread_id"
                                            ] = effective_send_thread_id
                                        output_val = (
                                            await py_func(**tool_kwargs)
                                            if asyncio.iscoroutinefunction(py_func)
                                            else await asyncio.to_thread(
                                                py_func, **tool_kwargs
                                            )
                                        )
                                        tool_output = str(output_val)
                                    except Exception as e_tool:
                                        tool_output = f"Error executing tool '{func_name}': {e_tool}"
                                        logger.error(
                                            f"Chat {chat_id}: {tool_output}",
                                            exc_info=True,
                                        )
                                else:
                                    tool_output = (
                                        f"Error: Tool '{func_name}' not available."
                                    )
                                tool_results.append(
                                    {
                                        "tool_call_id": tool_id,
                                        "role": "tool",
                                        "name": func_name,
                                        "content": tool_output,
                                    }
                                )
                            llm_history.extend(tool_results)
                            db_manager.save_history(chat_id, effective_topic_context_id, llm_history)
                        elif ai_msg_obj.content is not None:
                            text_from_this_iteration = str(ai_msg_obj.content)
                            logger.info(
                                f"Chat {chat_id}: AI final text (attempt {empty_retry_count+1}, iter {current_iteration}): '{text_from_this_iteration[:120].replace(chr(10),' ')}...'"
                            )
                            break
                        else:
                            logger.warning(
                                f"Chat {chat_id}: AI response (iter {current_iteration}) had no tool_calls and content was None."
                            )
                            text_from_this_iteration = (
                                "AI provided an empty response. Please try rephrasing."
                            )
                            llm_history[-1] = {
                                "role": "assistant",
                                "content": text_from_this_iteration,
                            }
                            break

                    if (
                        current_iteration >= MAX_TOOL_ITERATIONS
                        and not text_from_this_iteration
                    ):
                        logger.warning(
                            f"Chat {chat_id}: Max tool iterations ({MAX_TOOL_ITERATIONS}) reached."
                        )
                        text_from_this_iteration = "I tried my tools multiple times but couldn't get a final answer. Could you rephrase?"
                        llm_history.append(
                            {"role": "assistant", "content": text_from_this_iteration}
                        )
                    if tool_status_msg:
                        try:
                            await tool_status_msg.delete()
                        except Exception as e_del_tsm:
                            logger.warning(
                                f"Chat {chat_id}: Could not delete tool status: {e_del_tsm}"
                            )

                    final_text_from_llm_before_media_extraction = (
                        text_from_this_iteration
                    )
                    image_urls_to_send, audio_urls_to_send = [], []
                    text_part_after_media_extraction = (
                        final_text_from_llm_before_media_extraction
                    )
                    img_pattern = re.compile(
                        r"(https://files\.shapes\.inc/[\w.-]+\.(?:png|jpg|jpeg|gif|webp))\b",
                        re.I,
                    )
                    audio_pattern = re.compile(
                        r"(https://files\.shapes\.inc/[\w.-]+\.(?:mp3|ogg|wav|m4a|flac))\b",
                        re.I,
                    )
                    media_matches = []
                    for p, t in [(img_pattern, "image"), (audio_pattern, "audio")]:
                        for m in p.finditer(text_part_after_media_extraction):
                            media_matches.append({"match": m, "type": t})
                    media_matches.sort(key=lambda item: item["match"].start())
                    plain_segments, last_idx_media = [], 0
                    for item in media_matches:
                        match, url = item["match"], item["match"].group(0)
                        plain_segments.append(
                            text_part_after_media_extraction[
                                last_idx_media : match.start()
                            ]
                        )
                        if item["type"] == "image":
                            image_urls_to_send.append(url)
                        else:
                            audio_urls_to_send.append(url)
                        last_idx_media = match.end()
                    plain_segments.append(
                        text_part_after_media_extraction[last_idx_media:]
                    )
                    text_part_after_media_extraction = "".join(plain_segments)
                    text_part_after_media_extraction = re.sub(
                        r"\[([^\]]*)\]\(\s*\)", r"\1", text_part_after_media_extraction
                    )
                    text_part_after_media_extraction = re.sub(
                        r"(\r\n|\r|\n){2,}", "\n", text_part_after_media_extraction
                    ).strip()

                    if (
                        not text_part_after_media_extraction.strip()
                        and not image_urls_to_send
                        and not audio_urls_to_send
                    ):
                        empty_retry_count += 1
                        logger.warning(
                            f"Chat {chat_id} (context_id {effective_topic_context_id}): AI returned empty. Retry {empty_retry_count}/{MAX_EMPTY_RETRIES}."
                        )
                        llm_history = history_before_this_attempt
                        if empty_retry_count >= MAX_EMPTY_RETRIES:
                            logger.error(
                                f"Chat {chat_id}: All {MAX_EMPTY_RETRIES} retries empty. Aborting."
                            )
                        else:
                            await asyncio.sleep(0.5)
                            continue
                    else:
                        logger.info(
                            f"Chat {chat_id}: Valid AI response on attempt {empty_retry_count+1}."
                        )
                        is_response_valid = True
                        break

                if not is_response_valid:
                    logger.warning(
                        f"Chat {chat_id} (context_id {effective_topic_context_id}): Final AI text empty. Defaulting to error msg."
                    )
                    error_text = "I'm sorry, I couldn't generate a valid response. Please try again."
                    if llm_history:
                        llm_history.append(
                            {
                                "role": "assistant",
                                "content": error_text,
                            }
                        )
                    # Send the error message directly
                    await send_message_to_chat_or_general(
                        context.bot,
                        chat_id,
                        error_text,
                        preferred_thread_id=effective_send_thread_id,
                    )
                else:
                    # --- HYBRID SENDING LOGIC ---
                    if typing_task and not typing_task.done():
                        typing_task.cancel()  # Cancel "typing..." before we start sending content.

                    logger.info(
                        f"Chat {chat_id} (context_id {effective_topic_context_id}, send_thread_id {effective_send_thread_id}): AI Response for user: >>>{final_text_from_llm_before_media_extraction[:120].replace(chr(10),'/N')}...<<<"
                    )

                    # Manually parse for Shapes.inc media URLs first.
                    image_urls_to_send, audio_urls_to_send = [], []
                    text_after_stripping_media = final_text_from_llm_before_media_extraction

                    img_pattern = re.compile(r"(https://files\.shapes\.inc/[\w.-]+\.(?:png|jpg|jpeg|gif|webp))\b", re.I)
                    audio_pattern = re.compile(r"(https://files\.shapes\.inc/[\w.-]+\.(?:mp3|ogg|wav|m4a|flac))\b", re.I)

                    # Extract and remove image URLs
                    found_images = img_pattern.findall(text_after_stripping_media)
                    if found_images:
                        image_urls_to_send.extend(found_images)
                        text_after_stripping_media = img_pattern.sub("", text_after_stripping_media).strip()

                    # Extract and remove audio URLs
                    found_audio = audio_pattern.findall(text_after_stripping_media)
                    if found_audio:
                        audio_urls_to_send.extend(found_audio)
                        text_after_stripping_media = audio_pattern.sub("", text_after_stripping_media).strip()

                    # Send the extracted media files.
                    for img_url in image_urls_to_send:
                        try:
                            await send_photo_to_chat_or_general(
                                context.bot, chat_id, img_url, preferred_thread_id=effective_send_thread_id
                            )
                        except Exception as e:
                            logger.error(f"Failed to send Shapes.inc image {img_url}: {e}")
                        await asyncio.sleep(0.5)

                    for audio_url in audio_urls_to_send:
                        try:
                            await send_audio_to_chat_or_general(
                                context.bot, chat_id, audio_url, preferred_thread_id=effective_send_thread_id
                            )
                        except Exception as e:
                            logger.error(f"Failed to send Shapes.inc audio {audio_url}: {e}")
                        await asyncio.sleep(0.5)

                    # Process the *remaining* text with telegramify-markdown.
                    text_after_stripping_media = re.sub(r"\[([^\]]*)\]\(\s*\)", r"\1", text_after_stripping_media)
                    text_after_stripping_media = re.sub(r"(\r\n|\r|\n){2,}", "\n", text_after_stripping_media).strip()

                    if text_after_stripping_media:
                        # This function now correctly handles the rest of the message.
                        await send_telegramify_message(
                            context,
                            chat_id,
                            text_after_stripping_media,
                            effective_send_thread_id,
                        )
                    else:
                        logger.info(f"Chat {chat_id}: No remaining text to send after processing media URLs.")

            except InternalServerError as e_ise:
                log_ctx_info = (
                    f"Chat {chat_id} (context_id {effective_topic_context_id})"
                )
                err_msg = (
                    "The AI is taking too long (gateway timeout)."
                    if hasattr(e_ise, "response")
                    and e_ise.response
                    and e_ise.response.status_code == 504
                    else "AI service internal error. Try later."
                )
                logger.error(
                    f"{log_ctx_info}: OpenAI InternalServerError: {e_ise}",
                    exc_info=True,
                )
                if not (
                    llm_history
                    and llm_history[-1].get("role") == "assistant"
                    and llm_history[-1].get("content") == err_msg
                ):
                    llm_history.append({"role": "assistant", "content": err_msg})
                    db_manager.save_history(chat_id, effective_topic_context_id, llm_history)
                if chat_id:
                    try:
                        await send_message_to_chat_or_general(
                            context.bot,
                            chat_id,
                            err_msg,
                            preferred_thread_id=effective_send_thread_id,
                        )
                    except Exception as e_send_err:
                        logger.error(f"Error sending ISE msg: {e_send_err}")

            except APITimeoutError as e_timeout:
                log_ctx_info = (
                    f"Chat {chat_id} (context_id {effective_topic_context_id})"
                )
                err_msg = (
                    "AI is too slow, request timed out. Try shorter query or later."
                )
                logger.error(
                    f"{log_ctx_info}: OpenAI APITimeoutError (client-side).",
                    exc_info=True,
                )
                if not (
                    llm_history
                    and llm_history[-1].get("role") == "assistant"
                    and llm_history[-1].get("content") == err_msg
                ):
                    llm_history.append({"role": "assistant", "content": err_msg})
                    db_manager.save_history(chat_id, effective_topic_context_id, llm_history)
                if chat_id:
                    try:
                        await send_message_to_chat_or_general(
                            context.bot,
                            chat_id,
                            err_msg,
                            preferred_thread_id=effective_send_thread_id,
                        )
                    except Exception as e_send_err:
                        logger.error(f"Error sending Timeout msg: {e_send_err}")

            except telegram.error.BadRequest as e_tg_badreq:
                logger.error(
                    f"Outer Telegram BadRequest for chat {chat_id} (context_id {effective_topic_context_id}, send_thread_id {effective_send_thread_id}): {e_tg_badreq}. Raw AI: '{final_text_from_llm_before_media_extraction[:200]}'",
                    exc_info=True,
                )
                try:
                    plain_fb = final_text_from_llm_before_media_extraction
                    if len(plain_fb) <= 4096:
                        await context.bot.send_message(
                            chat_id, text=plain_fb, message_thread_id=None
                        )  # Fallback to general
                    else:
                        logger.warning(
                            f"Chat {chat_id}: Outer plain fallback too long ({len(plain_fb)}), splitting for general."
                        )
                        for i, chunk in enumerate(
                            [
                                plain_fb[j : j + 4096]
                                for j in range(0, len(plain_fb), 4096)
                            ]
                        ):
                            hdr = (
                                f"[Fallback Pt {i+1}/{ (len(plain_fb)+4095)//4096 }]\n"
                                if len(plain_fb) > 4096
                                else ""
                            )
                            await context.bot.send_message(
                                chat_id, text=hdr + chunk, message_thread_id=None
                            )  # Fallback to general
                            if i < len(plain_fb) // 4096:
                                await asyncio.sleep(0.5)
                except Exception as e_fb_send:
                    logger.error(
                        f"Chat {chat_id}: Outer plain fallback send failed: {e_fb_send}"
                    )
                    try:
                        await context.bot.send_message(
                            chat_id,
                            "A general error occurred while formatting my response. (OBRF)",
                            message_thread_id=None,
                        )  # Fallback to general
                    except Exception as e_final:
                        logger.error(f"Even final OBRF message failed: {e_final}")

            except (
                httpx.NetworkError,
                httpx.TimeoutException,
                httpx.ConnectError,
                telegram.error.NetworkError,
                telegram.error.TimedOut,
            ) as e_net:
                logger.error(
                    f"Network error for chat {chat_id} (context_id {effective_topic_context_id}): {e_net}",
                    exc_info=False,
                )
                if chat_id:
                    try:
                        await send_message_to_chat_or_general(
                            context.bot,
                            chat_id,
                            "⚠️ Network issues. Try again later.",
                            preferred_thread_id=effective_send_thread_id,
                        )
                    except Exception as e_send_net_err:
                        logger.error(
                            f"Chat {chat_id}: Failed to send network error notification: {e_send_net_err}"
                        )

            except Exception as e_main:
                logger.error(
                    f"General unhandled error in process_msg for chat {chat_id} (context_id {effective_topic_context_id}): {e_main}",
                    exc_info=True,
                )
                if chat_id:
                    try:
                        await send_message_to_chat_or_general(
                            context.bot,
                            chat_id,
                            "😵‍💫 Oops! Something went wrong. Noted. Try again. (General error, most likely RATE LIMIT)",
                            preferred_thread_id=effective_send_thread_id,
                        )
                    except Exception as e_send_gen_err:
                        logger.error(
                            f"Chat {chat_id}: Failed to send general error notification: {e_send_gen_err}"
                        )
            finally:
                if typing_task and not typing_task.done():
                    typing_task.cancel()
                    logger.debug(
                        f"Chat {chat_id} (context_id {effective_topic_context_id}, send_thread_id {effective_send_thread_id}): Typing task cancelled."
                    )
        finally:
            logger.info(
                f"Processing finished. Releasing lock for context key: {lock_key}."
            )
# --- END OF Main Message Handler ---

# --- ERROR HANDLER ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Log the error with traceback
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    tb_list = traceback.format_exception(
        None, context.error, context.error.__traceback__
    )
    tb_string = "".join(tb_list)

    # Format update object for logging
    update_str = "Update data not available or not an Update instance."
    effective_chat_id_for_error = "N/A"  # Default
    message_thread_id_for_error: Optional[int] = None

    if isinstance(update, Update):
        try:
            update_str = json.dumps(
                update.to_dict(), indent=2, ensure_ascii=False, default=str
            )
        except Exception:
            update_str = str(update)  # Fallback if to_dict fails
        if update.effective_chat:
            effective_chat_id_for_error = str(update.effective_chat.id)
        # Get thread ID from effective_message if available (covers commands, etc.)
        if (
            update.effective_message
            and update.effective_message.message_thread_id is not None
        ):
            message_thread_id_for_error = update.effective_message.message_thread_id
    elif update:  # If update is not None but not an Update instance
        update_str = str(update)

    # Format context data (truncated)
    chat_data_str = str(context.chat_data)[:500] if context.chat_data else "N/A"
    user_data_str = str(context.user_data)[:500] if context.user_data else "N/A"

    # Prepare error message for admin (plain text)
    thread_info_for_error_log = (
        f"(thread {message_thread_id_for_error})"
        if message_thread_id_for_error is not None
        else ""
    )
    send_message_plain = (
        f"Bot Exception in chat {effective_chat_id_for_error} {thread_info_for_error_log}:\n"
        f"Update: {update_str[:1500]}...\n\n"
        f"Chat Data: {chat_data_str}\nUser Data: {user_data_str}\n\n"
        f"Traceback (last 1500 chars):\n{tb_string[-1500:]}"
    )

    # Determine owner chat ID for notification
    chat_id_to_notify_owner: Optional[int] = None
    if NOTIFY_OWNER_ON_ERROR and BOT_OWNERS:  # Check if notifications are enabled AND owners are set
        try:
            chat_id_to_notify_owner = int(
                BOT_OWNERS[0]
            )  # Attempt to get the first owner's user ID
        except (ValueError, IndexError, TypeError):
            logger.error(
                "BOT_OWNERS is not configured with a valid user ID. Cannot send error report."
            )

    # --- Notify User (Optional) ---
    user_notified_by_handler = False
    if isinstance(update, Update) and update.effective_chat:
        # Avoid sending generic error if it's a network/API timeout, as those are handled in process_message_entrypoint
        # Check against imported exceptions
        if not isinstance(
            context.error,
            (
                InternalServerError,
                APITimeoutError,
                telegram.error.NetworkError,
                httpx.NetworkError,
                httpx.TimeoutException,
            ),
        ):  # Added httpx.TimeoutException
            try:
                # Send a user-friendly HTML message
                user_error_message = f"<b>Bot Error:</b> <pre>{html.escape(str(context.error))}</pre>\n<i>An unexpected error occurred. The admin has been notified if configured.</i>"
                if len(user_error_message) <= 4096:  # Telegram message length limit
                    # Use send_message_to_chat_or_general for user notification too, respecting thread
                    await send_message_to_chat_or_general(
                        context.bot,
                        update.effective_chat.id,
                        user_error_message,
                        preferred_thread_id=message_thread_id_for_error,  # Pass thread ID
                        parse_mode=ParseMode.HTML,
                    )
                    user_notified_by_handler = True
            except Exception as e_send_user_err:
                logger.error(
                    f"Failed to send user-friendly error to chat {update.effective_chat.id} (thread {message_thread_id_for_error}): {e_send_user_err}"
                )

    # --- Notify Admin ---
    if chat_id_to_notify_owner:
        # Avoid duplicate notification if admin is the one who experienced the error and was already notified
        if (
            user_notified_by_handler
            and update.effective_chat
            and update.effective_chat.id == chat_id_to_notify_owner
        ):
            logger.info(
                f"Admin was the user in chat {chat_id_to_notify_owner} (thread {message_thread_id_for_error}) and already notified about the error. Skipping redundant admin report."
            )
        else:
            max_len = 4096  # Telegram message length limit
            try:
                # Prefer HTML for admin if error is short and not potentially full of HTML itself
                # Use imported InternalServerError
                is_potentially_html_error = isinstance(
                    context.error, (InternalServerError, httpx.HTTPStatusError)
                )  # These might contain HTML in response

                thread_info_html = (
                    f"(thread {message_thread_id_for_error})"
                    if message_thread_id_for_error is not None
                    else ""
                )
                # Admin messages always go to the admin's direct chat, so no thread_id needed for the send_message call below

                # Try sending a short HTML version first if suitable
                if (
                    not is_potentially_html_error
                    and len(send_message_plain) < max_len - 200
                ):  # If plain text is short enough for HTML wrapper
                    short_html_err = f"<b>Bot Error in chat {effective_chat_id_for_error} {thread_info_html}:</b>\n<pre>{html.escape(str(context.error))}</pre>\n<i>(Full details in server logs. Update/TB follows if space.)</i>"
                    if (
                        len(short_html_err) <= max_len
                    ):  # Check if HTML version is within limits
                        await context.bot.send_message(
                            chat_id=chat_id_to_notify_owner,
                            text=short_html_err,
                            parse_mode=ParseMode.HTML,
                        )
                    else:  # HTML version too long, revert to plain
                        is_potentially_html_error = True  # Force plain text path

                # Send as plain text if it's long or potentially contains HTML
                if (
                    is_potentially_html_error
                    or len(send_message_plain) >= max_len - 200
                ):
                    if len(send_message_plain) <= max_len:
                        await context.bot.send_message(
                            chat_id=chat_id_to_notify_owner, text=send_message_plain
                        )
                    else:  # Split long plain text message for admin
                        num_err_chunks = (
                            len(send_message_plain) + max_len - 1
                        ) // max_len
                        for i_err, chunk in enumerate(
                            [
                                send_message_plain[j : j + max_len]
                                for j in range(0, len(send_message_plain), max_len)
                            ]
                        ):
                            # Add context header to each chunk
                            hdr = (
                                f"[BOT ERR Pt {i_err+1}/{num_err_chunks} Chat {effective_chat_id_for_error} {thread_info_for_error_log}]\n"
                                if num_err_chunks > 1
                                else f"[BOT ERR Chat {effective_chat_id_for_error} {thread_info_for_error_log}]\n"
                            )
                            # Ensure chunk with header doesn't exceed max_len
                            await context.bot.send_message(
                                chat_id=chat_id_to_notify_owner,
                                text=(hdr + chunk)[:max_len],
                            )
                            if i_err < num_err_chunks - 1:
                                await asyncio.sleep(0.5)  # Brief pause between chunks
            except Exception as e_send_err:  # Fallback if sending detailed error fails
                logger.error(
                    f"Failed sending detailed error report to admin {chat_id_to_notify_owner}: {e_send_err}"
                )
                # Send a minimal plain text error to admin
                try:
                    await context.bot.send_message(
                        chat_id=chat_id_to_notify_owner,
                        text=f"Bot Error in chat {effective_chat_id_for_error} {thread_info_for_error_log}: {str(context.error)[:1000]}\n(Details in server logs. Report sending failed.)",
                    )
                except Exception as e_final_fb:
                    logger.error(
                        f"Final admin error report fallback failed: {e_final_fb}"
                    )
    # Log if no notification was sent anywhere
    elif not user_notified_by_handler:
        logger.error(
            f"No chat ID found to send error message via Telegram (admin not set, or user already got specific error from main handler). Error details for chat {effective_chat_id_for_error} {thread_info_for_error_log} logged to server."
        )


# --- END OF ERROR HANDLER ---


async def post_initialization(application: Application) -> None:
    """Actions to perform after the bot is initialized, like setting commands."""
    bot_commands = [
        BotCommand("start", "Display the welcome message."),
        BotCommand("help", "Show this help message."),
        BotCommand("newchat", "Clear conversation history for this topic/chat."),
        BotCommand("activate", "(Groups/Topics) Respond to all messages here."),
        BotCommand("deactivate", "(Groups/Topics) Stop responding to all messages here."),
        BotCommand("auth_shapes", "Connect your Shapes.inc account."),
    ]
    if (
        BING_IMAGE_CREATOR_AVAILABLE and BING_AUTH_COOKIE
    ):  # Only show /imagine if fully configured
        bot_commands.append(
            BotCommand("imagine", "Generate images from a prompt (Bing).")
        )

    # setbingcookie is an admin command, only show if BING is available and there are owners
    if BING_IMAGE_CREATOR_AVAILABLE and BOT_OWNERS:
        bot_commands.append(
            BotCommand("setbingcookie", "(Owner) Update Bing auth cookie.")
        )

    try:
        await application.bot.set_my_commands(bot_commands)
        logger.info(
            f"Successfully set bot commands: {[cmd.command for cmd in bot_commands]}"
        )
    except Exception as e:
        logger.error(f"Failed to set bot commands: {e}")


if __name__ == "__main__":

    # Get the DB path
    DATABASE_PATH = os.getenv("DATABASE_PATH", "bot_database.db")

    # Ensure the directory for the database exists inside the container
    db_directory = os.path.dirname(DATABASE_PATH)
    if db_directory:
        os.makedirs(db_directory, exist_ok=True)

    # Initialize the Database Manager with the specified path
    db_manager = DatabaseManager(DATABASE_PATH)
    
    # Load all persistent state into in-memory caches for fast access
    user_auth_tokens = db_manager.load_all_user_tokens()
    chat_histories = db_manager.load_all_histories()
    activated_chats_topics = db_manager.load_all_activated_topics()

    logger.info(f"Loaded {len(user_auth_tokens)} user tokens from DB.")
    logger.info(f"Loaded {len(chat_histories)} conversation histories from DB.")
    logger.info(f"Loaded {len(activated_chats_topics)} activated topics from DB.")

    # Record bot startup time (UTC)
    BOT_STARTUP_TIMESTAMP = datetime.now(dt_timezone.utc)
    logger.info(f"Bot starting up at: {BOT_STARTUP_TIMESTAMP}")
    if IGNORE_OLD_MESSAGES_ON_STARTUP:
        logger.info("Bot will ignore messages received before this startup time.")

    # Increase timeouts for network requests, especially for sending files
    # Default is 5s, which is too short for uploads. We'll set a longer read/write timeout.
    app_builder = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .concurrent_updates(True)
        .connect_timeout(10.0)
        .read_timeout(60.0)
        .write_timeout(60.0)
    )  # Increased timeouts

    app_builder.post_init(post_initialization)  # Register post_init hook
    app = app_builder.build()

    # --- Create the ConversationHandler for Authentication ---
    auth_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("auth_shapes", auth_start)],
        states={
            AWAITING_CODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, auth_receive_code)],
        },
        fallbacks=[CommandHandler("cancel", auth_cancel)],
        per_user=True,  # This is crucial to keep conversations separate for each user
        per_chat=True,  # And for each user within a specific chat
    )

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
    app.add_handler(auth_conv_handler)
    # Only add imagine/setbingcookie if library is available
    if BING_IMAGE_CREATOR_AVAILABLE:
        # Only add /imagine if cookie is also set initially
        if BING_AUTH_COOKIE:
            app.add_handler(CommandHandler("imagine", imagine_command))
        # Admin can set cookie even if not initially present
        app.add_handler(CommandHandler("setbingcookie", set_bing_cookie_command))

    # --- Handler for Forum Topic Updates (Created/Edited) ---
    # This handler is specifically for populating the topic_names_cache.
    app.add_handler(
        MessageHandler(
            filters.StatusUpdate.FORUM_TOPIC_CREATED
            | filters.StatusUpdate.FORUM_TOPIC_EDITED,
            handle_forum_topic_updates,
        )
    )
    # Message Handler for text, photos, voice, and replies (but not commands)
    app.add_handler(
        MessageHandler(
            (
                filters.TEXT
                | filters.PHOTO
                | filters.VOICE
                | filters.Sticker.ALL
                | filters.Document.ALL
                | filters.REPLY
            )
            & (~filters.COMMAND)
            & (~filters.StatusUpdate.ALL),
            process_message_entrypoint,
        )
    )

    # Error Handler
    app.add_error_handler(error_handler)

    logger.info("Bot is starting to poll for updates...")
    try:
        # Start polling
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    # Specific catch for initial network errors during startup
    except telegram.error.NetworkError as ne:
        logger.critical(
            f"CRITICAL: Initial NetworkError during polling setup: {ne}. Check network/token.",
            exc_info=True,
        )
    # Catch any other critical errors during startup/polling loop
    except Exception as main_e:
        logger.critical(
            f"CRITICAL: Unhandled exception at main polling level: {main_e}",
            exc_info=True,
        )
