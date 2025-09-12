# Telegram AI Chatbot with Shapes.inc & Tools

This project is a Python-based Telegram bot that leverages the Shapes.inc platform (an OpenAI-compatible API) for conversational AI. It's designed to be extensible, allowing multiple bot "personalities" to run from the same codebase using Docker Compose and separate environment files.

The bot can:
*   **Persist its state & settings:** Utilizes a **SQLite database** to save conversation histories, user authentication data, bot configuration, and now includes **user moderation status (ignored users) and detailed usage statistics**.
*   **Authenticate Users:** Offers an `/auth_shapes` flow for users to link their Shapes.inc account, enabling personalized API interactions.
*   Engage in text-based conversations.
*   **Process Advanced Media:** Understand and respond to a wide range of media, including static images, **animated stickers (.tgs), video stickers (.webm), GIFs, and videos** by extracting key frames for visual context.
*   Process voice messages sent by users (as direct voice messages or document attachments).
*   **Process text from uploaded documents:** Supports `.txt`, `.md`, `.docx`, and `.odt` files.
*   Utilize tools like a calculator, weather information, web search, creating polls, fetching game deals, moderating users, getting user/chat info, and generating anime-style images.
*   Potentially generate images using Bing Image Creator (this feature remains experimental).
*   **Support Telegram Group Topics:**
    *   Maintain independent, **persistent** conversation histories per topic (and the general chat area) if `SEPARATE_TOPIC_HISTORIES` is enabled.
    *   Operate its "free will" mode contextually within each topic/general area.
    *   Allow **group administrators** to activate/deactivate the bot for listening to all messages within a specific topic.
    *   Target tools (like polls) to the specific topic/chat where the command was issued.
    *   Implement processing locks to handle messages sequentially within a given conversational context.
*   In group chats (including topics), it can operate in a "free will" mode, occasionally interjecting into conversations based on recent message context within that specific chat/topic.

**Disclaimer:** This codebase is mostly the result of "vibe coding" and was developed rapidly, evolving from an older Poe.com Telegram bot. While new features like persistence and authentication have been added, it's functional but should be considered "quick and dirty." Significant cleanup, refactoring, and more robust error handling would be beneficial for production use. Some features, like the Bing Image generation, might be unreliable or broken.

## Features

*   **Database Persistence:** The bot uses a **SQLite database** to store its state, ensuring data is not lost on restart. This includes:
    *   Conversation histories for every chat and topic.
    *   User authentication tokens obtained from the Shapes.inc authentication flow.
    *   Topic activation settings (`/activate` status).
    *   **Bot configuration**, allowing settings to persist and be changed dynamically.
    *   **User moderation data**, including a list of ignored (blocked) users and their expiration times.
    *   **Detailed usage statistics** for messages and LLM API calls.
*   **Advanced Message Formatting:** Uses the `telegramify-markdown` library to:
    *   Intelligently split long messages without breaking markdown formatting.
    *   Automatically render code blocks as image snippets or file attachments (e.g., `.py`, `.html`).
*   **Shapes.inc User Authentication:**
    *   A new `/auth_shapes` command initiates a secure, multi-step flow for users to connect their Shapes.inc account.
    *   This allows the bot to make API calls on behalf of the authenticated user for a more personalized experience and memory.
*   **Dynamic & Remote Configuration (Owner-Only):**
    *   Bot settings are loaded from `.env` files at startup and can be **overridden by values stored in the database**.
    *   New `/viewsettings` command to display the current bot configuration securely.
    *   New `/setsetting` command allows bot owners to **change most settings on-the-fly** without restarting the bot. Changes are saved to the database and are persistent.
*   **Statistics and Analytics (Owner-Only):**
    *   The bot now logs detailed statistics for every message and LLM API call to its database.
    *   A new `/stats` command provides a comprehensive overview of usage, including token counts, model usage, API call metrics, user/group activity, and more.
*   **User Moderation System (Owner-Only & AI Tool):**
    *   A persistent, database-backed system to ignore (block) users from interacting with the bot for a specified duration.
    *   New owner commands (`/ignore`, `/unignore`, `/listignored`) for manual user management.
    *   A new `manage_ignored_user` tool allows the AI to programmatically block users.
*   **State-Aware API Interaction:** The bot now sends only the most recent message turn to the Shapes.inc API, relying on its stateful nature to reduce network overhead.
*   **Conversational AI:** Powered by Shapes.inc (via `openai` library targeting a custom base URL).
*   **Advanced Multi-Modal Input:** Can process text, voice messages, and text-based documents (`.txt`, `.md`, `.docx`, `.odt`), plus a wide range of visual media:
    *   Static images (photos, stickers, documents).
    *   **Animated Stickers (.tgs):** Renders keyframes to understand the animation.
    *   **Video Stickers (.webm):** Extracts keyframes to understand the video loop.
    *   **GIFs and Videos:** Extracts keyframes from `.mp4` and `.gif` files for visual context.
    *   **Media Groups (Albums):** Intelligently groups photos/videos sent as an album into a single context before processing.
*   **Tool Usage (Function Calling):**
    *   `calculator`: Evaluates mathematical expressions.
    *   `get_weather`: Fetches current, hourly, or daily weather forecasts.
    *   `web_search`: Performs web searches using DuckDuckGo.
    *   `create_poll_in_chat`: Creates a poll in the current chat/topic.
    *   `get_game_deals`: Fetches information about free game giveaways.
    *   `restrict_user_in_chat`: Temporarily mutes a user in the chat.
    *   `get_user_info`: Retrieves comprehensive information about a chat member.
    *   `get_chat_info`: Retrieves information about the current chat, such as title and member count.
    *   `manage_ignored_user`: Blocks or unblocks a user from interacting with the bot.
    *   `generate_anime_image`: Generates an anime-style image using Perchance API.
*   **Image Generation (Experimental):** `/imagine` command using Bing Image Creator (requires `BING_AUTH_COOKIE`).
*   **Telegram Group Topic Support:**
    *   Configurable independent and persistent conversation histories per topic (and general chat) via `SEPARATE_TOPIC_HISTORIES`.
    *   Free will mode, `/newchat`, and tool usage are scoped to the current topic/general chat.
    *   `/activate` & `/deactivate`: **(Group Admin-only)** Bot listens to all messages in the current topic/general chat.
    *   Caches topic names for better contextual understanding.
    *   Introduced processing locks to ensure messages within the same conversational context are processed one at a time.
*   **Group "Free Will" Mode:** Bot can spontaneously respond in group chats/topics based on configurable probability and message context specific to that chat/topic.
*   **Granular Permissions:**
    *   **Whitelist:** `ALLOWED_USERS` and `ALLOWED_CHATS` restrict who can interact with the bot.
    *   **Ownership:** A separate `BOT_OWNERS` list grants administrative privileges for sensitive commands.
    *   **Configurable Error Reporting:** `NOTIFY_OWNER_ON_ERROR` flag to enable/disable error DMs to the owner.
*   **Conversation History Management:** `/newchat` command now clears the conversation history for the current chat/topic from both memory and the **database**. Note: This primarily affects the bot's local context. The bot now sends a more specific `X-Channel-Id` header that might help Shapes.inc with their history scoping.
*   **Dockerized Deployment:** Easily run multiple bot instances with `docker-compose`, now with a **persistent data volume** for the database.
*   **Dynamic Configuration:** Most settings are managed via environment variables.
*   **Ignore Old Messages (Optional):** Can be configured to ignore messages received before the bot started up.
*   **Dynamic Bot Commands:** Bot commands list in Telegram UI is updated on startup based on available features.
*   **Concurrent Update Processing:** Bot is configured to handle multiple updates concurrently, with safety locks.

## Tech Stack

*   **Python 3.11** (as per Dockerfile)
*   **`sqlite3`**: Used for all database operations and data persistence.
*   **`python-telegram-bot[ext]>=21.11,<22.0`**: For Telegram Bot API interaction.
*   **`telegramify-markdown[mermaid]==0.5.1`**: For advanced markdown parsing and message sending.
*   **`openai>=1.107.2,<2.0.0`**: To interact with the Shapes.inc API.
*   **`python-dotenv>=1.1.0,<1.2.0`**: For managing environment variables.
*   **`httpx[http2]>=0.27.0,<0.28.0`**: For making HTTP requests.
*   **`pytz>=2025.2,<2026.0`**, **`beautifulsoup4>=4.13.5,<4.14.0`**: Used by weather and web search tools.
*   **`BingImageCreator>=0.5.0,<0.6.0`**: (Optional) For the `/imagine` command.
*   **`python-docx>=1.2.0,<2.0.0`**, **`odfpy>=1.4.1,<2.0.0`**: For reading `.docx` and `.odt` files.
*   **`Pillow>=11.3.0,<11.4.0`**: For basic image processing, especially with GIFs.
*   **`rlottie-python>=1.3.8,<1.4.0`**: For rendering frames from animated `.tgs` (Lottie) stickers.
*   **`opencv-python>=4.12.0,<5.0.0`**: For extracting frames from video stickers (`.webm`) and other video formats.
*   **`numpy>=2.2.0,<2.3.0`**: A core dependency for `opencv-python`.
*   **Docker & Docker Compose**: For containerization and multi-bot deployment.

## Prerequisites

*   Python 3.10+ (3.11 recommended)
*   pip (Python package installer)
*   Docker and Docker Compose (if using Docker for deployment)
*   A Telegram Bot Token for each bot instance.
*   A Shapes.inc API Key.
*   A Shapes.inc Shape Username/Vanity URL for each bot instance.
*   A Shapes.inc App ID (required for the `/auth_shapes` command).
*   Your own Telegram User ID to set as the `BOT_OWNERS`.
*   (Optional) Bing Auth Cookie if you want to test/use the `/imagine` command.
*   (Optional) `PERCHANCE_USER_KEY` if you want to use the `generate_anime_image` tool.

## Setup and Configuration

The bot is configured primarily through environment variables. The `docker-compose.yml` is set up to use a `common.env` file for shared settings and bot-specific `.env` files (e.g., `nova-ai.env`, `discordaddictamy.env`) for per-bot configurations.

Note: The bot now uses a hierarchical settings system. At startup, it first loads default values from your `.env` files. It then connects to the database and loads any settings stored there, which **override** the defaults. This allows you to use the new owner commands to change settings while the bot is running, and those changes will be persistent.

### 1. Environment Variables

You'll need to create the following `.env` files in the project root:

**a) `common.env` (for shared settings):**
Create this file and populate it with settings common to all bot instances. See `example.env` for all possible variables.

```ini
# common.env
SHAPESINC_API_KEY=your_shapes_inc_api_key_here
SHAPESINC_APP_ID=your_shapes_inc_app_id_here # Required for /auth_shapes
DATABASE_PATH=/data/bot_database.db # IMPORTANT: Path inside the container for persistent storage

# --- Permissions & Administration ---
# Optional: User and Chat Whitelisting (applies to all bots unless overridden)
# ALLOWED_USERS=
# ALLOWED_CHATS=

# Bot Owners / Admins (comma-separated list of user IDs)
# The first ID in this list will receive error reports if enabled.
BOT_OWNERS=12345678

# Control Error Notifications
# Set to "false" to prevent the bot owner from being DMed with error reports.
NOTIFY_OWNER_ON_ERROR=true
# ------------------------------------

# Optional: Override Shapes API Base URL
# SHAPES_API_BASE_URL=https://api.shapes.inc/v1/

# Free will settings
GROUP_FREE_WILL_ENABLED=true
GROUP_FREE_WILL_PROBABILITY=0.05 # 0.05 is 5%
GROUP_FREE_WILL_CONTEXT_MESSAGES=5 # Number of recent messages within the topic/chat to consider

# Optional: Sets whether each TOPIC/THREAD in a telegram group should be a separate chat history
# Set to true for separate histories per topic, false for one history per group.
SEPARATE_TOPIC_HISTORIES=true

# Enable Tool Use
ENABLE_TOOL_USE=true

# Optional: Ignore messages older than bot startup time
# IGNORE_OLD_MESSAGES_ON_STARTUP=false

# Perchance user key for generate_anime_image tool
# PERCHANCE_USER_KEY=your_perchance_user_key_here

# Optional: Bing Image Creator Cookie (if you want to try /imagine)
# BING_AUTH_COOKIE=your_bing_auth_cookie_here
```

**b) Bot-Specific `.env` Files (e.g., `nova-ai.env`):**
For each bot instance defined in `docker-compose.yml`, create a corresponding `.env` file. These files **must** contain `BOT_TOKEN` and `SHAPESINC_SHAPE_USERNAME`.

Example for `nova-ai.env`:
```ini
# nova-ai.env
BOT_TOKEN=your_telegram_bot_token_for_nova_ai_here
SHAPESINC_SHAPE_USERNAME=nova-ai # The Shapes.inc username for this bot
```

Example for `discordaddictamy.env`:
```ini
# discordaddictamy.env
BOT_TOKEN=your_telegram_bot_token_for_amy_here
SHAPESINC_SHAPE_USERNAME=discordaddictamy
```

**Important:**
*   Copy `example.env` to get a template for all variables.
*   The variables `BOT_TOKEN` and `SHAPESINC_SHAPE_USERNAME` are **required** in the bot-specific `.env` files.
*   Do **not** commit your actual `.env` files to version control. They are included in `.gitignore`.

### 2. Local Setup (for development or running a single instance without Docker)

If you want to run a single bot instance locally without Docker:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/xXG0DLessXx/shapes-inc-telegram-bot
    cd shapes-inc-telegram-bot
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a single `.env` file in the project root:**
    This file must contain all necessary variables, including `BOT_TOKEN`, `SHAPESINC_SHAPE_USERNAME`, `SHAPESINC_API_KEY`, `SHAPESINC_APP_ID`, `BOT_OWNERS`, and a local `DATABASE_PATH` (e.g., `DATABASE_PATH=bot_database.db`).

    ```ini
    # .env (for local, single bot run)
    BOT_TOKEN=your_telegram_bot_token_here
    SHAPESINC_API_KEY=your_shapes_inc_api_key_here
    SHAPESINC_SHAPE_USERNAME=your_shape_username_here
    SHAPESINC_APP_ID=your_shapes_inc_app_id_here
    BOT_OWNERS=your_telegram_user_id_here
    DATABASE_PATH=bot_database.db

    # Free will settings
    GROUP_FREE_WILL_ENABLED=true
    GROUP_FREE_WILL_PROBABILITY=0.05
    GROUP_FREE_WILL_CONTEXT_MESSAGES=5
    
    # Optional: Sets whether each TOPIC/THREAD in a telegram group should be a separate chat history
    SEPARATE_TOPIC_HISTORIES=true
    
    # Enable Tool Use
    ENABLE_TOOL_USE=true
    
    # Control Error Notifications
    NOTIFY_OWNER_ON_ERROR=true
    
    # Optional settings
    # IGNORE_OLD_MESSAGES_ON_STARTUP=false
    # PERCHANCE_USER_KEY=your_perchance_user_key_here
    # BING_AUTH_COOKIE=your_bing_auth_cookie_here
    ```

### 3. Docker Setup (Recommended for multi-bot deployment)

1.  Ensure Docker and Docker Compose are installed.
2.  Clone the repository (if not already done).
3.  Create the `common.env` file and all required bot-specific `.env` files as described above.
4.  **Build the Docker image:**
    ```bash
    docker-compose build
    ```
5.  **Run all defined bot services in detached mode:**
    ```bash
    docker-compose up -d
    ```

## Running the Bot

### Locally (Single Instance)

1.  Set up your `.env` file and install dependencies as described in "Local Setup".
2.  Activate your virtual environment.
3.  Run the bot:
    ```bash
    python bot.py
    ```
    The bot will start polling for updates. Press `CTRL+C` to stop.

### With Docker Compose (Multiple Instances)

1.  Ensure you have set up `common.env` and all required bot-specific `.env` files.
2.  **Build the Docker image:**
    (Only needed once, or when `Dockerfile` or `requirements.txt` change)
    ```bash
    docker-compose build
    ```

3.  **Run all defined bot services in detached mode:**
    ```bash
    docker-compose up -d
    ```

4.  **To run specific bot(s):**
    ```bash
    docker-compose up -d bot-nova-ai  # Runs only nova-ai
    docker-compose up -d bot-nova-ai bot-discordaddictamy # Runs both
    ```

5.  **To view logs:**
    ```bash
    docker-compose logs -f              # Logs for all services
    docker-compose logs -f bot-nova-ai  # Logs for a specific service
    ```

6.  **To stop the bots:**
    ```bash
    docker-compose down
    ```

A `bot_data` directory will be created in your project folder, containing the persistent SQLite database from the Docker volume.

## Available Commands

*   `/start`: Displays a welcome message.
*   `/help`: Shows this help message, including available tools and free will status.
*   `/auth_shapes`: Starts the process to connect your Shapes.inc account.
*   `/cancel`: Aborts a multi-step process like authentication.
*   `/newchat`: Clears the bot's side of the conversation history for the current chat/topic. The bot now sends a specific `X-Channel-Id` header that might help Shapes.inc with their history scoping.
*   `/activate`: **(Group Admin-only)** Makes the bot respond to every message in the current group/topic.
*   `/deactivate`: **(Group Admin-only)** Stops the bot from responding to every message in the current group/topic.
*   `/imagine <prompt>`: (Experimental) Generates images using Bing Image Creator.
*   `/stats`: **(Bot Owner-only)** View bot usage statistics.
*   `/ignore <id> <duration> [reason]`: **(Bot Owner-only)** Block a user from interacting with the bot.
*   `/unignore <id>`: **(Bot Owner-only)** Unblock a user.
*   `/listignored`: **(Bot Owner-only)** List all currently blocked users.
*   `/setbingcookie <cookie_value>`: **(Bot Owner-only)** Updates the Bing authentication cookie **and saves it to the database**. Restricted to users listed in `BOT_OWNERS`.
*   `/viewsettings`: **(Bot Owner-only)** Displays the current bot configuration, hiding sensitive values.
*   `/setsetting <NAME> <value>`: **(Bot Owner-only)** Changes a setting's value on-the-fly and saves it to the database.
*   `/setperchancekey <key>`: **(Bot Owner-only)** Sets the Perchance API key and saves it.

### How to Interact with the Bot

You can interact with the bot in several ways:

**In Direct Messages:**
*   Send it any text message.
*   Send it an image (with or without a caption, or as a document).
*   Send it a voice message (or as an audio document).
*   Send it a text document (`.txt`, `.md`, `.docx`, `.odt`).

**In Group chats (including those with Topics):**
*   Reply directly to one of the bot's messages.
*   Mention the bot by its username (e.g., `@your_bot_username <your message>`).
*   If a group admin has used `/activate`, the bot will respond to all messages in that specific context.
*   Wait for it to interject based on its "free will" settings for that specific chat/topic.

## Important Notes & Known Issues

*   **Persistence & Data**: The bot uses a SQLite database. When using Docker, this file is stored in a named volume (`bot_data`) to ensure it persists.
*   **Admin vs. Owner Controls**: Group-level commands like `/activate` are restricted to **group administrators**. Bot-level administrative commands like `/setbingcookie` or `/ignore` are restricted to **bot owners** defined in the `.env` file.
*   **Security:** Be very careful with your API keys, bot tokens, and `SHAPESINC_APP_ID`. Do not commit `.env` files containing secrets.
*   **Code Quality:** As stated, this is "quick and dirty" code. It's functional but could benefit from significant refactoring.
*   **Advanced Media Processing:** The bot now processes animated/video stickers, GIFs, and videos by extracting a limited number of frames to send to the vision model. This provides context but is not a full video analysis. File size limits are in place to prevent excessive processing.
*   **Media Group (Album) Handling:** When you send multiple images/videos as an album, the bot will wait a moment to collect all of them before processing them as a single, combined message.
*   **Anime Image Generation (`generate_anime_image` tool):** This tool uses the Perchance API and requires a `PERCHANCE_USER_KEY`.
*   **Voice/Audio Message Processing:** The bot sends the Telegram-provided URL for audio messages directly to the Shapes.inc API. The API must be able to access and process these URLs.
*   **Document Processing:** The bot can extract text from `.txt`, `.md`, `.docx`, and `.odt` files. There's a `MAX_TEXT_FILE_SIZE` limit (default 500KB).
*   **Telegram Topic Support Details:**
    *   The `SEPARATE_TOPIC_HISTORIES` variable controls whether conversation history is scoped per-topic or per-group.
    *   The bot sends a specific `X-Channel-Id` (e.g., `chatid_topicid` or `chatid_general`) to the Shapes.inc API, which should help the API scope its context.
    *   If `IGNORE_OLD_MESSAGES_ON_STARTUP` is set to `true`, the bot will skip processing messages that occurred before it started.
*   **Concurrent Updates & Locking:** The bot processes updates concurrently. To prevent race conditions, a locking mechanism ensures that messages for a specific chat/topic are handled one by one.
*   **Dynamic Settings System:** The bot's configuration is now dynamic. Bot owners can use the `/setsetting` command to change values like `GROUP_FREE_WILL_PROBABILITY` or the list of `ACTIVE_TOOLS` without a restart. The new values are immediately applied and saved to the database. The `.env` files are now primarily used to set the initial defaults.

## Contributing

This project was developed rapidly. If you wish to improve it:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -am 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Create a new Pull Request.

Focus areas for improvement could include:
*   Refactoring `bot.py` into smaller, more manageable modules.
*   Adding unit and integration tests.
*   Improving error handling and user feedback, especially for tool failures.
*   Stabilizing or replacing the Bing Image generation feature.

## License

This project is currently unlicensed. Feel free to use it as a base, but be mindful of the disclaimers.