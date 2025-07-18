# Telegram AI Chatbot with Shapes.inc & Tools

This project is a Python-based Telegram bot that leverages the Shapes.inc platform (an OpenAI-compatible API) for conversational AI. It's designed to be extensible, allowing multiple bot "personalities" to run from the same codebase using Docker Compose and separate environment files.

The bot can:
*   **Persist its state:** Utilizes a **SQLite database** to save conversation histories, user authentication data, and chat settings, ensuring data survives restarts.
*   **Authenticate Users:** Offers an `/auth_shapes` flow for users to link their Shapes.inc account, enabling personalized API interactions.
*   Engage in text-based conversations.
*   Understand and respond to images sent by users (as direct photos or document attachments).
*   Process voice messages sent by users (as direct voice messages or document attachments).
*   **Process text from uploaded documents:** Supports `.txt`, `.md`, `.docx`, and `.odt` files.
*   Utilize tools like a calculator, weather information, web search, creating polls, fetching game deals, moderating users, getting user info, and generating anime-style images.
*   Potentially generate images using Bing Image Creator (this feature remains experimental).
*   **Support Telegram Group Topics:**
    *   Maintain independent, **persistent** conversation histories per topic (and the general chat area) if `SEPARATE_TOPIC_HISTORIES` is enabled.
    *   Operate its "free will" mode contextually within each topic/general area.
    *   Allow **administrators** to activate/deactivate the bot for listening to all messages within a specific topic.
    *   Target tools (like polls) to the specific topic/chat where the command was issued.
    *   Implement processing locks to handle messages sequentially within a given conversational context.
*   In group chats (including topics), it can operate in a "free will" mode, occasionally interjecting into conversations based on recent message context within that specific chat/topic.

**Disclaimer:** This codebase is mostly the result of "vibe coding" and was developed rapidly, evolving from an older Poe.com Telegram bot. While new features like persistence and authentication have been added, it's functional but should be considered "quick and dirty." Significant cleanup, refactoring, and more robust error handling would be beneficial for production use. Some features, like the Bing Image generation, might be unreliable or broken.

## Features

*   **Database Persistence:** The bot uses a **SQLite database** to store its state, ensuring data is not lost on restart. This includes:
    *   Conversation histories for every chat and topic.
    *   User authentication tokens obtained from the Shapes.inc authentication flow.
    *   Topic activation settings (`/activate` status).
*   **Shapes.inc User Authentication:**
    *   A new `/auth_shapes` command initiates a secure, multi-step flow for users to connect their Shapes.inc account.
    *   This allows the bot to make API calls on behalf of the authenticated user for a more personalized experience and memory.
*   **Conversational AI:** Powered by Shapes.inc (via `openai` library targeting a custom base URL).
*   **Multi-Modal Input:** Can process text, images (direct or as documents), voice messages (direct or as documents), and text-based documents (`.txt`, `.md`, `.docx`, `.odt`).
*   **Tool Usage (Function Calling):**
    *   `calculator`: Evaluates mathematical expressions.
    *   `get_weather`: Fetches current, hourly, or daily weather forecasts.
    *   `web_search`: Performs web searches using DuckDuckGo.
    *   `create_poll_in_chat`: Creates a poll in the current chat/topic.
    *   `get_game_deals`: Fetches information about free game giveaways.
    *   `restrict_user_in_chat`: Temporarily mutes a user in the chat.
    *   `get_user_info`: Retrieves comprehensive information about a chat member.
    *   `generate_anime_image`: Generates an anime-style image using Perchance API.
*   **Image Generation (Experimental):** `/imagine` command using Bing Image Creator (requires `BING_AUTH_COOKIE`).
*   **Telegram Group Topic Support:**
    *   Configurable independent and persistent conversation histories per topic (and general chat) via `SEPARATE_TOPIC_HISTORIES`.
    *   Free will mode, `/newchat`, and tool usage are scoped to the current topic/general chat.
    *   `/activate` & `/deactivate`: **(Admin-only)** Bot listens to all messages in the current topic/general chat.
    *   Caches topic names for better contextual understanding.
    *   Introduced processing locks to ensure messages within the same conversational context are processed one at a time.
*   **Group "Free Will" Mode:** Bot can spontaneously respond in group chats/topics based on configurable probability and message context specific to that chat/topic.
*   **User/Chat Whitelisting:** Restrict bot access to specific users or chats.
*   **Conversation History Management:** `/newchat` command now clears the conversation history for the current chat/topic from both memory and the **database**. Note: This primarily affects the bot's local context. Shapes.inc manages its own context on their side; standard Shape commands like `!wack` might be more effective for a full reset on their platform.
*   **MarkdownV2 Support:** Advanced message formatting with intelligent splitting for long messages.
*   **Dockerized Deployment:** Easily run multiple bot instances with `docker-compose`, now with a **persistent data volume** for the database.
*   **Dynamic Configuration:** Most settings are managed via environment variables.
*   **Ignore Old Messages (Optional):** Can be configured to ignore messages received before the bot started up.
*   **Dynamic Bot Commands:** Bot commands list in Telegram UI is updated on startup based on available features.
*   **Concurrent Update Processing:** Bot is configured to handle multiple updates concurrently, with safety locks.

## Tech Stack

*   **Python 3.11** (as per Dockerfile)
*   **`sqlite3`**: Used for all database operations and data persistence (part of the Python standard library).
*   **`python-telegram-bot[ext]>=21.1.1,<22.0.0`**: For Telegram Bot API interaction.
*   **`openai>=1.35.7,<2.0.0`**: To interact with the Shapes.inc API.
*   **`python-dotenv>=1.0.1,<1.1.0`**: For managing environment variables.
*   **`httpx[http2]>=0.27.0,<0.28.0`**: For making HTTP requests.
*   **`pytz`, `beautifulsoup4`**: Used by weather and web search tools.
*   **`BingImageCreator`**: (Optional) For the `/imagine` command.
*   **`python-docx`, `odfpy`**: For reading `.docx` and `.odt` files.
*   **Docker & Docker Compose**: For containerization and multi-bot deployment.

## Prerequisites

*   Python 3.10+ (3.11 recommended)
*   pip (Python package installer)
*   Docker and Docker Compose (if using Docker for deployment)
*   A Telegram Bot Token for each bot instance.
*   A Shapes.inc API Key.
*   A Shapes.inc Shape Username/Vanity URL for each bot instance.
*   A Shapes.inc App ID (required for the `/auth_shapes` command).
*   (Optional) Bing Auth Cookie if you want to test/use the `/imagine` command.
*   (Optional) `PERCHANCE_USER_KEY` if you want to use the `generate_anime_image` tool.

## Setup and Configuration

The bot is configured primarily through environment variables. The `docker-compose.yml` is set up to use a `common.env` file for shared settings and bot-specific `.env` files (e.g., `nova-ai.env`, `discordaddictamy.env`) for per-bot configurations like tokens and shape usernames.

### 1. Environment Variables

You'll need to create the following `.env` files in the project root:

**a) `common.env` (for shared settings):**
Create this file and populate it with settings common to all bot instances. See `example.env` for all possible variables.

```ini
# common.env
SHAPESINC_API_KEY=your_shapes_inc_api_key_here
SHAPESINC_APP_ID=your_shapes_inc_app_id_here # Required for /auth_shapes
DATABASE_PATH=/data/bot_database.db # IMPORTANT: Path inside the container for persistent storage

# Optional: Override Shapes API Base URL
# SHAPES_API_BASE_URL=https://api.shapes.inc/v1/

# Optional: User and Chat Whitelisting (applies to all bots unless overridden)
# ALLOWED_USERS=12345678,87654321
# ALLOWED_CHATS=-100123456789,-100987654321

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
# IGNORE_OLD_MESSAGES_ON_STARTUP=false # Set to true to enable

# Perchance user key for generate_anime_image tool
# PERCHANCE_USER_KEY=your_perchance_user_key_here

# Optional: Bing Image Creator Cookie (if you want to try /imagine)
# BING_AUTH_COOKIE=your_bing_auth_cookie_here
```

**b) Bot-Specific `.env` Files (e.g., `nova-ai.env`, `discordaddictamy.env`):**
For each bot instance defined in `docker-compose.yml`, create a corresponding `.env` file.
These files **must** contain `BOT_TOKEN` and `SHAPESINC_SHAPE_USERNAME`.

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
*   Do **not** commit your actual `.env` files (especially those with tokens/keys) to version control. Add them to your `.gitignore`. (They should already be excluded but make sure).

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
    This file will be loaded directly by `bot.py`. It needs to contain all necessary variables, including `BOT_TOKEN` and `SHAPESINC_SHAPE_USERNAME`, `SHAPESINC_APP_ID`, and a local `DATABASE_PATH` (e.g., `DATABASE_PATH=bot_database.db`) for the single bot you intend to run.
    ```ini
    # .env (for local, single bot run)
    BOT_TOKEN=your_telegram_bot_token_here
    SHAPESINC_API_KEY=your_shapes_inc_api_key_here
    SHAPESINC_SHAPE_USERNAME=your_shape_username_here

    # ... other common settings from example.env ...
    SEPARATE_TOPIC_HISTORIES=true
    GROUP_FREE_WILL_ENABLED=true
    ENABLE_TOOL_USE=true
    IGNORE_OLD_MESSAGES_ON_STARTUP=false
    # PERCHANCE_USER_KEY=...
    # etc.
    ```

### 3. Docker Setup (Recommended for multi-bot deployment)

1.  Ensure Docker and Docker Compose are installed.
2.  Clone the repository (if not already done).
3.  Create the `common.env` file as described above.
4.  For each bot service defined in `docker-compose.yml` (e.g., `bot-nova-ai`, `bot-discordaddictamy`), create its corresponding `.env` file (e.g., `nova-ai.env`, `discordaddictamy.env`) with its `BOT_TOKEN` and `SHAPESINC_SHAPE_USERNAME`. The `docker-compose.yml` is pre-configured to mount a volume at `/data`, so the recommended `DATABASE_PATH` will work out of the box.

## Running the Bot

### Locally (Single Instance)

1.  Ensure you have set up your `.env` file in the project root as described in "Local Setup".
2.  Activate your virtual environment.
3.  Run the bot:
    ```bash
    python bot.py
    ```
    The bot will start polling for updates. Press `CTRL+C` to stop. It will create `bot_database.db` in your project root.

### With Docker Compose (Multiple Instances)

1.  Ensure you have created `common.env` and all required bot-specific `.env` files.
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
*   `/auth_shapes`: Starts the process to connect your Shapes.inc account for a personalized experience.
*   `/cancel`: Aborts a multi-step process like authentication.
*   `/newchat`: Clears the bot's side of the conversation history for the current chat/topic (respecting `SEPARATE_TOPIC_HISTORIES` setting). This helps the bot "forget" the recent local context. However, Shapes.inc manages its own context on their end; standard Shape commands like `!wack` might be more relevant for a full context reset on their platform. The bot now sends a more specific `X-Channel-Id` header that might help Shapes.inc with their history scoping.
*   `/activate`: **(Admin-only)** (Groups/Topics only) Make the bot respond to every message in the current group (general chat) or specific topic.
*   `/deactivate`: **(Admin-only)** (Groups/Topics only) Stop the bot from responding to every message in the current group/topic, reverting to default behavior (mentions, replies, free will).
*   `/imagine <prompt>`: (Experimental) Generates images based on your prompt using Bing Image Creator. Requires `BING_AUTH_COOKIE` to be set and `BingImageCreator` library to be functional.
*   `/setbingcookie <cookie_value>`: (Admin-only) Updates the Bing authentication cookie. Restricted to users listed in `ALLOWED_USERS`.

To interact with the bot, you can:
*   Send it a direct message.
*   Send it an image (with or without a caption, or as a document).
*   Send it a voice message (or as an audio document).
*   Send it a text document (`.txt`, `.md`, `.docx`, `.odt`).
*   In group chats (including those with Topics):
    *   Reply directly to one of the bot's messages.
    *   Mention the bot by its username (e.g., `@your_bot_username <your message>`).
    *   If in a group/topic where `/activate` has been used, the bot will respond to all messages in that specific context.
    *   Wait for it to interject based on its "free will" settings for that specific chat/topic (if enabled, respecting `SEPARATE_TOPIC_HISTORIES`).

## Important Notes & Known Issues

*   **Persistence & Data**: The bot now uses a SQLite database (`bot_database.db`). When using Docker, this file is stored in a named volume (`bot_data`) to ensure it persists even if containers are removed and rebuilt.
*   **Admin Controls**: The `/activate` and `/deactivate` commands are now restricted to group administrators to prevent misuse.
*   **Code Quality:** As stated, this is "quick and dirty" code. It's functional but could benefit from significant refactoring.
*   **Security:** Be very careful with your API keys, bot tokens, and `SHAPESINC_APP_ID`. Do not commit `.env` files containing secrets to public repositories.
*   **Bing Image Generation (`/imagine`):** This feature remains experimental and prone to breaking.
*   **Anime Image Generation (`generate_anime_image` tool):** This new tool uses the Perchance API. It requires a `PERCHANCE_USER_KEY`. The stability of this third-party API is not guaranteed.
*   **Document Processing:** The bot can extract text from `.txt`, `.md`, `.docx`, and `.odt` files. Images and audio files sent as documents are also processed. There's a `MAX_TEXT_FILE_SIZE` limit (default 500KB).
*   **Voice/Audio Message Processing:** The bot sends the Telegram-provided URL for voice/audio messages (and audio documents) directly to the Shapes.inc API. The API must be ableto access and process these URLs.
*   **Error Handling:** While there's a general error handler, specific errors within tool execution or API calls might not always be gracefully reported to the user. Retries for empty AI responses have been added.
*   **Telegram Topic Support Details:**
    *   The `SEPARATE_TOPIC_HISTORIES` environment variable controls whether conversation history, free will context, and `/newchat` are scoped per-topic or per-group.
    *   The bot attempts to cache topic names for more user-friendly logging and contextual prompts.
    *   The bot sends a specific `X-Channel-Id` (e.g., `chatid_topicid` or `chatid_general`) to the Shapes.inc API, which *might* help the API scope its context, depending on their implementation.
    *   If `IGNORE_OLD_MESSAGES_ON_STARTUP` is set to `true`, the bot will skip processing messages (including topic creation/edit events for caching) that occurred before it started.
*   **Typing Indicator:** Generally reliable, but minor inconsistencies might still occur in complex topic/reply scenarios.
*   **Security:** Be very careful with your API keys and bot tokens. Do not commit `.env` files containing secrets to public repositories. The whitelisting feature (`ALLOWED_USERS`, `ALLOWED_CHATS`) provides a basic level of access control.
*   **Concurrent Updates & Locking:** The bot now processes updates concurrently (`app_builder.concurrent_updates(True)`). To prevent race conditions within the same conversation, a locking mechanism (`processing_locks`) is used, ensuring that messages for a specific chat (or topic, if `SEPARATE_TOPIC_HISTORIES` is true) are handled one by one.

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