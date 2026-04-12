import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import os

# Set dummy environment variables for initialization
os.environ["GEMINI_API_KEY"] = "dummy_key"
os.environ["OWM_API_KEY"] = "dummy_key"

from src.gemini import Gemini

@pytest.mark.asyncio
async def test_send_message_no_function_call():
    # Setup
    with patch("src.gemini.genai.Client"):
        gemini = Gemini()
    mock_chat = AsyncMock()

    # Mock response with no function call
    mock_response = MagicMock()
    mock_response.text = "Hello, I am Ahri!"
    mock_response.candidates = [
        MagicMock(content=MagicMock(parts=[MagicMock(function_call=None)]))
    ]
    mock_chat.send_message.return_value = mock_response

    # Execute
    result = await gemini.send_message("User: Hi", mock_chat)

    # Assert
    assert result == "Hello, I am Ahri!"
    mock_chat.send_message.assert_called_once_with("User: Hi")
    # Verify no second call was made
    assert mock_chat.send_message.call_count == 1

@pytest.mark.asyncio
async def test_send_message_with_function_call():
    # Setup
    with patch("src.gemini.genai.Client"):
        gemini = Gemini()
    mock_chat = AsyncMock()

    # Mock response with a function call
    mock_function_call = MagicMock()
    mock_function_call.name = "get_date_time"

    mock_response = MagicMock()
    mock_response.candidates = [
        MagicMock(content=MagicMock(parts=[MagicMock(function_call=mock_function_call)]))
    ]
    mock_chat.send_message.return_value = mock_response

    # Mock plugin manager response
    mock_plugin_response = MagicMock()
    mock_plugin_response.text = "The current time is 12:00 PM"

    # Note: Using the private attribute access since it's what we want to mock
    with patch.object(gemini, "_Gemini__plugin_manager", new_callable=AsyncMock) as mock_plugin_manager:
        mock_plugin_manager.get_function_response.return_value = mock_plugin_response

        # Execute
        result = await gemini.send_message("User: What time is it?", mock_chat)

        # Assert
        assert result == "The current time is 12:00 PM"
        mock_plugin_manager.get_function_response.assert_called_once_with(mock_function_call, mock_chat)

@pytest.mark.asyncio
async def test_get_chat_with_user_name():
    # Setup
    with patch("src.gemini.genai.Client") as mock_client_class:
        mock_aio_client = MagicMock()
        mock_client_class.return_value.aio = mock_aio_client
        with patch.dict(os.environ, {"GEMINI_MODEL_NAME": "gemini-2.5-flash"}):
            gemini = Gemini()

    mock_history = [{"role": "user", "parts": [{"text": "Hello"}]}]
    user_name = "Mario"

    # Execute
    gemini.get_chat(history=mock_history, user_name=user_name)

    # Assert
    mock_aio_client.chats.create.assert_called_once()
    args, kwargs = mock_aio_client.chats.create.call_args
    assert kwargs['model'] == "gemini-2.5-flash"
    assert kwargs['history'] == mock_history
    assert "Mario" in kwargs['config'].system_instruction

@pytest.mark.asyncio
async def test_get_chat_father_by_username():
    # Setup
    with patch("src.gemini.genai.Client") as mock_client_class:
        mock_aio_client = MagicMock()
        mock_client_class.return_value.aio = mock_aio_client
        gemini = Gemini()

    # Execute
    gemini.get_chat(history=[], user_name="SomeName", username="shiro_mb")

    # Assert
    args, kwargs = mock_aio_client.chats.create.call_args
    assert "TUO PADRE" in kwargs['config'].system_instruction

@pytest.mark.asyncio
async def test_get_chat_boyfriend_by_username():
    # Setup
    with patch("src.gemini.genai.Client") as mock_client_class:
        mock_aio_client = MagicMock()
        mock_client_class.return_value.aio = mock_aio_client
        gemini = Gemini()

    # Execute
    gemini.get_chat(history=[], user_name="SomeName", username="antonydpk")

    # Assert
    args, kwargs = mock_aio_client.chats.create.call_args
    assert "IL TUO FIDANZATO" in kwargs['config'].system_instruction

@pytest.mark.asyncio
async def test_get_chat_boyfriend_by_name_antonio():
    # Setup
    with patch("src.gemini.genai.Client") as mock_client_class:
        mock_aio_client = MagicMock()
        mock_client_class.return_value.aio = mock_aio_client
        gemini = Gemini()

    # Execute
    gemini.get_chat(history=[], user_name="Antonio")

    # Assert
    args, kwargs = mock_aio_client.chats.create.call_args
    assert "IL TUO FIDANZATO" in kwargs['config'].system_instruction
