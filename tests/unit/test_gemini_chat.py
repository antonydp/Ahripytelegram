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
    result = await gemini.send_message("Hi", mock_chat)

    # Assert
    assert result == "Hello, I am Ahri!"
    mock_chat.send_message.assert_called_once_with("Hi")
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
        result = await gemini.send_message("What time is it?", mock_chat)

        # Assert
        assert result == "The current time is 12:00 PM"
        mock_plugin_manager.get_function_response.assert_called_once_with(mock_function_call, mock_chat)
