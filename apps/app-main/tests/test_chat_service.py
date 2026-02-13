"""Tests for ChatService."""

import pytest

from app_main.services.chat_service import ChatService
from tests.conftest import make_chat_message, make_chat_session


class TestChatServiceGetSession:

    @pytest.mark.asyncio
    async def test_get_session(self, chat_session_repo, chat_message_repo):
        session = make_chat_session()
        chat_session_repo.get.return_value = session
        service = ChatService(chat_session_repo, chat_message_repo)

        result = await service.get_session("chat_session:test1")

        chat_session_repo.get.assert_called_once_with("chat_session:test1")
        assert result == session


class TestChatServiceCreateSession:

    @pytest.mark.asyncio
    async def test_create_empty_session(self, chat_session_repo, chat_message_repo):
        session = make_chat_session()
        chat_session_repo.create.return_value = session
        service = ChatService(chat_session_repo, chat_message_repo)

        result = await service.create_session()

        chat_session_repo.create.assert_called_once_with({})
        assert result == session

    @pytest.mark.asyncio
    async def test_create_with_notebook(self, chat_session_repo, chat_message_repo):
        session = make_chat_session()
        chat_session_repo.create.return_value = session
        service = ChatService(chat_session_repo, chat_message_repo)

        await service.create_session(notebook_id="notebook:1")

        chat_session_repo.relate_to_notebook.assert_called_once_with(
            "chat_session:test1", "notebook:1"
        )

    @pytest.mark.asyncio
    async def test_create_with_source(self, chat_session_repo, chat_message_repo):
        session = make_chat_session()
        chat_session_repo.create.return_value = session
        service = ChatService(chat_session_repo, chat_message_repo)

        await service.create_session(source_id="source:1")

        chat_session_repo.relate_to_source.assert_called_once_with(
            "chat_session:test1", "source:1"
        )


class TestChatServiceDeleteSession:

    @pytest.mark.asyncio
    async def test_delete(self, chat_session_repo, chat_message_repo):
        chat_session_repo.delete.return_value = True
        service = ChatService(chat_session_repo, chat_message_repo)

        assert await service.delete_session("session:1") is True


class TestChatServiceMessages:

    @pytest.mark.asyncio
    async def test_get_messages(self, chat_session_repo, chat_message_repo):
        msgs = [make_chat_message()]
        chat_message_repo.get_session_messages.return_value = msgs
        service = ChatService(chat_session_repo, chat_message_repo)

        result = await service.get_messages("session:1", limit=10)

        chat_message_repo.get_session_messages.assert_called_once_with("session:1", 10)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_add_message(self, chat_session_repo, chat_message_repo):
        msg = make_chat_message(role="assistant", content="Hi!")
        chat_message_repo.create.return_value = msg
        service = ChatService(chat_session_repo, chat_message_repo)

        result = await service.add_message("session:1", "assistant", "Hi!")

        chat_message_repo.create.assert_called_once_with({
            "session": "session:1",
            "role": "assistant",
            "content": "Hi!",
        })
        assert result == msg
