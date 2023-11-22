import json
from datetime import datetime, timedelta
from typing import List, Literal, Optional, Tuple

import fire
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm


class Message(BaseModel):
    date: datetime
    author: str
    text: str


class Chat(BaseModel):
    name: str
    type: Literal["personal_chat", "private_group", "private_supergroup"]
    messages: List[Message]
    sessions: Optional[List[List[Message]]] = []


def load_chats(path: str) -> Tuple[List[Chat], Tuple[int | None, str | None]]:
    chats: List[Chat] = []
    target_id, target_name = None, None
    logger.info(f"Loading chats from '{path}'...")
    with open(path, "r") as f:
        for chat in json.load(f)["chats"]["list"]:
            # It means we encountered 'Saved Messages', from which
            # we can extract id and a name of a target person
            if "name" not in chat:
                target_id = int(chat["id"])
                target_name = str(next(msg for msg in chat["messages"] if msg["from_id"] == f"user{target_id}")["from"])
            # If chat does not contain name that means we
            # encountered "Deleted Account"
            elif chat["name"]:
                messages = [
                    Message(
                        date=msg["date"],
                        author=msg["from"],
                        text="".join([text_entity["text"] for text_entity in msg["text_entities"]])
                        + msg.get("sticker_emoji", ""),
                    )
                    for msg in chat["messages"]
                    if "from" in msg and msg["from"] and (msg["text_entities"] or "sticker_emoji" in msg)
                ]
                if messages:
                    chat = Chat(name=chat["name"], type=chat["type"], messages=messages)
                    chats.append(chat)
    logger.info(f"Found {len(chats)} chats in file '{path}'")
    if not target_name:
        logger.warning(f"Was not able to detect target name from 'Saved Messages'!")
    return chats, (target_id, target_name)


def transform_chats(
    input: str,
    output: str,
    target_name: str | None = None,
    last_x_months: int = 24,
    session_minutes_threshold: int = 10,
    concat_one_user_messages_delimeter: str = "\n>>> ",
):
    """
    Transforms chat histories from .json telegram export to .json with a list of sessions.
    Session is a list of messages, where each message is a dict with fields 'author' and 'text'.

    :param input: Path to .json telegram export, usually called result.json
    :param output: Path to output .json file
    :param target_name: The name of the person to target. This person will be present in every session. If empty, will be tried to be detected from "Saved Messages"
    :param last_x_months: Number of last months to use messages from
    :param session_minutes_threshold: Threshold in minutes where messages will belong to the same session
    :param concat_one_user_messages_delimeter: Users might type several messages one after each other. They are concatenated using this delimeter
    """
    chats, (_, extracted_target_name) = load_chats(input)
    if not target_name:
        target_name = extracted_target_name
    logger.info(f"Preparing dataset for user with name '{target_name}'...")

    # Dropping all messages which are older than given date
    for chat in chats:
        chat.messages = [msg for msg in chat.messages if msg.date > datetime.now() - timedelta(days=last_x_months * 30)]
    chats = [chat for chat in chats if chat.messages]
    logger.info(f"After filtering by date, there are {len(chats)} chats left")

    # Create sessions for each chat by combining messages within
    # session_minutes_threshold into one session
    for chat in chats:
        sessions = []
        current_session = []
        for msg in chat.messages:
            if not current_session or (msg.date - current_session[-1].date).seconds / 60 < session_minutes_threshold:
                current_session.append(msg)
            else:
                sessions.append(current_session)
                current_session = [msg]
        if current_session:
            sessions.append(current_session)
        chat.sessions = sessions
    logger.info(f"Combined messages into sessions")

    # Combine consecutive messages from single user into one message
    for chat in chats:
        sessions = []
        for session in chat.sessions:
            current_session = []
            current_message = session[0]
            current_message.text = concat_one_user_messages_delimeter.lstrip() + current_message.text
            for msg in session[1:]:
                if msg.author == current_message.author:
                    current_message.text += concat_one_user_messages_delimeter + msg.text
                else:
                    current_session.append(current_message)
                    current_message = msg
                    current_message.text = concat_one_user_messages_delimeter.lstrip() + current_message.text
            current_session.append(current_message)
            sessions.append(current_session)
        chat.sessions = sessions
    logger.info(f"Combined consecutive messages from single user into one message")

    # Only leave sessions which have target_name in them
    # (1st does not count as we can't use it for training)
    for chat in chats:
        chat.sessions = [session for session in chat.sessions if any(msg.author == target_name for msg in session[1:])]

    # # Cut off messages in each session by last message from target_name
    # for chat in chats:
    #     for session in chat.sessions:
    #         session[:] = session[: max(i for i, msg in enumerate(session) if msg.author == target_name) + 1]

    # Remove date from messages
    for chat in chats:
        for session in chat.sessions:
            for msg in session:
                del msg.date

    # Write sessions to jsonl
    all_sessions = []
    for chat in chats:
        for session in chat.sessions:
            all_sessions.append(session)
    with open(output, "w") as f:
        json.dump(
            [[{"author": msg.author, "text": msg.text} for msg in session] for session in all_sessions],
            f,
            indent=4,
            ensure_ascii=False,
        )
    logger.info(
        f"Took {len(all_sessions)} sessions from {len(chats)} chats and wrote them to '{output}'. Average session length is {round(sum(len(session) for session in all_sessions) / len(all_sessions), 2)} messages"
    )


if __name__ == "__main__":
    fire.Fire(transform_chats)
