"""Messages examples:

Usual text message:    
```
 {
  "id": 123,
  "type": "message",
  "date": "2023-10-31T15:23:38",
  "date_unixtime": "1698746018",
  "from": "Username",
  "from_id": "user123",
  "text": "ты где?",
  "text_entities": [
   {
    "type": "plain",
    "text": "ты где?"
   }
  ]
 }
 ```

Multiple text entities:
```
 {
  "id": 345,
  "type": "message",
  "date": "2023-10-25T01:56:50",
  "date_unixtime": "1698179210",
  "from": "Username",
  "from_id": "user456",
  "text": [
   "California suspends GM Cruise's autonomous vehicle deployment | Hacker News\n",
   {
    "type": "link",
    "text": "https://news.ycombinator.com/item?id=38002752"
   }
  ],
  "text_entities": [
   {
    "type": "plain",
    "text": "California suspends GM Cruise's autonomous vehicle deployment | Hacker News\n"
   },
   {
    "type": "link",
    "text": "https://news.ycombinator.com/item?id=38002752"
   }
  ]
 }
 ```

 Sticker:
 ```
 {
  "id": 789,
  "type": "message",
  "date": "2023-10-30T23:24:20",
  "date_unixtime": "1698688460",
  "from": "Username",
  "from_id": "user789",
  "file": "(File not included. Change data exporting settings to download.)",
  "thumbnail": "(File not included. Change data exporting settings to download.)",
  "media_type": "sticker",
  "sticker_emoji": "🤗",
  "width": 512,
  "height": 501,
  "text": "",
  "text_entities": []
 }
 ```
"""


description = """Convertes the output of telegram messages export to context-answer dataset format.

To export data, go to 'Setting' -> 'Advanced' -> 'Export Telegram Data' and unselect everything except 'Personal chats' and 'Private groups' (don't select 'Only my messages there'). As output format choose 'Machine-readable JSON'.

There are few additional thing happen if you have appropriate ENVs set:
1. result.json will contain names as they are written in your contacts, but on inference telegram bot will not have access to your contacts and it will see the names of the users as they are written in their profiles. Why this is a problem? Imagine you have your friend named 'bruh' in your contact. The model will learn how to communicate with 'bruh' but on inference it will see completely different name like 'John Smith' so it will not be as intertaining as it could've been. To adress this issue, we want to replace all names by their true values.
   But we face a problem: in result.json you will see only user id's as identifiers, but telegram does not allow you to get user information from their ids if you hadn't communicated previously. So we will need your main account to extract profile information first.
   Here comes another problem: if you will retrieve user name in the way above, it will return you the exactly same handle 'bruh', which you personally have for them. But we at least know username at this step, so utilizing another telegram account which is not aware of your contacts, we are able to retrieve the actual names.
   So, to make this work, you need to create two apps in https://my.telegram.org/apps and set the following ENVs:
   - TELETHON_API_ID_PERSONAL
   - TELETHON_API_HASH_PERSONAL
   - TELETHON_API_ID_RANDOM
   - TELETHON_API_HASH_RANDOM
2. If you have set DEEPL_API_KEY, the names will be translated to the language specified in `translate_names` function. This is useful when you are training on your native language so the translation of a names helps model to not switch language every time (because names are usually in english). If you have it enabled, remember to translate incoming names on inference side too!"""

import argparse
import ast
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Literal, Optional, Tuple

import fire
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from tqdm import tqdm

# pd.set_option('display.max_colwidth', None)


class Message(BaseModel):
    date: datetime
    author: str
    text: str


class Chat(BaseModel):
    name: str
    type: Literal["personal_chat", "private_group", "private_supergroup"]
    messages: List[Message]
    sessions: Optional[List[List[Message]]] = []


def get_chats(path: str) -> Tuple[List[Chat], Tuple[int, str]]:
    chats: List[Chat] = []
    target_id, target_name = -1, ""
    with open(path, "r") as f:
        for chat in tqdm(json.load(f)["chats"]["list"], desc="Loading chats..."):
            # saved messages; extracting id and a name
            # of a target person
            if "name" not in chat:
                target_id = int(chat["id"])
                target_name = str(next(msg for msg in chat["messages"] if msg["from_id"] == f"user{target_id}")["from"])
            # if it is not "Deleted Account"
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
    logger.info(f"Preparing dataset for user with name '{target_name}'")
    return (chats, (target_id, target_name))


def parse(
    input: str,
    output: str,
    target_name: str | None = None,
    last_x_months: int = 24,
    session_minutes_threshold: int = 10,
    concat_one_user_messages_delimeter: str = "\n>>> ",
):
    """
    :param input: Path to result.json (output from telegram export)
    :param output: Output file
    :param target_name: The name of the person to greet. If empty, will be tried to be detected from "Saved Messages"
    :param last_x_months: Number of last months to use messages from
    :param session_minutes_threshold: Threshold in minutes where messages will belong to the same session
    :param concat_one_user_messages_delimeter: Users might type several messages one after each other. They are concatenated using this delimeter
    """
    (chats, (target_id, target_name)) = get_chats(input)

    # drop all messages which are older than date
    for chat in chats:
        chat.messages = [msg for msg in chat.messages if msg.date > datetime.now() - timedelta(days=last_x_months * 30)]
    chats = [chat for chat in chats if chat.messages]
    logger.info(f"After filtering by date, there are {len(chats)} chats left")

    # greedy create sessions for each chat by combining messages withing
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

    # greedy combine messages from single user into one message
    for chat in chats:
        sessions = []
        current_user = None
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
    logger.info(f"Combined messages from single user into one message")

    # only leave sessions which have target_name in them
    # (1st does not count as we can't use it for training)
    for chat in chats:
        chat.sessions = [session for session in chat.sessions if any(msg.author == target_name for msg in session[1:])]

    # # cut off messages in each session by last message from target_name
    # for chat in chats:
    #     for session in chat.sessions:
    #         session[:] = session[: max(i for i, msg in enumerate(session) if msg.author == target_name) + 1]

    # remove date from messages
    for chat in chats:
        for session in chat.sessions:
            for msg in session:
                del msg.date

    # write sessions to jsonl
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
    logger.info(f"Took {len(all_sessions)} sessions from {len(chats)} chats and wrote them to {output}. Average session length is {sum(len(session) for session in all_sessions) / len(all_sessions)} messages")

    # logger.info(pformat((next(chat for chat in chats if "Taya" in chat.name)).sessions[-3]))


if __name__ == "__main__":
    fire.Fire(parse)


# def load_json(path: str) -> dict:
#     Path(".cache").mkdir(exist_ok=True)

#     cache = {}
#     if os.path.exists(path):
#         with open(path, "r") as f:
#             cache = json.load(f)
#         logger.info(f"Found json in {path} with {len(cache.keys())} keys!")
#     return cache


# def write_json(cache: dict, path: str):
#     with open(path, "w", encoding="utf8") as f:
#         json.dump(cache, f, indent=4, ensure_ascii=False)


# def replace_names_by_original(id_list: List[int], cache_path="./.cache/users.json") -> List[str]:
#     """Returns None values for unknown ids."""
#     from telethon.sync import TelegramClient

#     names = []

#     cache = load_json(cache_path)
#     with TelegramClient(
#         "./.cache/personal",
#         int(os.environ["TELETHON_API_ID_PERSONAL"]),
#         os.environ["TELETHON_API_HASH_PERSONAL"],
#     ) as client_personal, TelegramClient(
#         "./.cache/random",
#         int(os.environ["TELETHON_API_ID_RANDOM"]),
#         os.environ["TELETHON_API_HASH_RANDOM"],
#     ) as client_random:
#         for user_id in tqdm(id_list):
#             if str(user_id) not in cache:
#                 try:
#                     user_personal = client_personal.get_entity(user_id)
#                     username = user_personal.username
#                     user = client_random.get_entity(username)
#                     name = user.first_name + " " + user.last_name if user.last_name else user.first_name
#                 except (ValueError, TypeError):
#                     name = username = None
#                 cache[str(user_id)] = {"username": username, "name": name}
#                 write_json(cache, cache_path)
#             names.append(cache[str(user_id)]["name"])

#     logger.info(f"Replaced {len(set(names))} names with original ones.")
#     return names


# def translate_names(names: List[str], lang="ru", cache_path="./.cache/translation.json") -> List[str]:
#     import deepl

#     translator = deepl.Translator(os.environ["DEEPL_API_KEY"])

#     res = []

#     cache = load_json(cache_path)
#     for name in tqdm(names):
#         if name not in cache:
#             cache[name] = translator.translate_text(name, target_lang=lang).text
#             write_json(cache, cache_path)
#         res.append(cache[name])
#     logger.info(f"Translated {len(set(res))} names.")
#     return res


# def extract_messages(chats: List[Dict], last_x_months: int) -> pd.DataFrame:
#     rows = []
#     for chat in chats:
#         # skip saved_messages
#         if "name" not in chat:
#             continue
#         row = {"name": chat["name"], "type": chat["type"], "id": chat["id"]}
#         for message in chat["messages"]:
#             if message["type"] == "message" and (message["text_entities"] or "sticker_emoji" in message) and "forwarded_from" not in message:
#                 rows.append(
#                     row
#                     | {
#                         "date": message["date"],
#                         "from": message["from"],
#                         "from_id": int(message["from_id"].replace("user", "").replace("channel", "")),
#                         "text": "".join(entity["text"] for entity in message["text_entities"]) + message.get("sticker_emoji", ""),
#                     }
#                 )
#     messages = pd.DataFrame(rows)
#     messages["date"] = pd.to_datetime(messages["date"])
#     messages = messages[datetime.now() - messages["date"] <= timedelta(days=30 * last_x_months)]

#     logger.info(f"Extracted {len(messages)} messages")

#     return messages


# def extract_sessions(
#     messages: pd.DataFrame,
#     target_id: int,
#     session_minutes_threshold: int,
#     concat_one_user_messages_delimeter: str,
# ):
#     # filter out chats where target_id is not present
#     messages = messages[messages["name"].isin(messages[messages["from_id"] == target_id]["name"].unique())]
#     logger.info(f"Total number of chats: {messages['name'].nunique()}")

#     # add session columns (nearby messages go to the same session)
#     messages["session_id"] = (
#         (messages["date"].diff() > pd.Timedelta(minutes=session_minutes_threshold))
#         | (messages["name"].ne(messages["name"].shift()))
#     ).cumsum()

#     # merge messages from single user
#     messages["user_change"] = messages["from_id"].ne(messages["from_id"].shift()).cumsum()
#     messages = (
#         messages.groupby(["session_id", "user_change"])
#         .agg(
#             {
#                 "name": "first",
#                 "from": "first",
#                 "from_id": "first",
#                 "text": lambda x: concat_one_user_messages_delimeter.join(x),
#             }
#         )
#         .reset_index()
#     )
#     messages = messages.drop("user_change", axis=1)

#     # filter session with 'None' - from senders
#     sessions_with_none = messages[messages["from"].isna()]["session_id"].unique()
#     messages = messages[~messages["session_id"].isin(sessions_with_none)]

#     # filter session where no target_id is present
#     sessions_with_target = messages[messages["from_id"] == target_id]["session_id"].unique()
#     messages = messages[messages["session_id"].isin(sessions_with_target)]

#     # filter sessions with length == 1 or length == 2 and target_id msg is first (because no context for train)
#     messages = messages.groupby("session_id").filter(
#         lambda group: len(group) > 2 or (len(group) == 2 and group.iloc[0]["from_id"] != target_id)
#     )

#     logger.info(f"Total number of sessions: {messages['session_id'].nunique()}")
#     logger.info(
#         f"Average number of messages per session: {round(messages.groupby('session_id').size().mean(), 2)}",
#     )

#     return messages


# def apply_name_transforms(messages: pd.DataFrame) -> pd.DataFrame:
#     # replace names by original ones
#     if all(
#         x in os.environ
#         for x in [
#             "TELETHON_API_ID_PERSONAL",
#             "TELETHON_API_HASH_PERSONAL",
#             "TELETHON_API_ID_RANDOM",
#             "TELETHON_API_HASH_RANDOM",
#         ]
#     ):
#         names = replace_names_by_original(messages["from_id"])
#         messages["from"] = [
#             names[index] if names[index] is not None else row["from"]
#             for index, (_, row) in enumerate(messages.iterrows())
#         ]

#     # translate names
#     if "DEEPL_API_KEY" in os.environ:
#         messages["from"] = translate_names(messages["from"])

#     return messages


# def prepare_dataset(messages: pd.DataFrame, target_id: int, concat_messages_delimeter: str) -> List[Dict[str, str]]:
#     dataset = []

#     sessions = messages["session_id"].unique()
#     for session in tqdm(sessions):
#         session_df = messages[messages["session_id"] == session]
#         message_history = ""
#         for idx, row in session_df.iterrows():
#             if message_history and row["from_id"] == target_id:
#                 dataset.append(dict(context=message_history + f"{row['from']}:", answer=row["text"]))
#             message_history += f"{row['from']}: {row['text']}{concat_messages_delimeter}"

#     logger.info(f"Total number of samples: {len(dataset)}")

#     return dataset


# def parse_newline(value):
#     """Parse args with `\n` in them without duplicating `\` symbol."""
#     try:
#         return ast.literal_eval(f"'{value}'")
#     except (ValueError, SyntaxError):
#         return value


# class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
#     """Proper displaying default values with `\n` in them."""

#     def _get_help_string(self, action):
#         help_str = action.help
#         if action.default is not argparse.SUPPRESS:
#             defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
#             if action.option_strings or action.nargs in defaulting_nargs:
#                 help_str += f" (default: {action.default!r})"
#         return help_str


# def parse_args() -> Dict[str, Any]:
#     parser = argparse.ArgumentParser(description=description, formatter_class=CustomHelpFormatter)
#     parser.add_argument(
#         "-i",
#         "--input",
#         help="Path to result.json (output from telegram export)",
#         required=True,
#         default=argparse.SUPPRESS,
#     )
#     parser.add_argument("-o", "--output", help="Output file", required=True, default=argparse.SUPPRESS)
#     parser.add_argument("--last-x-months", help="Number of last months to use messages from", default=24)
#     parser.add_argument(
#         "--session-minutes-threshold",
#         help="Threshold in minutes where messages will belong to the same session",
#         default=10,
#     )
#     parser.add_argument(
#         "--concat-one-user-messages-delimeter",
#         type=parse_newline,
#         help="Users might type several messages one after each other. They are concatenated using this delimeter",
#         default="\n>>> ",
#     )
#     parser.add_argument(
#         "--concat-messages-delimeter",
#         type=parse_newline,
#         help="How to concat messages from different users",
#         default="\n\n",
#     )
#     args = vars(parser.parse_args())
#     logger.info(f"Args:\n{pformat(args)}")
#     return args


# def main():
#     args = parse_args()
#     chats, target_id = load_chats(args["input"])
#     messages = extract_messages(chats, args["last_x_months"])
#     messages = extract_sessions(
#         messages,
#         target_id,
#         args["session_minutes_threshold"],
#         args["concat_one_user_messages_delimeter"],
#     )
#     messages = apply_name_transforms(messages)
#     dataset = prepare_dataset(messages, target_id, args["concat_messages_delimeter"])
#     write_json(dataset, args["output"])


# if __name__ == "__main__":
#     main()
