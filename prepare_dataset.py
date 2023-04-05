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
import datetime
import json
import os
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Tuple

import pandas as pd
from loguru import logger
from tqdm import tqdm

# pd.set_option('display.max_colwidth', None)


def load_chats(path: str) -> Tuple[List[Dict], int]:
    with open(path, 'r') as f:
        chats = json.load(f)["chats"]["list"]
    target_id = next(chat for chat in chats if chat["type"] == "saved_messages")["id"]
    logger.info(f"Found {len(chats)} chats")
    logger.info(f"Preparing dataset for user with id {target_id}")
    return chats, target_id


def load_json(path: str) -> dict:
    Path(".cache").mkdir(exist_ok=True)

    cache = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            cache = json.load(f)
        logger.info(f"Found json in {path} with {len(cache.keys())} keys!")
    return cache


def write_json(cache: dict, path: str):
    with open(path, "w", encoding='utf8') as f:
        json.dump(cache, f, indent=4, ensure_ascii=False)


def replace_names_by_original(id_list: List[int], cache_path="./.cache/users.json") -> List[str]:
    """Returns None values for unknown ids."""
    from telethon.sync import TelegramClient

    names = []

    cache = load_json(cache_path)
    with TelegramClient(
        './.cache/personal',
        os.environ["TELETHON_API_ID_PERSONAL"],
        os.environ["TELETHON_API_HASH_PERSONAL"],
    ) as client_personal, TelegramClient(
        './.cache/random',
        os.environ["TELETHON_API_ID_RANDOM"],
        os.environ["TELETHON_API_HASH_RANDOM"],
    ) as client_random:
        for user_id in tqdm(id_list):
            if str(user_id) not in cache:
                try:
                    user_personal = client_personal.get_entity(user_id)
                    username = user_personal.username
                    user = client_random.get_entity(username)
                    name = (
                        user.first_name + " " + user.last_name
                        if user.last_name
                        else user.first_name
                    )
                except (ValueError, TypeError) as e:
                    name = username = None
                cache[str(user_id)] = {"username": username, "name": name}
                write_json(cache, cache_path)
            names.append(cache[str(user_id)]["name"])

    logger.info(f"Replaced {len(set(names))} names with original ones.")
    return names


def translate_names(
    names: List[str], lang="ru", cache_path="./.cache/translation.json"
) -> List[str]:
    import deepl

    translator = deepl.Translator(os.environ["DEEPL_API_KEY"])

    res = []

    cache = load_json(cache_path)
    for name in tqdm(names):
        if name not in cache:
            cache[name] = translator.translate_text(name, target_lang=lang).text
            write_json(cache, cache_path)
        res.append(cache[name])
    logger.info(f"Translated {len(set(res))} names.")
    return res


def extract_messages(chats: List[Dict], last_x_months: int) -> pd.DataFrame:
    rows = []
    for chat in chats:
        # skip saved_messages
        if "name" not in chat:
            continue
        row = {"name": chat["name"], "type": chat["type"], "id": chat["id"]}
        for message in chat["messages"]:
            if (
                message["type"] == "message"
                and message["text_entities"]
                and "forwarded_from" not in message
            ):
                rows.append(
                    row
                    | {
                        "date": message["date"],
                        "from": message["from"],
                        "from_id": int(
                            message["from_id"].replace("user", "").replace("channel", "")
                        ),
                        "text": "".join(entity["text"] for entity in message["text_entities"]),
                    }
                )
    messages = pd.DataFrame(rows)
    messages["date"] = pd.to_datetime(messages["date"])
    messages = messages[
        datetime.datetime.now() - messages["date"] <= datetime.timedelta(days=30 * last_x_months)
    ]

    logger.info(f"Extracted {len(messages)} messages")

    return messages


def extract_sessions(
    messages: pd.DataFrame,
    target_id: int,
    session_minutes_threshold: int,
    concat_one_user_messages_delimeter: str,
):
    # filter out chats where target_id is not present
    messages = messages[
        messages["name"].isin(messages[messages["from_id"] == target_id]["name"].unique())
    ]
    logger.info(f"Total number of chats: {messages['name'].nunique()}")

    # add session columns (nearby messages go to the same session)
    messages["session_id"] = (
        (messages["date"].diff() > pd.Timedelta(minutes=session_minutes_threshold))
        | (messages["name"].ne(messages["name"].shift()))
    ).cumsum()

    # merge messages from single user
    messages['user_change'] = messages['from_id'].ne(messages['from_id'].shift()).cumsum()
    messages = (
        messages.groupby(['session_id', 'user_change'])
        .agg(
            {
                'name': 'first',
                'from': 'first',
                'from_id': 'first',
                'text': lambda x: concat_one_user_messages_delimeter.join(x),
            }
        )
        .reset_index()
    )
    messages = messages.drop('user_change', axis=1)

    # filter session with 'None' - from senders
    sessions_with_none = messages[messages['from'].isna()]['session_id'].unique()
    messages = messages[~messages['session_id'].isin(sessions_with_none)]

    # filter session where no target_id is present
    sessions_with_target = messages[messages['from_id'] == target_id]['session_id'].unique()
    messages = messages[messages['session_id'].isin(sessions_with_target)]

    # filter sessions with length == 1 or length == 2 and target_id msg is first (because no context for train)
    messages = messages.groupby('session_id').filter(
        lambda group: len(group) > 2 or (len(group) == 2 and group.iloc[0]['from_id'] != target_id)
    )

    logger.info(f"Total number of sessions: {messages['session_id'].nunique()}")
    logger.info(
        f"Average number of messages per session: {round(messages.groupby('session_id').size().mean(), 2)}",
    )

    return messages


def apply_name_transforms(messages: pd.DataFrame) -> pd.DataFrame:
    # replace names by original ones
    if all(
        x in os.environ
        for x in [
            "TELETHON_API_ID_PERSONAL",
            "TELETHON_API_HASH_PERSONAL",
            "TELETHON_API_ID_RANDOM",
            "TELETHON_API_HASH_RANDOM",
        ]
    ):
        names = replace_names_by_original(messages["from_id"])
        messages["from"] = [
            names[index] if names[index] is not None else row['from']
            for index, (_, row) in enumerate(messages.iterrows())
        ]

    # translate names
    if "DEEPL_API_KEY" in os.environ:
        messages["from"] = translate_names(messages["from"])

    return messages


def prepare_dataset(
    messages: pd.DataFrame, target_id: int, concat_messages_delimeter: str
) -> List[Dict[str, str]]:
    dataset = []

    sessions = messages['session_id'].unique()
    for session in tqdm(sessions):
        session_df = messages[messages['session_id'] == session]
        message_history = ''
        for idx, row in session_df.iterrows():
            if message_history and row['from_id'] == target_id:
                dataset.append(
                    dict(context=message_history + f"{row['from']}: ", answer=row['text'])
                )
            message_history += f"{row['from']}: {row['text']}{concat_messages_delimeter}"

    logger.info(f"Total number of samples: {len(dataset)}")

    return dataset


def parse_newline(value):
    """Parse args with `\n` in them without duplicating `\` symbol."""
    try:
        return ast.literal_eval(f"'{value}'")
    except (ValueError, SyntaxError):
        return value


class CustomHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Proper displaying default values with `\n` in them."""

    def _get_help_string(self, action):
        help_str = action.help
        if action.default is not argparse.SUPPRESS:
            defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
            if action.option_strings or action.nargs in defaulting_nargs:
                help_str += f" (default: {action.default!r})"
        return help_str


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=description, formatter_class=CustomHelpFormatter)
    parser.add_argument(
        '-i',
        '--input',
        help='Path to result.json (output from telegram export)',
        required=True,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '-o', '--output', help='Output file', required=True, default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--last-x-months', help='Number of last months to use messages from', default=24
    )
    parser.add_argument(
        '--session-minutes-threshold',
        help='Threshold in minutes where messages will belong to the same session',
        default=10,
    )
    parser.add_argument(
        '--concat-one-user-messages-delimeter',
        type=parse_newline,
        help='Users might type several messages one after each other. They are concatenated using this delimeter',
        default="\n>>> ",
    )
    parser.add_argument(
        '--concat-messages-delimeter',
        type=parse_newline,
        help='How to concat messages from different users',
        default="\n\n",
    )
    args = vars(parser.parse_args())
    logger.info(f"Args:\n{pformat(args)}")
    return args


def main():
    args = parse_args()
    chats, target_id = load_chats(args["input"])
    messages = extract_messages(chats, args["last_x_months"])
    messages = extract_sessions(
        messages,
        target_id,
        args["session_minutes_threshold"],
        args["concat_one_user_messages_delimeter"],
    )
    messages = apply_name_transforms(messages)
    dataset = prepare_dataset(messages, target_id, args["concat_messages_delimeter"])
    write_json(dataset, args["output"])


if __name__ == '__main__':
    main()
