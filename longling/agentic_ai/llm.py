# coding: utf-8
# tongshiwei@2025/3/19

import logging
from typing import Literal

import openai

logger = logging.getLogger(__name__)


class OpenaiLLM(object):
    def __init__(self, endpoint, api_key, api_version):
        self.client = openai.AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            max_retries=3,
            timeout=30,
        )

    @classmethod
    def as_messages(cls, content, role: Literal["user", "assistant", "system"]) -> dict:
        return {"role": role, "content": content}

    def __call__(self, messages, model, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                timeout=30,
                extra_body={
                    "use_cache": True
                },
            )
            return response.choices[0].message.content
        except openai.APIStatusError as e:
            logger.error(f"status code:{e.status_code} request_id={e.request_id} response={e.response.text}")
        except Exception as e:
            logger.exception(e)
