import asyncio
import logging
import os
import json
import uuid
import requests
from requests.exceptions import HTTPError

from tqdm import tqdm
from openai import OpenAI
import torch
import numpy as np
import pandas as pd
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MAX_RETRY = 2

def llama_process_sync(prompt, model_name, client, base64_image=None):
    """
    同步请求，支持:
     - 如果 model_name 为 gcp-claude37-sonnet-thinking，走专门的 ByteDance thinking API，
       同时支持多模态（文本+图片）请求，使用与 Anthropic 示例一致的消息体格式
     - 否则沿用原有 OpenAI/AzureOpenAI 客户端逻辑
    """
    # —— 专门分支：gcp-claude37-sonnet-thinking —— #
    if model_name == "gcp-claude37-sonnet-thinking":
        url = client.base_url if hasattr(client, 'base_url') else client.azure_endpoint
        headers = {
            "Content-Type": "application/json",
            "X-TT-LOGID": str(uuid.uuid4()),
            "caller": "liniuniu",
        }
        # 构造多模态消息体，匹配 Anthropic 示例：直接在 content 中使用 source 字段
        if base64_image:
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        else:
            content = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]

        payload = {
            "model": model_name,
            "max_tokens": 20000,
            "stream": False,
            "messages": [
                {"role": "user", "content": content}
            ],
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        }
        # 打印请求体便于调试
        logging.error(f"Request payload: {json.dumps(payload, ensure_ascii=False)}")
        try:
            resp = requests.post(
                url,
                params={"ak": client.api_key},
                headers=headers,
                json=payload,
                timeout=1200
            )
            resp.raise_for_status()
            data = resp.json()
            # 如果顶层有分段 content，则拼接 thinking + text
            segments = data.get("content")
            if isinstance(segments, list):
                thinking = "".join(
                    part.get("thinking", "") 
                    for part in segments 
                    if part.get("type") == "thinking"
                )
                text = "".join(
                    part.get("text", "") 
                    for part in segments 
                    if part.get("type") == "text"
                )
                return thinking + text

            # 否则退回到 choices 里的消息
            msg = data["choices"][0]["message"]
            # 有时 thinking 字段就在 message 里
            thinking = msg.get("thinking", "") or msg.get("reasoning_content", "")
            content = msg.get("content", "")
            return thinking + content
        except HTTPError as e:
            status = e.response.status_code if e.response is not None else 'Unknown'
            body = e.response.text if e.response is not None else ''
            logging.error(f"ByteDance API HTTPError {status}: {e}")
            logging.error(f"Response body: {body}")
            return f"HTTPError {status}: {body}"
        except Exception:
            logging.exception("ByteDance API request failed")
            return ""

    # —— 其他模型走原逻辑 —— #
    for attempt in range(1, MAX_RETRY + 1):
        try:
            if base64_image:
                if 'claude' in model_name:
                    chat_response = client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type":"text","text":prompt},
                                {"type":"image","image_url":{
                                    "url": f"data:image/png;base64,{base64_image}"
                                }}
                            ]
                        }]
                    )
                else:
                    chat_response = client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }]
                    )
            else:
                if 'o3' in model_name or 'o1' in model_name:
                    chat_response = client.chat.completions.create(
                        model=model_name,
                        max_tokens=15000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                else:
                    chat_response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}]
                    )
            # —— 统一返回逻辑，优先拼接 reasoning_content —— #
            choice = chat_response.choices[-1].message
            reasoning = getattr(choice, "reasoning_content", None)
            if reasoning:
                logging.info(f"Reasoning:{reasoning} ;\n\nContent: {choice.content}")
                return reasoning + choice.content
            return choice.content
        except Exception:
            logging.exception(f"LLM call failed on attempt {attempt}/{MAX_RETRY}")
            if attempt == MAX_RETRY:
                return ""
    return ""





async def llama_process(prompt, model_name, address, key, base64_image=None):
    """
    异步包装，根据地址判断使用 AzureOpenAI 还是 OpenAI，
    以及上面同步逻辑中对 thinking 模型的特殊处理。
    """
    if model_name == "gcp-claude37-sonnet-thinking":
        # 构造一个“伪客户端”来携带 base_url 和 api_key（AK）
        class _BDClient:
            def __init__(self, base_url, ak):
                self.base_url = base_url
                self.api_key = ak
        client = _BDClient(address, key)
    else:
        if 'gpt/openapi' in address:
            client = openai.AzureOpenAI(
                azure_endpoint=address,
                api_version="2023-07-01-preview",
                api_key=key,
            )
        else:
            client = OpenAI(api_key=key, base_url=address)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        llama_process_sync,
        prompt,
        model_name,
        client,
        base64_image
    )

async def predict(item_list, sem, model_name, address, key):
    with tqdm(total=len(item_list),
              desc="Generating predictions......",
              leave=True,
              dynamic_ncols=True) as pbar:

        async def run(item):
            async with sem:
                try:
                    resp = await llama_process(
                        item["prompt"], model_name, address, key, item.get("base64_image")
                    )
                except asyncio.TimeoutError:
                    logging.error(f"Predict timeout for prompt: {item['prompt']}")
                    resp = ""
                pbar.update(1)
                return resp

        tasks = [run(item) for item in item_list]
        responses = await asyncio.gather(*tasks)

    torch.cuda.empty_cache()
    for item, resp in zip(item_list, responses):
        item.setdefault("response", []).append(resp)
    return item_list

def save_process(item_list, output_dir, file_name):
    for item in item_list:
        item['have_image'] = bool(item.get('base64_image'))
        item['base64_image'] = ""
    scores = [item.get("score", 0) for item in item_list]
    avg_score = np.mean(scores)
    logging.info(f"Avg score is {avg_score} in {file_name}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, file_name)
    pd.DataFrame(item_list).to_json(out_path, orient="records", lines=True, force_ascii=False)
    logging.info(f"Data has been saved to {out_path}")

    with open(os.path.join(output_dir, "score.txt"), "a") as f:
        f.write(f"{file_name}: {avg_score}\n")

    try:
        for ext in (".jsonl", ".json"):
            ck = os.path.join(output_dir, f"{file_name}_checkpoint{ext}")
            if os.path.exists(ck):
                os.remove(ck)
                logging.info(f"Removed checkpoint file: {ck}")
    except Exception as e:
        logging.warning(f"Failed to remove checkpoint file: {e}")
