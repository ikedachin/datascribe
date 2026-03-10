import json
import math
import time
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

from commons.utils_msg import msg_debug, msg_error, msg_info, msg_success

class SanitizePipeline:
    """テキストのサニタイゼーション（翻訳による浄化）を行うパイプラインクラス。
    
    日本語テキストを英語に翻訳し、再度日本語に翻訳することで、
    テキストの品質を向上させるパイプラインを提供します。
    """
    
    def __init__(self, settings: Dict):
        """SanitizePipelineを初期化します。
        
        Args:
            settings: パイプラインの設定を含む辞書。
                以下のキーを含むことができます：
                - openrouter: OpenRouterを使用するかどうか (bool)
                - openai_api_key: APIキー (str)
                - openrouter_server_url: サーバーURL (str)
                - openrouter_model_name: モデル名 (str)
                - SERVER_URL: ローカルサーバーURL (str)
                - MODEL_NAME: モデル名 (str)
                - infer_config: 推論設定 (dict)
                - output_path: 出力先パス (str)
                - prompts: プロンプト設定のリスト (List[Dict])
        """
        self.settings = settings
        if settings.get("openrouter", False):
            api_key = settings.get("openai_api_key", "dummy")
            server_url = settings.get("openrouter_server_url", "https://openrouter.ai/api/v1")
            model_name = settings.get("openrouter_model_name", None)
            self.runtime_label = "openrouter"
        else:
            api_key = "dummy"
            server_url = settings.get("SERVER_URL", "http://localhost:8000/v1")
            model_name = settings.get("MODEL_NAME", None)
            self.runtime_label = "local"

        self.inference_config = dict(settings.get("infer_config", {}))
        self.inference_config.update(
            {
                "API_KEY": api_key,
                "SERVER_URL": server_url,
                "MODEL_NAME": model_name,
            }
        )
        self.client = OpenAI(
            base_url=self.inference_config.get("SERVER_URL"),
            api_key=self.inference_config.get("API_KEY"),
        )
        self.output_dir = (
            Path(settings.get("output_path", "./json_output/qa"))
            .expanduser()
            .resolve()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = self._load_prompts(settings.get("prompts", []))
        self.batch_size = settings.get("batch_size", 4)
        self.max_retries = settings.get("max_retries", 3)
        self.wait_seconds = settings.get("wait_seconds", 5)


    def _load_prompts(self, prompts_settings: List[Dict]) -> Dict[str, str]:
        """プロンプトファイルを読み込んで辞書形式で返します。
        
        Args:
            prompts_settings: プロンプトファイルのパス設定のリスト。
                各要素は {key: filepath} の形式の辞書。
        
        Returns:
            キーとプロンプト文字列のマッピング辞書。
        """
        prompts_dict: Dict[str, str] = {}
        for prompt_path_dict in prompts_settings:
            key, prompt_path = list(prompt_path_dict.items())[0]
            if prompt_path and Path(prompt_path).is_file():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompts_dict[key] = f.read()
        return prompts_dict

    def _infer_text(self, prompt: str) -> str:
        """単一のプロンプトに対して推論を実行します。
        
        Args:
            prompt: 推論に使用するプロンプト文字列。
        
        Returns:
            モデルの推論結果のテキスト。
        """
        for _ in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.inference_config.get("MODEL_NAME"),
                    messages=[{"role": "user", "content":  [{"type": "text", "text": prompt}]}],
                    max_tokens=self.inference_config.get("max_tokens", 2048),
                    temperature=self.inference_config.get("temperature", 0),
                    top_p=self.inference_config.get("top_p", 1.0),
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error during inference: {e}. Retrying...")
                time.sleep(self.wait_seconds)
        raise RuntimeError("Max retries exceeded for inference.")

    def _infer_texts(self, prompts: List[str]) -> List[str]:
        """複数のプロンプトに対して並列で推論を実行します。
        
        Args:
            prompts: 推論に使用するプロンプト文字列のリスト。
        
        Returns:
            各プロンプトに対するモデルの推論結果のリスト。
            空のリストが入力された場合は空のリストを返します。
        """
        if not prompts:
            return []
        
        max_workers = len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._infer_text, prompts))

    def tfidf_cosine_similarity(
        self,
        text_a: str,
        text_b: str,
        ngram_range: Tuple[int, int] = (1, 2),
        analyzer: str = "char",
    ) -> float:
        text_a = text_a or ""
        text_b = text_b or ""
        """n-gram TF-IDF と cosine similarity で2文の類似度を算出します。

        Args:
            text_a: 比較対象テキストA
            text_b: 比較対象テキストB
            ngram_range: n-gram の最小・最大 (min_n, max_n)
            analyzer: "char" または "word"

        Returns:
            cosine similarity (0.0 - 1.0)

        1.0: 完全に一致（または非常に類似）
        0.5～0.9: 高い類似性
        0.3～0.5: 中程度の類似性
        0.0～0.3: 低い類似性
        0.0: 全く類似していない、または比較不可能

        特殊ケース
        片方または両方のテキストが空の場合: 0.0
        n-gramが生成できない場合: 0.0
        ベクトルのノルムが0の場合: 0.0

        計算の仕組み
        n-gram抽出: 各テキストから文字または単語のn-gramを抽出

        デフォルト: 1-gram と 2-gram の文字ベース
        TF-IDF重み付け:

        TF (Term Frequency): 各n-gramの出現頻度
        IDF (Inverse Document Frequency): 希少性の重み
        コサイン類似度: 2つのTF-IDFベクトル間の角度から類似度を算出

        """
        def _ngrams(text: str, n: int) -> List[str]:
            if analyzer == "word":
                tokens = text.split()
                if len(tokens) < n:
                    return []
                return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            if len(text) < n:
                return []
            return [text[i : i + n] for i in range(len(text) - n + 1)]

        def _tfidf_vector(text: str, idf: Dict[str, float]) -> Dict[str, float]:
            counts: Dict[str, int] = {}
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for gram in _ngrams(text, n):
                    counts[gram] = counts.get(gram, 0) + 1
            if not counts:
                return {}
            return {term: freq * idf.get(term, 0.0) for term, freq in counts.items()}

        texts = [text_a or "", text_b or ""]
        doc_terms: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        for text in texts:
            counts: Dict[str, int] = {}
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for gram in _ngrams(text, n):
                    counts[gram] = counts.get(gram, 0) + 1
            doc_terms.append(counts)
            for term in counts.keys():
                df[term] = df.get(term, 0) + 1

        if not df:
            return 0.0

        n_docs = len(texts)
        idf = {term: (math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0) for term, freq in df.items()}

        vec_a = _tfidf_vector(text_a, idf)
        vec_b = _tfidf_vector(text_b, idf)
        if not vec_a or not vec_b:
            return 0.0

        dot = sum(vec_a.get(term, 0.0) * vec_b.get(term, 0.0) for term in vec_a.keys())
        norm_a = sum(v * v for v in vec_a.values()) ** 0.5
        norm_b = sum(v * v for v in vec_b.values()) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def sanitize_batch(self, batched_data: list[dict]) -> List[Dict]:
        """バッチ処理でテキストの翻訳サニタイゼーションを実行します。
        
        日本語→英語→日本語の翻訳を通じてテキストを浄化し、
        オプションでリファインと評価を行います。
        
        Args:
            data: 入力データ(辞書)。
        
        Returns:
            処理結果を含む辞書のリスト。
        
        Raises:
            ValueError: jp_en_promptまたはen_jp_promptが設定されていない場合。
        """

        # --------------------
        # プロンプトの取得
        # --------------------
        jp_en_prompt = self.prompts.get("jp_en_prompt", None)
        en_jp_prompt = self.prompts.get("en_jp_prompt", None)
        refine_prompt = self.prompts.get("refine_prompt", None)
        eval_prompt = self.prompts.get("eval_prompt", None)

        if not jp_en_prompt:
            raise ValueError("jp_en_prompt must be set in prompts for translation.") 
        if not en_jp_prompt:
            raise ValueError("en_jp_prompt must be set in prompts for translation.")
        
        # --------------------
        # ターゲットキーの設定
        # --------------------

        if self.settings.get("target_key"):
            target_key = [self.settings["target_key"]]
        else:
            print(msg_info(f"Text file was read. Using 'text' as default key."))
            target_key = ['text']

        print(msg_info(f"Sanitizing key: {target_key}"))
        # 英訳/箇条書き化
        jp_en_prompts = [jp_en_prompt.format(text=data[target_key]) for data in batched_data]
        en_texts = self._infer_texts(jp_en_prompts)
        print(msg_debug(f"英訳/箇条書き: {en_texts[0][:100]}..."))
        # 和訳/箇条書き→日本語復元
        # 和訳/箇条書き→日本語復元
        en_jp_prompts = [en_jp_prompt.format(text=en_text) for en_text in en_texts]
        sanitized_texts = self._infer_texts(en_jp_prompts)
        print(msg_debug(f"和訳/箇条書き→日本語復元: {sanitized_texts[0][:100]}..."))
        # refine
        if refine_prompt:
            refine_prompts = [refine_prompt.format(source_text=data[target_key], sanitized_text=sanitized_text) for data, sanitized_text in zip(batched_data, sanitized_texts)]
            sanitized_texts = self._infer_texts(refine_prompts)
            print(msg_debug(f"再構成: {sanitized_texts[0][:100]}..."))
        # 評価
        if eval_prompt:
            eval_prompts = [eval_prompt.format(source_text=data[target_key], sanitized_text=sanitized_text) for data, sanitized_text in zip(batched_data, sanitized_texts)]
            eval_points = self._infer_texts(eval_prompts)
            print(msg_debug(f"一致度評価結果: {eval_points[0]} points"))

        if eval_prompt:
            for i, (sanitized_text, mid_text, eval_point) in enumerate(zip(sanitized_texts, en_texts, eval_points)):
                batched_data[i][f'eval_{target_key}'] = eval_point
                batched_data[i][f'sanitized_{target_key}'] = sanitized_text
                batched_data[i][f'similarity_{target_key}'] = self.tfidf_cosine_similarity(text_a=batched_data[i][target_key], text_b=sanitized_text)
        else:
            for i, (sanitized_text, mid_text) in enumerate(zip(sanitized_texts, en_texts)):
                batched_data[i][f'sanitized_{target_key}'] = sanitized_text
                batched_data[i][f'similarity_{target_key}'] = self.tfidf_cosine_similarity(text_a=batched_data[i][target_key], text_b=sanitized_text)
        self.save_results(batched_data)

        # self._cache_sanitized(batched_data)

    def save_results(self, batch_data: List[Dict]) -> None:
        """処理結果をJSONL形式で保存します。        
        Args:
            batch_data: 保存する辞書のリスト。
        """
        if not batch_data:
            return
        else:
            for data in batch_data:
                if "book" in data:
                    result_path = self.output_dir / f"sanitized_{data['book']}.jsonl"
                    with open(result_path, "a", encoding="utf-8") as f:
                        for data in batch_data:
                            json.dump(data, f, ensure_ascii=False)
                            f.write("\n")

    # def _cache_sanitized(self, batched_data, data_path_stem: str) -> None:
    #     sanitized_list_path = self.output_dir / "sanitized.jsonl"
    #     existing_stems = set()
    #     if sanitized_list_path.exists():
    #         with open(sanitized_list_path, "r", encoding="utf-8") as f:
    #             for line in f:
    #                 line = line.strip()
    #                 if line:
    #                     existing_stems.add(line)
    #     if data_path_stem not in existing_stems:
    #         with open(sanitized_list_path, "a", encoding="utf-8") as f:
    #             f.write(str(data_path_stem) + "\n")



