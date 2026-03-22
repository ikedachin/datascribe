import os
import re
import json
import base64
import uuid
import ast
import time
from copy import deepcopy

from pathlib import Path
from openai import OpenAI
from typing import Union, List, Dict
from PIL import Image

import numpy as np
import tqdm

from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

from commons.utils_msg import msg_debug, msg_error, msg_info, msg_success

class OcrPipeline:
    """推論パイプラインクラス"""
    keys = [
        "id",
        "page",
        "images_path",
        "ocr",
        "topics",
        "objects_content",
        "images",
        "objects",
        "table",
        "figures",
        "photos",
        "generator"
    ]
    def __init__(self,**inference_config: Dict):
        self.inference_config = inference_config
        self.file_name = inference_config.get("file_name")

        self.client = OpenAI(
            base_url=inference_config.get("SERVER_URL"), 
            api_key=inference_config.get("API_KEY")
        )
        self.results = []

        self.output_path = Path(inference_config.get("output_path", "./outputs/")).expanduser().resolve()
        self.file_name = inference_config.get("file_name")

        self.json_files = []
        self.errors = {}

        # check
        print(msg_info("=" * 30))
        print(msg_info("Pipeline initialized."))
        print(msg_info(f"PDF filename: {self.file_name}"))
        print(msg_info(f"Output path set to: {self.output_path}"))
        # print(msg_info(f"Initialized Pipeline with images: {self.images_path}"))
        print(msg_info(f"Inference config: {self.inference_config}"))
        print(msg_info(f"OpenAI Client initialized with base_url: {self.client.base_url}"))
        print(msg_info(f"OpenAI Client initialized with api_key: {self.client.api_key[:5]}..."))
        print(msg_info(f"Full output path: {str(self.output_path)}"))
        print(msg_debug(f"Inference settings: {self.inference_config}"))
        print(msg_info("=" * 30))



    def _infer(self, 
               prompt: str, 
               images: Union[None, list[Path]] = None, 
               previous_result: str = None) -> str:
        """モデルを使って応答を生成"""
        # contentを構築
        if previous_result:
            content = [{"type": "text", "text": prompt}, {"type": "text", "text": previous_result}]
        else:
            content = [{"type": "text", "text": prompt}]
        # print(f"[DEBUG] content: {content}")

        # image_base64: str -> Path -> List(base64), Path -> List(base64), List(Path) -> List(base64)
        if images:
            # 1. まず入力をリストに正規化する
            # 単一入力（パス／文字列／BytesIO／bytes／PIL.Image）を配列に正規化
            single_types = (str, Path, BytesIO, bytes, bytearray)
            try:
                from PIL import Image as PILImage
                single_types = single_types + (PILImage.Image,)
            except Exception:
                PILImage = None

            if isinstance(images, single_types):
                sources = [images]
            else:
                # iterable をリスト化（BytesIO は単一扱いしているためここには来ない）
                sources = list(images)

            # 2. Path化 と エンコード と デコード を一括で行う
            images_base64 = [
                self._encode_image(Path(src) if isinstance(src, str) else src) for src in sources
            ]

        # 各画像をcontentに追加
        if images_base64:
            for img_b64 in images_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    }
                })

        max_retries = int(self.inference_config.get("max_retries", self.inference_config.get("max_retry", 3)))
        wait_seconds = float(self.inference_config.get("wait_seconds", 5))
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.inference_config.get("MODEL_NAME"),
                    messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                    ],
                    max_tokens=self.inference_config.get("max_tokens", 2048),
                    temperature=self.inference_config.get("temperature", 0),
                    top_p=self.inference_config.get("top_p", 1.0),
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"[WARNING] Inference attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise msg_error(f"Failed to generate response: {e}")
                if wait_seconds > 0:
                    tqdm.tqdm.write(msg_info(f"Retrying in {wait_seconds} seconds..."))
                    time.sleep(wait_seconds)

    def _pairing_images(self, batched_images: list) -> list[list]:
        """画像をペアリングします。"""
        paired_images = []
        for i in range(0, len(batched_images)):
            if i != len(batched_images) - 1:
                pair = batched_images[i:i + 2]
                paired_images.append(pair)
        return paired_images

    def batched_infer(self, 
                      key, 
                      prompt: str, 
                      batched_images: list[Union[Path, str]], 
                      previous_result: str = None) -> List[str]:
        """複数の画像に対して並列で推論を実行します。
        Args:
            keys: 推論結果をどのセルに保存するかを指定する
            prompts: 推論に使用するプロンプト文字列とイメージへのパスのリスト。
            [[prompt1, [image1, image2]], [prompt2, [image3]]]
        Returns:
            各プロンプトに対するモデルの推論結果のリスト。
            空のリストが入力された場合は空のリストを返します。
        """
        if not prompt:
            return []
        # print_info("[INFO] Starting parallel inference for prompts...")
        paired_images = self._pairing_images(batched_images)

        max_workers = max(1, len(paired_images))
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            if previous_result:
                # previous_result が単一文字列なら繰り返す、リストならそのまま使う
                if isinstance(previous_result, (str, Path)):
                    prev_list = [previous_result] * len(paired_images)
                else:
                    prev_list = list(previous_result)
                # 長さが合わない場合は短い側に合わせる（必要に応じて例外に変更可）
                if len(prev_list) != len(paired_images):
                    # 不一致時は短い方に合わせる
                    min_len = min(len(prev_list), len(paired_images))
                    paired_images = paired_images[:min_len]
                    prev_list = prev_list[:min_len]

                for imgs, prev in zip(paired_images, prev_list):
                    futures.append(executor.submit(self._infer, prompt, imgs, prev))
            else:
                for imgs in paired_images:
                    futures.append(executor.submit(self._infer, prompt, imgs))

            for fut in futures:
                results.append(fut.result())

            return results

    def _encode_image(self,image_path: Path) -> str:
        """画像をbase64エンコード。Path/str/bytes/BytesIO/PIL.Image を受け取れるように拡張。"""
        # Path または文字列パスが渡された場合
        if isinstance(image_path, (str, Path)):
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        # bytes が渡された場合
        if isinstance(image_path, (bytes, bytearray)):
            return base64.b64encode(bytes(image_path)).decode("utf-8")

        # BytesIO が渡された場合
        if isinstance(image_path, BytesIO):
            return base64.b64encode(image_path.getvalue()).decode("utf-8")

        # PIL Image が渡された場合
        try:
            from PIL import Image as PILImage
            if isinstance(image_path, PILImage.Image):
                buf = BytesIO()
                image_path.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            pass

        # それ以外は例外を投げる
        raise TypeError("Unsupported image type for _encode_image")

    def object_understanding(self, 
                      key: str, 
                      prompt: str, 
                      batched_images: list[Union[Path, str]], 
                      objects_bbox: list[list[int]],
                      previous_results: list[str] = None
                      ) -> List[str]:
        
        """オブジェクト理解処理を実行"""
        print(msg_info("Starting Object Understanding..."))
        # print(f"Input images: {batched_images}")
        paired_images = self._pairing_images(batched_images)

        # objects_bbox を受け取り側で正規化する（文字列で渡される可能性に対応）
        if objects_bbox is None:
            objects_bbox = [[] for _ in paired_images]
        else:
            # objects_bbox 自体が文字列であればパースを試みる
            if isinstance(objects_bbox, str):
                try:
                    objects_bbox = json.loads(objects_bbox)
                except Exception:
                    try:
                        objects_bbox = ast.literal_eval(objects_bbox)
                    except Exception:
                        objects_bbox = []

            # 各要素が文字列であれば個別にパース
            parsed_objects_bbox = []
            for item in objects_bbox:
                if isinstance(item, str):
                    try:
                        parsed = json.loads(item)
                    except Exception:
                        try:
                            parsed = ast.literal_eval(item)
                        except Exception:
                            parsed = []
                    parsed_objects_bbox.append(parsed)
                else:
                    parsed_objects_bbox.append(item)

            objects_bbox = parsed_objects_bbox

            print(msg_debug(f"Parsed objects_bbox: {objects_bbox}"))

        # paired_images と objects_bbox の長さが合わない場合は不足分を空リストで埋める
        if len(objects_bbox) != len(paired_images):
            raise ValueError("Length of objects_bbox and paired_images must match")
        # if len(objects_bbox) != len(paired_images):
        #     raise ValueError("Length of objects_bbox and paired_images must match")
        
        print(msg_debug(f"data nums: {len(paired_images)=}, {len(objects_bbox)=}"))
        
        results = []
        obj_global_index = 0
        for imgs, bboxes in zip(paired_images, objects_bbox):
            # print(f"{imgs=}\n{bboxes=}")
            result_temp = []
            if bboxes != []:
            #     results.append(result_temp)
            # else:
                objects_bbox_temp = []
                for bbox in bboxes:
                    # bbox が文字列ならパースを試みる
                    if isinstance(bbox, str):
                        try:
                            bbox = json.loads(bbox)
                        except Exception:
                            try:
                                bbox = ast.literal_eval(bbox)
                            except Exception:
                                print(msg_error("Could not parse bbox:"), bbox)
                                continue

                    # bbox の形式チェック
                    if not isinstance(bbox, (list, tuple)) or len(bbox) < 6:
                        print(msg_error("bbox has unexpected format:"), bbox)
                        continue

                    if bbox[0] == "image1":
                        img = imgs[0]
                    else:
                        img = imgs[1]

                    objects_bbox_temp.append((img, bbox))
                    # print(f"{objects_bbox_temp=}")

                bbox_images_base_64 = [self._crop_image(img, bbox) for img, bbox in objects_bbox_temp]
                # print(f"len(bbox_images_base_64): {len(bbox_images_base_64)}")

                for i, img_data in enumerate(bbox_images_base_64):
                    # img_data は BytesIO を返すようになった（または None/False）
                    if not img_data:
                        continue

                    prev = None
                    if previous_results and obj_global_index < len(previous_results):
                        prev = previous_results[obj_global_index]

                    formatted_prompt = prompt.format(ocr=prev) if prev is not None else prompt
                    # try:
                    #     print(f"{formatted_prompt[:30]=}\n{formatted_prompt[-30:]=}")
                    # except Exception:
                    #     pass

                    response = self._infer(formatted_prompt, img_data)
                    result_temp.append(response)
            results.append(result_temp)
            # print(result_temp)
            obj_global_index += 1
            # print(f"len(result_temp): {len(result_temp)}, len(results): {len(results)}")
            # print(results)
        # response = self.batched_infer("object_content", prompt, bbox_images_base_64, previous_result)
        # print(msg_debug(f"Response length(object_content): {len(results)=}"))
        return results
    
    def _crop_image(self, image_path: str, box_info: list) -> str:
        """画像をクロップして BytesIO を返す（メモリ上のみで操作）。"""
        img = Image.open(image_path) # 日本語対策
        img_array = np.array(img)
        _, _, s_x, s_y, e_x, e_y = box_info
        s_x, s_y, e_x, e_y = int(s_x * img_array.shape[1] * 0.9 * 0.001), int(s_y * img_array.shape[0] * 0.9 * 0.001), int(e_x * img_array.shape[1] * 1.1 * 0.001), int(e_y * img_array.shape[0] * 1.1 * 0.001)
        cropped = img_array[s_y:e_y, s_x:e_x]
        cropped_img = Image.fromarray(cropped)
        buffered = BytesIO()
        cropped_img.save(buffered, format="PNG")
        buffered.seek(0)
        return buffered
        # return base64.b64encode(buffered.getvalue()).decode("utf-8")
    

    def _save_result(self, key,  images_paths: Union[Path, List[Path]], results: Union[None, list] = None) -> None:
        """結果をJSONLファイルに保存するメソッド"""
        save_path = self.output_path / f"{self.file_name}.jsonl"
        os.makedirs(self.output_path.parent, exist_ok=True)
        for img, result in zip(images_paths, results):
            try:
                page = int(float(Path(img).stem.split("page")[-1]))
            except:
                page = None
                
            with open(save_path, "a", encoding="utf-8") as jsonl_file:
                json_string = json.dumps({
                    key: result,
                    "page": page,
                    "generator": self.inference_config.get("MODEL_NAME", "")
                    }, ensure_ascii=False)
                jsonl_file.write(json_string + "\n")



    def initialize_result(self) -> None:
        """結果辞書を初期化"""
        self.result= {
            "page": None,
            "ocr": "",
            "contents": "",
            "images": [],
            "objects": [],
            "table": [],
            "figures": [],
            "photos": []
        }

    def _error(self, process_name: str, file_path: Path = None, page: int = None) -> None:
        """エラーハンドリング用メソッド"""
        print("[ERROR] An error occurred during processing.")
        # 必要に応じてエラーログを保存する。エラーはself.errorsに格納
        page = page if page is not None else self.result.get("page", None)

        if file_path.stem in self.errors.keys():
            self.errors[file_path.stem].append([process_name, page])
        else:
            self.errors[file_path.stem] = [[process_name, page]]


    def ocr(
        self, 
        prompt: str, 
        images_path: Union[Path, List[Path], None] = None,
        file_path: Path = None,
        page: int = None,
        ) -> str:
        """パイプラインを実行
        Args:
            prompt: テキストプロンプト
            images_path: 単一の画像パス、画像パスのリスト、またはNone
            result: 結果格納用辞書"""
        print(f"[INFO] Starting OCR for page {page}...")
        
        if images_path is not None:
            images_path = [img for img in images_path] if isinstance(images_path, list) else [images_path]
        
        content = self._create_messages(prompt, images_path)

        response = self._inference(
            prompt=prompt, # システムプロンプト
            content=content, 
            **self.inference_config
        )

        if page is not None:
            self.result["page"] = page
        else:
            self.result["page"] = 1

        if response and hasattr(response, 'choices') and response.choices:
            self.result['ocr'] = response.choices[0].message.content
        else:
            print("[ERROR] Invalid response from API")
            self._error("ocr", file_path, page)
        
        self.result["images"] = [str(img) for img in images_path] if images_path else []
        self._save_result(images_path)
        return self.result["ocr"]

    def ocr_batch(
        self,
        prompt: str,
        images_paths: List[Union[Path, List[Path], None]],
        file_path: Path = None,
        start_page: int = 1,
        ) -> List[Dict]:
        """複数ページのOCRをまとめて処理"""
        contents = []
        normalized_images_paths = []
        for images_path in images_paths:
            if images_path is not None:
                images_path = [img for img in images_path] if isinstance(images_path, list) else [images_path]
            normalized_images_paths.append(images_path)
            contents.append(self._create_messages(prompt, images_path))

        responses = self._batched_inference(
            prompts=[prompt] * len(contents),
            contents=contents,
            **self.inference_config,
        )

        results = []
        for idx, (images_path, response) in enumerate(zip(normalized_images_paths, responses)):
            page = start_page + idx
            self.initialize_result()
            self.result["page"] = page

            if response and hasattr(response, 'choices') and response.choices:
                self.result['ocr'] = response.choices[0].message.content
            else:
                print("[ERROR] Invalid response from API")
                self._error("ocr", file_path, page)

            self.result["images"] = [str(img) for img in images_path] if images_path else []
            self._save_result(images_path)
            results.append(deepcopy(self.result))

        return results


    def content_understanding(
        self, 
        prompt: str, 
        images_path: Union[Path, List[Path], None] = None,
        file_path: Path = None,
        page: int = None,
        ) -> str:
        """内容理解パイプラインを実行
        Args:
            prompt: テキストプロンプト
            images_path: 単一の画像パス、画像パスのリスト、またはNone
            result: 結果格納用辞書"""
        if page is None:
            page = self.result["page"]
        print(f"[INFO] Starting Content Understanding for page {page}...")

        if self.result["ocr"] is not None and self.result["ocr"] != "":
            prompt = prompt.format(ocr=self.result["ocr"])
        
        if images_path is not None:
            images_path = [img for img in images_path] if isinstance(images_path, list) else [images_path]
        
        content = self._create_messages(prompt, images_path)

        response = self._inference(
            prompt=prompt, 
            content=content, 
            **self.inference_config
        )
        if page is not None and page != self.result["page"]  :
            self.result["page"] = page

        if response and hasattr(response, 'choices') and response.choices:
            self.result['contents'] = response.choices[0].message.content
        else:
            print("[ERROR] Invalid response from API")
            self._error("contents", file_path, page)
            self.result['contents'] = None
        self.result["images"] = [str(img) for img in images_path] if images_path else []
        self._save_result(images_path)
        return self.result["contents"]

    def objects_detection(
        self, 
        prompt: str, 
        images_path: Union[Path, List[Path], None] = None,
        file_path: Path = None,
        ) -> str:
        """Objects検出パイプラインを実行
        Args:
            prompt: テキストプロンプト
            images: 単一の画像パス、または画像パスのリスト
            result: 結果格納用辞書"""

        print(f"[INFO] Starting Objects Detection...")
        
        # 画像がない場合は空のオブジェクトリストを返す
        if images_path is None:
            print("[INFO] No images provided for object detection, returning empty list.")
            self.result["objects"] = []
            self.result["images"] = []
            self._save_result(images_path)
            return self.result["objects"]
        
        # promptにOCR結果を埋め込む
        if self.result["ocr"] is not None and self.result["ocr"] != "":
            prompt = prompt.format(ocr=self.result["ocr"])
        images_path = [img for img in images_path] if isinstance(images_path, list) else [images_path]
        content = self._create_messages(prompt, images_path)

        response = self._inference(
            prompt=prompt, 
            content=content, 
            **self.inference_config
        )
        # 出力が"[],`"`'"およびアルファベットのみに整形する
        if response and hasattr(response, 'choices') and response.choices:
            # self.result['contents'] = response.choices[0].message.content
            response_text = response.choices[0].message.content.strip()
            response_list = re.sub(r'[^a-zA-Z0-9,\[\]\{\}\":]', '', response_text)
        else:
            print("[ERROR] Invalid response from API")
            self._error("objects", file_path)    
        
        if isinstance(response_list, str):
            # listに変換を試みる
            try:
                response_list = json.loads(response_list)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSONデコードエラー: {e}")
                response_list = []

        self.result["objects"] = response_list
        self.result["images"] = [str(img) for img in images_path]
        self._save_result(images_path)
        return self.result["objects"]

    # def object_understanding(
    #     self, 
    #     prompts_dict: Dict, 
    #     images_path: Union[Path, List[Path], None] = None,
    #     file_path: Path = None,
    #     ) -> str:
    #     """Figure理解パイプラインを実行
    #     Args:
    #         prompt: テキストプロンプト
    #         images_path: 単一の画像パス、画像パスのリスト、またはNone
    #         result: 結果格納用辞書"""
        
    #     print(f"[INFO] Starting Object Understanding...")
        
    #     # 画像がない場合は処理をスキップ
    #     if images_path is None:
    #         print("[INFO] No images provided for object understanding, skipping.")
    #         return []

    #     if self.result['objects'] is not None and len(self.result['objects']) > 0:
    #         print(f"[DEBUG] Objects found: {self.result['objects']}")
    #         for i, obj in enumerate(self.result['objects']):
    #             print(f"[INFO] Processing object: {obj}")
    #             if 'image1' in obj:
    #                 image_path = images_path[0]
    #             elif 'image2' in obj:
    #                 image_path = images_path[1]
    #             else:
    #                 raise ValueError("Object does not contain 'image1' or 'image2' keys.")
    #             print(f"[INFO] Found image2 object path: {image_path}")

    #             crop_img_b64 = self._crop_image(image_path, obj)

    #             # promptにOCR結果を埋め込む
    #             if self.result["ocr"] is not None and self.result["ocr"] != "":
    #                 if 'figure' in obj:
    #                     print(f"[INFO] Using figure prompt for object {i}.")
    #                     prompt = prompts_dict['figure_understanding_prompt'].format(ocr=self.result["ocr"])
    #                 elif 'photo' in obj:
    #                     print(f"[INFO] Using photo prompt for object {i}.")
    #                     prompt = prompts_dict['photo_understanding_prompt'].format(ocr=self.result["ocr"])
    #                 elif 'table' in obj:
    #                     print(f"[INFO] Using table prompt for object {i}.")
    #                     prompt = prompts_dict['table_understanding_prompt'].format(ocr=self.result["ocr"])
    #             else:
    #                 if 'figure' in obj:
    #                     prompt = prompts_dict['figure_understanding_prompt']
    #                 elif 'photo' in obj:
    #                     prompt = prompts_dict['photo_understanding_prompt']
    #                 elif 'table' in obj:
    #                     prompt = prompts_dict['table_understanding_prompt']
    #             content = self._create_messages(prompt, [crop_img_b64])

    #             response = self._inference(
    #                 prompt=prompt, 
    #                 content=content, 
    #                 **self.inference_config
    #             )
    #             if response and hasattr(response, 'choices') and response.choices:
    #                 understand_content = response.choices[0].message.content
    #             else:
    #                 understand_content = ""
    #                 print("[ERROR] Invalid response from API")
    #                 self._error("object_understanding", file_path)

    #             if 'figure' in obj:
    #                 self.result["figures"].append(
    #                     [str(image_path), i,  understand_content]
    #                     )
    #             elif 'photo' in obj:
    #                 self.result["photos"].append(
    #                     [str(image_path), i,  understand_content]
    #                     )
    #             elif 'table' in obj:
    #                 self.result["table"].append(
    #                     [str(image_path), i,  understand_content]
    #                     )
    #             self._save_result(images_path)
    #         return self.result["figures"]
    #     else:
    #         print("[INFO] No objects found for figure understanding.")
    #         return []

    def remove_duplicates(self, prompt: str) -> List[Dict]:
        """重複削除処理"""
        print(f"[INFO] Starting Duplicate Removal...")

        # ファイル名でソートしてページ順を保証
        self.json_files.sort()

        for i in tqdm.tqdm(range(len(self.json_files) -1), desc="Removing duplicates"):
            print(f"[INFO] Removing duplicates between {self.json_files[i]} and {self.json_files[i+1]}...")      
            with open(self.json_files[i], "r") as f:
                json_data1 = json.load(f)
            with open(self.json_files[i+1], "r") as f:
                json_data2 = json.load(f)

            # OCR部分の重複削除
            prompt_ocr = prompt.format(text_a=json_data1.get("ocr", ""), text_b=json_data2.get("ocr", ""))
            content_ocr = [{"type": "text", "text": prompt_ocr}]
            response_ocr = self._inference(
                prompt=prompt_ocr, 
                content=content_ocr, 
                **self.inference_config
            )
            if response_ocr and hasattr(response_ocr, 'choices') and response_ocr.choices:
                response_ocr = response_ocr.choices[0].message.content
                json_data2["ocr"] = response_ocr.strip()
            else:
                print("[ERROR] Invalid response from API")
                self._error("duplicates_ocr", None, i+2)

            # 内容理解部分の重複削除
            prompt_contents = prompt.format(text_a=json_data1.get("contents", ""), text_b=json_data2.get("contents", ""))
            content_contents = [{"type": "text", "text": prompt_contents}]
            response_contents = self._inference(
                prompt=prompt_contents, 
                content=content_contents, 
                **self.inference_config
            )
            if response_contents and hasattr(response_contents, 'choices') and response_contents.choices:
                response_contents = response_contents.choices[0].message.content
                json_data2["contents"] = response_contents.strip()
            else:
                print("[ERROR] Invalid response from API")
                self._error("duplicates_contents", None, i+2)

            # jsonlで保存
            save_file_name = self.json_files[i].stem.split("_page")[0] + ".jsonl"
            print(f"[INFO] Saving deduplicated results to {self.output_path / save_file_name}")
            if i == 0:
                with open(self.output_path / save_file_name, "w") as jsonl_file:
                    json_string = json.dumps(json_data1, ensure_ascii=False)
                    jsonl_file.write(json_string + "\n")
            with open(self.output_path / save_file_name, "a") as jsonl_file:
                json_string = json.dumps(json_data2, ensure_ascii=False)
                jsonl_file.write(json_string + "\n")
