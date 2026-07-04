"""PDFを画像へ変換するためのユーティリティ群。"""

import sys
from pathlib import Path
import fitz  # PyMuPDF
import tqdm
from commons.utils_msg import msg_info, msg_debug, msg_error, msg_success

def convert_pdf_to_images(pdf_path: str, output_root: str, dpi: int = 200) -> list[Path]:
    """1つのPDFをページ単位でPNG画像へ変換する。

    Args:
        pdf_path: 変換対象のPDFファイルへのパス。
        output_root: 生成画像を格納するルートディレクトリ。PDF名のサブフォルダが自動生成される。
        dpi: ページをラスタライズする際の解像度（dpi）。

    Returns:
        生成したPNG画像へのファイルパスをページ順に格納したリスト。

    Raises:
        FileNotFoundError: 指定したPDFが存在しない場合。
        RuntimeError: PyMuPDFがPDFのオープンやレンダリングに失敗した場合。
    """
    images_from_pdf = []
    pdf_path = Path(pdf_path)
    pdf_stem = pdf_path.stem
    pdf_output_dir = Path(output_root) / pdf_stem
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    # print(msg_info(f"Converting: {pdf_path} -> {pdf_output_dir}"))

    doc = fitz.open(pdf_path)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_index in tqdm.tqdm(range(len(doc)), desc=msg_info("Converting PDF pages to images")):
        try:
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)

            img_filename = f"{pdf_stem}_page{page_index + 1:03d}.png"
            img_path = pdf_output_dir / img_filename
            pix.save(img_path)
            images_from_pdf.append(img_path)

            # tqdm.tqdm.write(msg_debug(f"  - page {page_index + 1}/{len(doc)} -> {img_path}"))
        except Exception as e:
            # tqdm.tqdm.write(msg_error(f"Error processing page {page_index + 1}: {e}"), file=sys.stderr)
            raise RuntimeError(f"Failed to process PDF {pdf_path}: {e}")
    doc.close()
    # print(msg_success(f"Done: {pdf_path}\n"))
    return images_from_pdf

if __name__ == "__main__":
    # Example usage
    pdf_file = Path("./test_pdfs/aiplan_g_20251223.pdf")
    output_directory = Path("./imgs/")
    convert_pdf_to_images(pdf_file, output_directory, dpi=200)


def list_pdf_files(pdfs_path) -> list[Path]:
    """単一PDFまたはディレクトリ直下の *.pdf を列挙する。"""
    if pdfs_path is None:
        return []
    pdfs_path = Path(pdfs_path)
    if pdfs_path.is_dir():
        return sorted(pdfs_path.glob("*.pdf"))
    if pdfs_path.is_file() and pdfs_path.suffix.lower() == ".pdf":
        return [pdfs_path]
    return []


def cleanup_output_images(output_dir) -> None:
    """output_dir 配下の画像ファイルを削除する。"""
    output_path = Path(output_dir)
    if not output_path.exists() or not output_path.is_dir():
        return

    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    deleted_count = 0

    for file_path in output_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in image_suffixes:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                tqdm.tqdm.write(msg_error(f"Failed to delete image file: {file_path} ({e})"))

    tqdm.tqdm.write(msg_info(f"Cleaned up {deleted_count} image file(s) in {output_path}"))
