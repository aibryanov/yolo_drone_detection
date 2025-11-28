import os
import gdown
import argparse


def download_weights(gdrive_url: str, output_path: str = "models/best.pt"):
    """
    Скачивает файл с Google Drive по ссылке и сохраняет в указанное место.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Скачивание весов в {output_path} ...")
    gdown.download(gdrive_url, output_path, quiet=False)
    print("Скачивание завершено.")

def parse_args():
    parser = argparse.ArgumentParser(description="Скачивание весов YOLO с Google Drive")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Ссылка на файл Google Drive"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/best.pt",
        help="Куда сохранить файл (по умолчанию models/best.pt)"
    )
    return parser.parse_args()

args = parse_args()
download_weights(args.url, args.output)