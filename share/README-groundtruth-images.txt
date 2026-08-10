Remaining cached ground-truth images for SD v1 and SD v2

File names correspond to row positions in the associated
parquet files: dl_000.jpg corresponds to row 0, dl_001.jpg to row 1, and so on.

Some source URLs were unavailable when the images were downloaded. For those
failed downloads, the downloader stored a black 256x256 placeholder, which was
skipped during ground-truth generation.

Archive statistics:
- SD v1: 500 files; 416 available images and 84 black placeholders
- SD v2: 500 files; 362 available images and 138 black placeholders